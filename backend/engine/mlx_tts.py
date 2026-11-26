"""YouClone Studio Speech Engine: Stage 1 (Orpheus-3B Expressive Actor + Whisper Gate)
+ Stage 2 (Seed-VC 44.1 kHz F0-Conditioned Diffusion + BigVGAN Voice Lock)."""

from __future__ import annotations

import gc
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly

from ..models import MODELS
from . import dsp
from .seed_vc_bridge import SeedVCBridge

try:
    import mlx.core as mx
    import mlx_whisper
    from mlx_audio.tts import load
    from . import orpheus_lora

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    mlx_whisper = None  # type: ignore[assignment]
    load = None  # type: ignore[assignment]
    orpheus_lora = None  # type: ignore[assignment]
    HAS_MLX = False

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

# Clean emojis and markdown formatting from pasted scripts while keeping natural punctuation
_MARKDOWN_RE = re.compile(r"[*_`~#]+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)
_PAUSE_TAG_RE = re.compile(r"\[pause\s+([0-9]+(?:\.[0-9]+)?)s?\]", re.I)

# Translate legacy YouClone bracket tags cleanly into Orpheus-3B native paralinguistic cues
_TAG_MAP = [
    (re.compile(r"\[sigh\]", re.I), "<sigh>"),
    (re.compile(r"\[chuckle\]", re.I), "<chuckle>"),
    (re.compile(r"\[(?:laughter|laugh)\]", re.I), "<laugh>"),
    (re.compile(r"\[(?:gasp|breath)\]", re.I), "<gasp>"),
    (re.compile(r"\[yawn\]", re.I), "<yawn>"),
    # Strip non-paralinguistic style bracket tags cleanly so Orpheus never reads them aloud
    (re.compile(r"\[(?:surprised|shocked|amazed|fear|fearful|scary|suspense|whisper|happy|joy|joyful|excited|furious|angry|urgent|serious|sad|somber|curious|dramatic|calm|explain|warm)\]", re.I), ""),
]

# Sentence splitter that keeps ellipses ("..."), "?!", and acronyms intact
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9<\"'“‘])")


def _whisper_snapshot() -> str:
    hub = WORKSPACE_ROOT / ".voxm5" / "huggingface" / "hub" / "models--mlx-community--whisper-large-v3-turbo" / "snapshots"
    if hub.is_dir():
        snaps = sorted([p for p in hub.iterdir() if p.is_dir()])
        if snaps:
            return str(snaps[-1])
    return "mlx-community/whisper-large-v3-turbo"


# Built-in single-token Orpheus-3B silent speaker tags (never spoken aloud as script words)
_NATIVE_ORPHEUS_VOICES = frozenset({"leo", "dan", "tara", "leah", "jess", "mia", "zac", "zoe"})

_NUM_WORD_MAP = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
}


def _clean_words(s: str) -> list[str]:
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&", " and ")
    s = s.lower().replace("'", "").replace("’", "")
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    # Normalize common ASR punctuation/acronym spacing (e.g. "p.m." -> "pm", "ps 2" -> "ps2")
    s = re.sub(r"\bp\s*m\b", "pm", s)
    s = re.sub(r"\ba\s*m\b", "am", s)
    s = re.sub(r"\bps\s*([2-5])\b", r"ps\1", s)
    s = re.sub(r"\bt\s*v\b", "tv", s)
    s = re.sub(r"\bg\s*t\s*a\b", "gta", s)
    out: list[str] = []
    for tok in s.split():
        out.append(_NUM_WORD_MAP.get(tok, tok))
    return out


def _word_overlap_score(expected: str, got: str) -> float:
    ew = _clean_words(expected)
    gw = _clean_words(got)
    if not ew:
        return 1.0
    n, m = len(ew), len(gw)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if ew[i] == gw[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[n][m]
    recall = lcs / max(1, n)
    prec = lcs / max(1, m)
    return 2.0 * recall * prec / max(1e-6, recall + prec)


def _beat_needs_reroll(expected: str, got: str) -> tuple[float, bool]:
    """Evaluate Whisper transcript fidelity against expected beat text.
    Flags mispronunciations, dropped openings, or hallucinated/leaked leading words."""
    ew = _clean_words(expected)
    gw = _clean_words(got)
    if not ew:
        return 1.0, False
    if not gw:
        return 0.0, True
    score = _word_overlap_score(expected, got)
    # Leading-word check: catch dropped/replaced openings like "When you" -> "Soon in" or "Climb" -> "Carn"
    lead_ok = (gw[0] == ew[0]) or (
        len(ew[0]) >= 4 and len(gw[0]) >= 4 and (ew[0][:3] == gw[0][:3] or ew[0] in gw[0] or gw[0] in ew[0])
    )
    if not lead_ok:
        score = max(0.0, score - 0.14)
    # Also penalize unexpected length drift (>35% fewer or >35% extra words)
    ratio = len(gw) / max(1, len(ew))
    if ratio < 0.72 or ratio > 1.32:
        score = max(0.0, score - 0.12)
    needs_reroll = (score < 0.90) or (not lead_ok)
    return score, needs_reroll


def _unload_whisper() -> None:
    """Release mlx_whisper's cached Metal model completely before loading Orpheus or Seed-VC."""
    mod = sys.modules.get("mlx_whisper.transcribe")
    if mod is not None and hasattr(mod, "ModelHolder"):
        mod.ModelHolder.model = None
        mod.ModelHolder.model_path = None
    gc.collect()
    if mx is not None:
        mx.synchronize()
        mx.clear_cache()


def _normalize_script_for_orpheus(raw_text: str) -> str:
    txt = raw_text.replace("\r\n", "\n").replace("…", "...").replace("—", " - ").replace("–", " - ")
    txt = txt.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    txt = _EMOJI_RE.sub("", txt)
    txt = _MARKDOWN_RE.sub("", txt)
    for pat, repl in _TAG_MAP:
        txt = pat.sub(repl, txt)
    # Keep true acronyms uppercase, normalize ALL-CAPS emphasis words cleanly (incl. possessives like VERCETTI'S)
    acronyms = {
        "I", "GTA", "NPC", "PC", "PS5", "PS4", "PS3", "PS2", "PS", "AI", "API", "GPU", "CPU",
        "FBI", "CIA", "USA", "UK", "USB", "RAM", "SSD", "FPS", "DLC", "HDR", "LED", "CEO",
        "DIY", "FAQ", "HUD", "RTX", "VR", "AR", "TV", "ID", "IQ", "DJ", "OK", "BOOM", "AM", "PM",
    }

    def _fix_caps(match: re.Match) -> str:
        w = match.group(0)
        if w in acronyms:
            return w
        if w.endswith("'S"):
            stem = w[:-2]
            return (stem if stem in acronyms else stem.lower()) + "'s"
        return w.lower()

    txt = re.sub(r"\b[A-Z]{2,}(?:'S)?\b", _fix_caps, txt)
    # Restore natural capitalization at start of lines and after sentence-ending punctuation (not mid-sentence "...")
    txt = re.sub(
        r"(^|\n|(?<!\.\.)[.!?]\s+[\"'(\[]*)([a-z])",
        lambda m: m.group(1) + m.group(2).upper(),
        txt,
    )
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()


def split_orpheus_beats(raw_text: str, base_pause: float = 0.30, max_words: int = 22) -> list[tuple[str, float]]:
    """Group script into natural 4-9 second dramatic beats (12-30 words each) so Orpheus-3B
    delivers connected semantic arcs without ever drifting or losing prosody continuity."""
    cleaned = _normalize_script_for_orpheus(raw_text)
    if not cleaned:
        return []

    # Respect explicit [pause Xs] tags and paragraph breaks
    raw_paragraphs = [p.strip() for p in re.split(r"\n+", cleaned) if p.strip()]
    sentences: list[tuple[str, float]] = []
    for para in raw_paragraphs:
        # Handle inline [pause X.Xs]
        parts = _PAUSE_TAG_RE.split(para)
        i = 0
        while i < len(parts):
            chunk = parts[i].strip()
            explicit_pause = float(parts[i + 1]) if (i + 1 < len(parts) and parts[i + 1]) else 0.0
            i += 2
            if not chunk:
                if sentences and explicit_pause > 0:
                    prev_t, prev_p = sentences[-1]
                    sentences[-1] = (prev_t, max(prev_p, explicit_pause))
                continue
            sub_sents = [s.strip() for s in _SENT_SPLIT_RE.split(chunk) if s.strip()]
            for j, s in enumerate(sub_sents):
                is_last = j == len(sub_sents) - 1
                p_after = explicit_pause if (is_last and explicit_pause > 0) else (base_pause if is_last else 0.0)
                sentences.append((s, p_after))

    # Merge short consecutive sentences into tight dramatic arcs (<= max_words)
    beats: list[tuple[str, float]] = []
    cur_text = ""
    cur_words = 0
    for s_text, s_pause in sentences:
        w_cnt = len(_clean_words(s_text))
        # Always split before a dramatic pivot or if adding exceeds max_words
        is_pivot = bool(re.match(r"^(?:but\.\.\.|so if you|imagine waiting|and please)", s_text, re.I))
        if cur_text and (cur_words + w_cnt > max_words or (is_pivot and cur_words >= 12)):
            beats.append((cur_text, base_pause))
            cur_text = s_text
            cur_words = w_cnt
        else:
            cur_text = f"{cur_text} {s_text}".strip() if cur_text else s_text
            cur_words += w_cnt
        if s_pause > base_pause + 0.08:
            beats.append((cur_text, s_pause))
            cur_text = ""
            cur_words = 0
    if cur_text:
        beats.append((cur_text, 0.10))

    return beats


def _trim_and_fade_24k(audio: np.ndarray, sr: int = 24000) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return audio
    thresh = max(1e-4, peak * 0.015)
    idx = np.flatnonzero(np.abs(audio) >= thresh)
    if idx.size:
        s = max(0, int(idx[0]) - int(0.025 * sr))
        e = min(audio.size, int(idx[-1]) + int(0.055 * sr))
        audio = audio[s:e]
    fade = min(int(0.012 * sr), audio.size // 8)
    if fade > 2:
        r = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio = audio.copy()
        audio[:fade] *= r
        audio[-fade:] *= r[::-1]
    return audio.astype(np.float32)


class Engine:
    def __init__(self):
        self.models: dict[str, tuple[object, float]] = {}
        self.active_lora: str | None = None
        self.idle = max(0, float(os.getenv("VOXM5_MODEL_IDLE", "900")))
        self.seed_vc = SeedVCBridge()

    @property
    def data_dir(self) -> Path:
        return Path(os.environ["VOXM5_DATA"]).resolve() if "VOXM5_DATA" in os.environ else Path(".").resolve()

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    def check_model_cached(self, key: str = "orpheus_seedvc") -> bool:
        repo = MODELS.get(key) or MODELS["orpheus_seedvc"]
        try:
            snap = snapshot_download(repo, allow_patterns=["config.json", "*.safetensors"], local_files_only=True)
            return any(Path(snap).rglob("*.safetensors"))
        except Exception:
            return False

    def download_model(self, key: str = "orpheus_seedvc") -> str:
        repo = MODELS.get(key) or MODELS["orpheus_seedvc"]
        snapshot_download(repo, ignore_patterns=["*.md", "*.png", "*.jpg"], max_workers=4)
        return repo

    def unload_all(self):
        """Release both Orpheus MLX weights and Seed-VC MPS daemon before heavy training."""
        self.models.clear()
        self.active_lora = None
        self.seed_vc.unload()
        gc.collect()
        mx.clear_cache()

    def evict(self):
        expired = [
            key
            for key, (_, last) in self.models.items()
            if self.idle and time.monotonic() - last > self.idle
        ]
        for key in expired:
            del self.models[key]
            if key == "orpheus_seedvc":
                self.active_lora = None
        if expired:
            gc.collect()
            mx.clear_cache()

    def load_orpheus(self):
        key = "orpheus_seedvc"
        mx.set_default_device(mx.gpu)
        try:
            mx.set_cache_limit(256 * 1024 * 1024)
        except Exception:
            pass
        if key not in self.models:
            # Ensure Seed-VC MPS daemon is not holding GPU memory while Orpheus loads
            self.seed_vc.unload()
            model = load(MODELS[key])
            mx.eval(model.parameters())
            mx.synchronize()
            self.active_lora = None
        else:
            model, _ = self.models[key]
        self.models[key] = (model, time.monotonic())
        return model

    def load(self, key: str = "orpheus_seedvc"):
        return self.load_orpheus()

    def _activate_orpheus_adapter(self, model, adapter_id: str | None):
        target = adapter_id or None
        if target == self.active_lora:
            return
        if target is None:
            orpheus_lora.strip_orpheus_lora(model)
            model.eval()
            mx.eval(model.model.parameters())
            self.active_lora = None
            return
        weights_file = self.adapters_dir / f"{target}.safetensors"
        if not weights_file.is_file():
            orpheus_lora.strip_orpheus_lora(model)
            self.active_lora = None
            return
        orpheus_lora.load_orpheus_lora(model, self.adapters_dir, target)
        self.active_lora = target

    def speaker_embedding(self, source):
        return None

    def _unload_orpheus(self):
        self.models.clear()
        self.active_lora = None
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def _generate_orpheus_beat(
        self,
        model,
        *,
        b_text: str,
        voice_name: str,
        eff_temp: float,
        eff_top_p: float,
        eff_rep: float,
        max_toks: int,
        b_seed: int,
        sr_o: int = 24000,
    ) -> np.ndarray | None:
        mx.random.seed(b_seed)
        chunks = []
        for res in model.generate(
            text=b_text,
            voice=voice_name,
            temperature=eff_temp,
            top_p=eff_top_p,
            repetition_penalty=eff_rep,
            max_tokens=max_toks,
            split_pattern="<<<NO_SPLIT>>>",
            verbose=False,
        ):
            chunks.append(np.asarray(res.audio, dtype=np.float32))
        mx.synchronize()
        mx.clear_cache()
        if not chunks:
            return None
        return _trim_and_fade_24k(np.concatenate(chunks), sr_o)

    def synthesize_studio_speech(
        self,
        *,
        text: str,
        voice_profile: dict,
        temperature: float = 0.58,
        top_p: float = 0.86,
        repetition_penalty: float = 1.18,
        stability: float = 55.0,
        similarity_boost: float = 75.0,
        style_exaggeration: float = 45.0,
        period_pause: float = 0.32,
        diffusion_steps: int | None = None,
        seed: int = 202,
        progress_cb: Callable[[dict], None] | None = None,
    ) -> np.ndarray:
        """Full 2-stage studio voice cloning & expressive synthesis.

        Stage 1a: Orpheus-3B (+ voice QLoRA for Pro voices, or matched actor for Instant voices)
                  using a native single-token speaker prefix (never leaks voice names into speech).
        Stage 1b: Zero-GPU-overlap Whisper-large-v3-turbo verbatim audit + targeted re-roll
                  on any misread beat (Orpheus unloaded during Whisper, Whisper unloaded before Stage 2).
        Stage 2:  Seed-VC 44.1 kHz F0-Conditioned Diffusion + BigVGAN (fine-tuned `ft_model.pth`
                  for Pro voices, or zero-shot 44.1 kHz for Instant voices), resampled to 48 kHz.
        """
        emit = progress_cb or (lambda _ev: None)
        beats = split_orpheus_beats(text, base_pause=float(np.clip(period_pause, 0.15, 1.5)))
        if not beats:
            raise ValueError("Enter text to synthesize.")

        # Zero GPU overlap: release Seed-VC & Whisper before Stage 1 Orpheus MLX runs
        self.seed_vc.unload()
        _unload_whisper()
        model = self.load_orpheus()
        adapter_id = voice_profile.get("orpheus_adapter")
        # Always enforce a native single-token Orpheus speaker prefix so custom voice names
        # can never leak into spoken audio while preserving full QLoRA prosody.
        raw_voice = (voice_profile.get("orpheus_voice") or "leo").strip().lower()
        voice_name = raw_voice if raw_voice in _NATIVE_ORPHEUS_VOICES else "leo"
        self._activate_orpheus_adapter(model, adapter_id)

        # Protect against legacy high Qwen/OmniVoice defaults (0.72 / 0.90 / 1.10) that make Orpheus slur
        base_temp = 0.58 if abs(float(temperature) - 0.72) < 1e-3 else float(temperature)
        temp_adj = ((50.0 - stability) / 100.0) * 0.10 + ((style_exaggeration - 45.0) / 100.0) * 0.06
        eff_temp = float(np.clip(base_temp + temp_adj, 0.46, 0.66))
        eff_top_p = float(np.clip(top_p, 0.82, 0.88))
        eff_rep = float(np.clip(repetition_penalty, 1.15, 1.22))
        eff_cfg = float(np.clip(0.55 + (similarity_boost / 100.0) * 0.30, 0.55, 0.88))
        eff_steps = int(diffusion_steps or voice_profile.get("diffusion_steps") or 35)
        eff_steps = max(15, min(50, eff_steps))

        sr_o = 24000
        beat_wavs: list[np.ndarray] = []

        for idx, (b_text, _pause_s) in enumerate(beats):
            emit({
                "type": "progress",
                "current": idx + 1,
                "total": len(beats),
                "stage": "acting",
                "emotion": f"Stage 1/2 · Acting beat {idx + 1}/{len(beats)}",
            })
            word_cnt = len(_clean_words(b_text))
            min_dur = max(0.75, word_cnt * 0.17)
            max_dur = max(4.8, word_cnt * 0.60)
            max_toks = min(1050, max(560, int(word_cnt * 52)))

            best_wav = None
            best_score = -1.0
            for attempt in range(3):
                b_seed = (int(seed) + idx * 41 + attempt * 17) % (2**31 - 1)
                prompt_text = b_text
                if attempt >= 1 and "GTA 6" in prompt_text:
                    prompt_text = prompt_text.replace("GTA 6", "G.T.A. 6")

                wav = self._generate_orpheus_beat(
                    model,
                    b_text=prompt_text,
                    voice_name=voice_name,
                    eff_temp=eff_temp + (0.02 if attempt >= 2 else 0.0),
                    eff_top_p=eff_top_p,
                    eff_rep=eff_rep,
                    max_toks=max_toks,
                    b_seed=b_seed,
                    sr_o=sr_o,
                )
                if wav is None:
                    continue
                dur = wav.size / float(sr_o)
                rms_val = float(np.sqrt(np.mean(wav * wav))) if wav.size else 0.0
                active_ratio = float(np.mean(np.abs(wav) > max(1e-3, rms_val * 0.2))) if wav.size else 0.0

                dur_ok = min_dur <= dur <= max_dur
                target_dur = max(1.2, word_cnt * 0.34)
                dur_closeness = max(0.0, 1.0 - abs(dur - target_dur) / max(1.5, target_dur))
                score = dur_closeness + (0.5 if dur_ok else -0.5) + (0.3 if active_ratio >= 0.25 else -0.4)
                if score > best_score:
                    best_score = score
                    best_wav = wav
                if dur_ok and rms_val >= 0.012 and active_ratio >= 0.25:
                    break

            if best_wav is None:
                raise ValueError(f"Failed to synthesize beat {idx + 1}.")
            beat_wavs.append(best_wav)

        # Release Orpheus-3B completely BEFORE running Whisper accuracy verification (0 GPU overlap!)
        self._unload_orpheus()

        # Stage 1b: Zero-GPU-overlap Whisper verbatim accuracy gate + targeted re-roll
        try:
            snap = _whisper_snapshot()
            emit({
                "type": "progress",
                "current": len(beats),
                "total": len(beats),
                "stage": "verify",
                "emotion": "Stage 1/2 · Verifying script accuracy",
            })
            flagged: list[tuple[int, float]] = []
            for idx, ((b_text, _pause_s), wav) in enumerate(zip(beats, beat_wavs)):
                if len(_clean_words(b_text)) < 2:
                    continue
                wav_16k = resample_poly(wav, 2, 3).astype(np.float32)
                asr = mlx_whisper.transcribe(
                    wav_16k,
                    path_or_hf_repo=snap,
                    language="en",
                    condition_on_previous_text=False,
                ).get("text", "").strip()
                sc, needs_reroll = _beat_needs_reroll(b_text, asr)
                if needs_reroll:
                    flagged.append((idx, sc))
            _unload_whisper()

            if flagged:
                # Re-roll ONLY the flagged beats with 2 fresh seeds each (still 0 GPU overlap!)
                model = self.load_orpheus()
                self._activate_orpheus_adapter(model, adapter_id)
                reroll_candidates: dict[int, list[np.ndarray]] = {}
                for pos, (idx, _old_sc) in enumerate(flagged, 1):
                    b_text, _pause_s = beats[idx]
                    emit({
                        "type": "progress",
                        "current": pos,
                        "total": len(flagged),
                        "stage": "verify",
                        "emotion": f"Stage 1/2 · Polishing pronunciation ({pos}/{len(flagged)})",
                    })
                    word_cnt = len(_clean_words(b_text))
                    max_toks = min(1050, max(560, int(word_cnt * 52)))
                    cands: list[np.ndarray] = []
                    for extra in range(2):
                        b_seed = (int(seed) + idx * 97 + (extra + 5) * 131) % (2**31 - 1)
                        cand_wav = self._generate_orpheus_beat(
                            model,
                            b_text=b_text,
                            voice_name=voice_name,
                            eff_temp=max(0.48, eff_temp - 0.04 * (extra + 1)),
                            eff_top_p=min(eff_top_p, 0.85),
                            eff_rep=eff_rep,
                            max_toks=max_toks,
                            b_seed=b_seed,
                            sr_o=sr_o,
                        )
                        if cand_wav is not None:
                            cands.append(cand_wav)
                    reroll_candidates[idx] = cands
                self._unload_orpheus()

                # Score re-rolled candidates with Whisper and keep strictly the highest-accuracy take
                for idx, old_sc in flagged:
                    b_text, _ = beats[idx]
                    best_sc = old_sc
                    best_w = beat_wavs[idx]
                    for cand_wav in reroll_candidates.get(idx, []):
                        wav_16k = resample_poly(cand_wav, 2, 3).astype(np.float32)
                        asr = mlx_whisper.transcribe(
                            wav_16k,
                            path_or_hf_repo=snap,
                            language="en",
                            condition_on_previous_text=False,
                        ).get("text", "").strip()
                        sc, _ = _beat_needs_reroll(b_text, asr)
                        if sc > best_sc:
                            best_sc = sc
                            best_w = cand_wav
                    beat_wavs[idx] = best_w
                _unload_whisper()
        except Exception as exc:
            print(f"[YouClone] Whisper verification skipped: {exc}", flush=True)
            _unload_whisper()
            self._unload_orpheus()

        # Assemble normalized Stage 1 audio
        parts: list[np.ndarray] = []
        for (_b_text, pause_s), best_wav in zip(beats, beat_wavs):
            rms = float(np.sqrt(np.mean(best_wav * best_wav)) + 1e-8)
            gain = min(2.0, max(0.55, 0.14 / rms))
            norm_wav = np.clip(best_wav * gain, -0.96, 0.96).astype(np.float32)
            parts.append(norm_wav)
            if pause_s > 0:
                parts.append(np.zeros((int(pause_s * sr_o),), dtype=np.float32))

        stage1_audio = np.concatenate(parts).astype(np.float32)

        # Stage 2: Seed-VC 44.1 kHz F0-Conditioned BigVGAN Voice Lock (sole owner of Metal GPU)
        ref_audio = voice_profile.get("ref_audio")
        if ref_audio and Path(ref_audio).is_file() and self.seed_vc.is_available():
            mode_tag = "PVC" if voice_profile.get("seedvc_checkpoint") else "Instant"
            emit({
                "type": "progress",
                "current": len(beats),
                "total": len(beats),
                "stage": "voice_lock",
                "emotion": f"Stage 2/2 · Locking {voice_profile.get('name', 'voice')} timbre ({mode_tag} 44.1 kHz)",
            })
            try:
                with tempfile.TemporaryDirectory(dir=self.data_dir) as tmpdir:
                    src_wav = Path(tmpdir) / "stage1.wav"
                    out_vc = Path(tmpdir) / "stage2_44k.wav"
                    sf.write(str(src_wav), stage1_audio, sr_o, subtype="FLOAT")
                    self.seed_vc.convert(
                        source_path=src_wav,
                        target_path=ref_audio,
                        output_path=out_vc,
                        checkpoint_path=voice_profile.get("seedvc_checkpoint"),
                        config_path=voice_profile.get("seedvc_config"),
                        diffusion_steps=eff_steps,
                        cfg_rate=eff_cfg,
                        auto_f0_adjust=True,
                    )
                    vc_sig, vc_sr = sf.read(str(out_vc), dtype="float32")
                    if vc_sig.ndim > 1:
                        vc_sig = vc_sig.mean(axis=1)
                    return dsp.resample(vc_sig, vc_sr).astype(np.float32)
            finally:
                # Release MPS buffers cleanly after conversion so next Stage 1 starts with 0 MB contention
                self.seed_vc.unload()

        return dsp.resample(stage1_audio, sr_o).astype(np.float32)

    def convert_speech_to_speech(
        self,
        source_wav: Path,
        voice_profile: dict,
        *,
        diffusion_steps: int = 35,
        cfg_rate: float = 0.75,
    ) -> np.ndarray:
        """Direct 44.1 kHz Seed-VC Speech-to-Speech conversion for Voice Changer tool."""
        ref_audio = voice_profile.get("ref_audio")
        if not ref_audio or not Path(ref_audio).is_file():
            raise ValueError("Selected voice profile has no reference audio.")
        with tempfile.TemporaryDirectory(dir=self.data_dir) as tmpdir:
            out_vc = Path(tmpdir) / "s2s_44k.wav"
            self.seed_vc.convert(
                source_path=source_wav,
                target_path=ref_audio,
                output_path=out_vc,
                checkpoint_path=voice_profile.get("seedvc_checkpoint"),
                config_path=voice_profile.get("seedvc_config"),
                diffusion_steps=diffusion_steps,
                cfg_rate=cfg_rate,
                auto_f0_adjust=True,
            )
            vc_sig, vc_sr = sf.read(str(out_vc), dtype="float32")
            if vc_sig.ndim > 1:
                vc_sig = vc_sig.mean(axis=1)
            return dsp.resample(vc_sig, vc_sr).astype(np.float32)


if not HAS_MLX:
    from .cuda_tts import CudaEngine as Engine  # type: ignore[assignment]
