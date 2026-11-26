"""Windows / Linux NVIDIA CUDA (+ CPU fallback) Speech Engine for YouClone.

Mirror of `mlx_tts.Engine` designed to run cleanly on 4 GB+ NVIDIA GPUs (e.g. RTX 2050 4 GB):
  - Stage 1a: Orpheus-3B Q4_K_M GGUF via `llama-cpp-python` (CUDA, ~2.1 GB VRAM), unloaded
              before SNAC 24 kHz decoding to guarantee zero VRAM overlap.
  - Stage 1b: Verbatim accuracy audit via `faster-whisper` (`small.en` int8 on CPU — 0 MB VRAM).
  - Stage 2:  Seed-VC 44.1 kHz F0-Conditioned Voice Lock (`fp16` on CUDA via `SeedVCBridge`).
"""

from __future__ import annotations

import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download, snapshot_download
from scipy.signal import resample_poly

from ..models import ORPHEUS_GGUF_FILE, ORPHEUS_GGUF_REPO, WHISPER_FASTER
from . import dsp
from .seed_vc_bridge import SeedVCBridge

# Orpheus-3B Llama-3 special control token IDs
_SOH_TOKEN = 128259
_BOS_TOKEN = 128000
_EOT_TOKEN = 128009
_EOH_TOKEN = 128260
_SOA_TOKEN = 128261
_SOS_TOKEN = 128257
_EOS_TOKEN = 128258
_EOA_TOKEN = 128262
_AUDIO_CODE_OFFSET = 128266
_NATIVE_ORPHEUS_VOICES = frozenset({"leo", "dan", "tara", "leah", "jess", "mia", "zac", "zoe"})


def _extract_snac_codes(token_ids: list[int]) -> list[list[int]] | None:
    """Convert raw Orpheus audio token IDs into 3-level SNAC 24 kHz codebook streams."""
    raw = [t for t in token_ids if t >= _AUDIO_CODE_OFFSET]
    n_frames = len(raw) // 7
    if n_frames < 2:
        return None
    raw = raw[: n_frames * 7]
    c0: list[int] = []
    c1: list[int] = []
    c2: list[int] = []
    for i in range(n_frames):
        base = i * 7
        vals = [raw[base + k] - _AUDIO_CODE_OFFSET - 4096 * k for k in range(7)]
        if any(v < 0 or v >= 4096 for v in vals):
            continue
        c0.append(vals[0])
        c1.extend((vals[1], vals[4]))
        c2.extend((vals[2], vals[3], vals[5], vals[6]))
    if not c0:
        return None
    return [c0, c1, c2]


def _decode_snac_24k(code_lists: list[list[list[int]]]) -> list[np.ndarray]:
    """Decode batched SNAC 24 kHz codebooks using PyTorch SNAC (CPU or CUDA, ~160 MB)."""
    import torch
    from snac import SNAC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    waves: list[np.ndarray] = []
    try:
        with torch.no_grad():
            for codes in code_lists:
                tensors = [torch.tensor(layer, dtype=torch.long, device=device).unsqueeze(0) for layer in codes]
                wav = snac_model.decode(tensors).squeeze().float().cpu().numpy()
                waves.append(wav.astype(np.float32))
    finally:
        del snac_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return waves


def _clear_cuda():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


class CudaEngine:
    """Drop-in Windows/Linux CUDA + CPU engine matching `mlx_tts.Engine`'s public API."""

    def __init__(self):
        self.models: dict[str, tuple[object, float]] = {}
        self.active_lora: str | None = None
        self.idle = max(0, float(os.getenv("VOXM5_MODEL_IDLE", "900")))
        self.seed_vc = SeedVCBridge()
        self._llm = None

    @property
    def data_dir(self) -> Path:
        return Path(os.environ["VOXM5_DATA"]).resolve() if "VOXM5_DATA" in os.environ else Path(".").resolve()

    @property
    def adapters_dir(self) -> Path:
        return self.data_dir / "adapters"

    def _resolve_gguf_path(self, local_only: bool = True) -> Path | None:
        try:
            path = hf_hub_download(
                repo_id=ORPHEUS_GGUF_REPO,
                filename=ORPHEUS_GGUF_FILE,
                local_files_only=local_only,
            )
            return Path(path)
        except Exception:
            return None

    def check_model_cached(self, key: str = "orpheus_seedvc") -> bool:
        return self._resolve_gguf_path(local_only=True) is not None

    def download_model(self, key: str = "orpheus_seedvc") -> str:
        hf_hub_download(repo_id=ORPHEUS_GGUF_REPO, filename=ORPHEUS_GGUF_FILE)
        snapshot_download("hubertsiuzdak/snac_24khz", ignore_patterns=["*.md", "*.png", "*.jpg"])
        return ORPHEUS_GGUF_REPO

    def unload_all(self):
        self._unload_llm()
        self.seed_vc.unload()
        _clear_cuda()

    def evict(self):
        if self._llm is not None and self.idle:
            _, last = self.models.get("orpheus_seedvc", (None, 0.0))
            if time.monotonic() - last > self.idle:
                self._unload_llm()

    def _unload_llm(self):
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception:
                pass
            self._llm = None
        self.models.clear()
        self.active_lora = None
        _clear_cuda()

    def _load_llm(self):
        if self._llm is not None:
            self.models["orpheus_seedvc"] = (self._llm, time.monotonic())
            return self._llm

        from llama_cpp import Llama

        self.seed_vc.unload()
        _clear_cuda()
        gguf = self._resolve_gguf_path(local_only=True) or self._resolve_gguf_path(local_only=False)
        if gguf is None:
            raise FileNotFoundError(
                f"Orpheus-3B GGUF not found. Run `python -m backend.download` to download {ORPHEUS_GGUF_REPO}."
            )

        # Tune GPU layer offload automatically for 4 GB vs 8 GB+ NVIDIA GPUs
        n_gpu_layers = -1
        try:
            import torch

            if not torch.cuda.is_available():
                n_gpu_layers = 0
            else:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                # Leave safe headroom for Windows desktop compositor on 4 GB GPUs (RTX 2050/3050)
                if vram_gb < 5.5:
                    n_gpu_layers = 22
        except Exception:
            n_gpu_layers = 0

        self._llm = Llama(
            model_path=str(gguf),
            n_ctx=2048,
            n_gpu_layers=n_gpu_layers,
            n_threads=max(4, (os.cpu_count() or 8) - 2),
            verbose=False,
        )
        self.models["orpheus_seedvc"] = (self._llm, time.monotonic())
        return self._llm

    def load(self, key: str = "orpheus_seedvc"):
        return self._load_llm()

    def speaker_embedding(self, source):
        return None

    def _generate_beat_codes(
        self,
        llm,
        *,
        b_text: str,
        voice_name: str,
        eff_temp: float,
        eff_top_p: float,
        eff_rep: float,
        max_toks: int,
        b_seed: int,
    ) -> list[list[int]] | None:
        text_tokens = llm.tokenize(f"{voice_name}: {b_text}".encode("utf-8"), add_bos=False, special=True)
        prompt_tokens = [_SOH_TOKEN, _BOS_TOKEN, *text_tokens, _EOT_TOKEN, _EOH_TOKEN, _SOA_TOKEN, _SOS_TOKEN]
        llm.reset()
        llm.set_seed(b_seed)

        generated: list[int] = []
        for tok in llm.generate(
            prompt_tokens,
            temp=eff_temp,
            top_p=eff_top_p,
            repeat_penalty=eff_rep,
        ):
            if tok in (_EOS_TOKEN, _EOA_TOKEN, _EOT_TOKEN):
                break
            generated.append(int(tok))
            if len(generated) >= max_toks:
                break
        return _extract_snac_codes(generated)

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
        from .mlx_tts import _beat_needs_reroll, _clean_words, _trim_and_fade_24k, split_orpheus_beats

        emit = progress_cb or (lambda _ev: None)
        beats = split_orpheus_beats(text, base_pause=float(np.clip(period_pause, 0.15, 1.5)))
        if not beats:
            raise ValueError("Enter text to synthesize.")

        raw_voice = (voice_profile.get("orpheus_voice") or "leo").strip().lower()
        voice_name = raw_voice if raw_voice in _NATIVE_ORPHEUS_VOICES else "leo"

        base_temp = 0.58 if abs(float(temperature) - 0.72) < 1e-3 else float(temperature)
        temp_adj = ((50.0 - stability) / 100.0) * 0.10 + ((style_exaggeration - 45.0) / 100.0) * 0.06
        eff_temp = float(np.clip(base_temp + temp_adj, 0.46, 0.66))
        eff_top_p = float(np.clip(top_p, 0.82, 0.88))
        eff_rep = float(np.clip(repetition_penalty, 1.15, 1.22))
        eff_cfg = float(np.clip(0.55 + (similarity_boost / 100.0) * 0.30, 0.55, 0.88))
        eff_steps = max(15, min(50, int(diffusion_steps or voice_profile.get("diffusion_steps") or 30)))

        # Stage 1a: Orpheus-3B Q4_K_M GGUF token generation
        llm = self._load_llm()
        beat_codes: list[list[list[int]]] = []
        for idx, (b_text, _pause_s) in enumerate(beats):
            emit({
                "type": "progress",
                "current": idx + 1,
                "total": len(beats),
                "stage": "acting",
                "emotion": f"Stage 1/2 · Acting beat {idx + 1}/{len(beats)}",
            })
            word_cnt = len(_clean_words(b_text))
            max_toks = min(1050, max(560, int(word_cnt * 52)))
            codes = None
            for attempt in range(3):
                b_seed = (int(seed) + idx * 41 + attempt * 17) % (2**31 - 1)
                codes = self._generate_beat_codes(
                    llm,
                    b_text=b_text,
                    voice_name=voice_name,
                    eff_temp=eff_temp + (0.02 if attempt >= 2 else 0.0),
                    eff_top_p=eff_top_p,
                    eff_rep=eff_rep,
                    max_toks=max_toks,
                    b_seed=b_seed,
                )
                if codes is not None:
                    break
            if codes is None:
                raise ValueError(f"Failed to synthesize beat {idx + 1}.")
            beat_codes.append(codes)

        # Unload Orpheus GGUF completely before SNAC decode & Whisper (0 VRAM overlap on 4 GB GPUs)
        self._unload_llm()

        sr_o = 24000
        beat_wavs = [_trim_and_fade_24k(w, sr_o) for w in _decode_snac_24k(beat_codes)]

        # Stage 1b: CPU int8 faster-whisper verbatim accuracy check (0 MB VRAM)
        try:
            from faster_whisper import WhisperModel

            emit({
                "type": "progress",
                "current": len(beats),
                "total": len(beats),
                "stage": "verify",
                "emotion": "Stage 1/2 · Verifying script accuracy",
            })
            fw = WhisperModel(WHISPER_FASTER, device="cpu", compute_type="int8")
            flagged: list[tuple[int, float]] = []
            for idx, ((b_text, _), wav) in enumerate(zip(beats, beat_wavs)):
                if len(_clean_words(b_text)) < 2:
                    continue
                wav_16k = resample_poly(wav, 2, 3).astype(np.float32)
                segs, _ = fw.transcribe(wav_16k, language="en", condition_on_previous_text=False)
                asr = " ".join(s.text.strip() for s in segs).strip()
                sc, needs_reroll = _beat_needs_reroll(b_text, asr)
                if needs_reroll:
                    flagged.append((idx, sc))

            if flagged:
                llm = self._load_llm()
                reroll_codes: list[tuple[int, list[list[int]]]] = []
                for pos, (idx, _) in enumerate(flagged, 1):
                    b_text, _ = beats[idx]
                    emit({
                        "type": "progress",
                        "current": pos,
                        "total": len(flagged),
                        "stage": "verify",
                        "emotion": f"Stage 1/2 · Polishing pronunciation ({pos}/{len(flagged)})",
                    })
                    word_cnt = len(_clean_words(b_text))
                    max_toks = min(1050, max(560, int(word_cnt * 52)))
                    for extra in range(2):
                        b_seed = (int(seed) + idx * 97 + (extra + 5) * 131) % (2**31 - 1)
                        c = self._generate_beat_codes(
                            llm,
                            b_text=b_text,
                            voice_name=voice_name,
                            eff_temp=max(0.48, eff_temp - 0.04 * (extra + 1)),
                            eff_top_p=min(eff_top_p, 0.85),
                            eff_rep=eff_rep,
                            max_toks=max_toks,
                            b_seed=b_seed,
                        )
                        if c is not None:
                            reroll_codes.append((idx, c))
                self._unload_llm()

                if reroll_codes:
                    decoded = _decode_snac_24k([c for _, c in reroll_codes])
                    best_scores = {idx: sc for idx, sc in flagged}
                    for (idx, _), raw_w in zip(reroll_codes, decoded):
                        cand_w = _trim_and_fade_24k(raw_w, sr_o)
                        wav_16k = resample_poly(cand_w, 2, 3).astype(np.float32)
                        segs, _ = fw.transcribe(wav_16k, language="en", condition_on_previous_text=False)
                        asr = " ".join(s.text.strip() for s in segs).strip()
                        sc, _ = _beat_needs_reroll(beats[idx][0], asr)
                        if sc > best_scores[idx]:
                            best_scores[idx] = sc
                            beat_wavs[idx] = cand_w
            del fw
            gc.collect()
        except Exception as exc:
            print(f"[YouClone CUDA] Whisper verification skipped: {exc}", flush=True)
            self._unload_llm()

        parts: list[np.ndarray] = []
        for (_b_text, pause_s), best_wav in zip(beats, beat_wavs):
            rms = float(np.sqrt(np.mean(best_wav * best_wav)) + 1e-8)
            gain = min(2.0, max(0.55, 0.14 / rms))
            parts.append(np.clip(best_wav * gain, -0.96, 0.96).astype(np.float32))
            if pause_s > 0:
                parts.append(np.zeros((int(pause_s * sr_o),), dtype=np.float32))
        stage1_audio = np.concatenate(parts).astype(np.float32)

        # Stage 2: Seed-VC 44.1 kHz F0-Conditioned Voice Lock (sole owner of CUDA GPU)
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
                self.seed_vc.unload()

        return dsp.resample(stage1_audio, sr_o).astype(np.float32)

    def convert_speech_to_speech(
        self,
        source_wav: Path,
        voice_profile: dict,
        *,
        diffusion_steps: int = 30,
        cfg_rate: float = 0.75,
    ) -> np.ndarray:
        ref_audio = voice_profile.get("ref_audio")
        if not ref_audio or not Path(ref_audio).is_file():
            raise ValueError("Selected voice profile has no reference audio.")
        self._unload_llm()
        with tempfile.TemporaryDirectory(dir=self.data_dir) as tmpdir:
            out_vc = Path(tmpdir) / "s2s_44k.wav"
            try:
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
            finally:
                self.seed_vc.unload()
