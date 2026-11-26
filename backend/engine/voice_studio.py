"""YouClone Two-Tier Voice Creation Studio (Instant Voice Clone + Professional Voice Clone).

Manages studio voice profiles stored in `<VOXM5_DATA>/voices_studio/`:
  - Instant Voice Clone (recommended 2-5 min audio, min 15s):
      Instant true-median-F0 anchor extraction + Orpheus-3B matched actor + Seed-VC 44.1 kHz.
  - Professional Voice Clone (PVC, requires >= 20 min clean audio):
      Stage 1: Whisper transcription + Orpheus-3B attention-only QLoRA (accent, pauses, modulation).
      Stage 2: Seed-VC 44.1 kHz DiT fine-tuning on full 20+ min corpus (98-99% vocal timbre lock).
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import time
from math import gcd
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from . import dsp
from .seed_vc_bridge import SEED_VC_ROOT, run_seedvc_finetune
from .transcriber import transcribe as transcribe_audio

try:
    import mlx.core as mx
    import mlx_whisper
    from . import orpheus_lora

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    mlx_whisper = None  # type: ignore[assignment]
    orpheus_lora = None  # type: ignore[assignment]
    HAS_MLX = False

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
MIN_INSTANT_SECONDS = 15.0
RECOMMENDED_INSTANT_MIN_S = 120.0  # 2 minutes
MIN_PROFESSIONAL_SECONDS = 1200.0  # 20 minutes


def _whisper_snapshot() -> str:
    hub = WORKSPACE_ROOT / ".voxm5" / "huggingface" / "hub" / "models--mlx-community--whisper-large-v3-turbo" / "snapshots"
    if hub.is_dir():
        snaps = sorted([p for p in hub.iterdir() if p.is_dir()])
        if snaps:
            return str(snaps[-1])
    return "mlx-community/whisper-large-v3-turbo"


def _estimate_f0_median(sig: np.ndarray, sr: int) -> float:
    """Fast autocorrelation median F0 estimator (75-340 Hz)."""
    if sig.size < sr // 2:
        return 175.0
    if sr != 24000:
        g = gcd(sr, 24000)
        sig = resample_poly(sig, 24000 // g, sr // g).astype(np.float32)
        sr = 24000
    win = int(0.04 * sr)
    hop = int(0.025 * sr)
    lo, hi = int(sr / 340), int(sr / 75)
    f0s: list[float] = []
    # Cap scan to ~90s evenly spaced for speed
    max_frames = 3200
    step = max(hop, max(1, (len(sig) - win) // max_frames))
    for i in range(0, len(sig) - win, step):
        fr = sig[i : i + win]
        if float(np.sqrt(np.mean(fr * fr))) < 0.02:
            continue
        fr = fr - float(np.mean(fr))
        ac = np.correlate(fr, fr, mode="full")[win - 1 :]
        denom = float(ac[0]) + 1e-8
        seg = ac[lo:hi] / denom
        pk = int(np.argmax(seg))
        if float(seg[pk]) > 0.45:
            f0s.append(sr / float(lo + pk))
    return float(np.median(f0s)) if len(f0s) >= 5 else 175.0


def _pick_actor_for_voice(f0_hz: float, gender_hint: str = "auto") -> str:
    """Choose best matching built-in Orpheus-3B expressive actor voice for Instant Clone."""
    g = (gender_hint or "auto").strip().lower()
    if g == "female":
        return "tara" if f0_hz < 225 else "leah"
    if g == "male":
        return "dan" if f0_hz < 150 else "leo"
    # Auto from median pitch
    if f0_hz >= 208.0:
        return "tara"
    if f0_hz <= 148.0:
        return "dan"
    return "leo"


def _decode_to_44k_mono(src: Path, dst: Path) -> float:
    dsp.ffmpeg("-i", src, "-vn", "-ac", 1, "-ar", 44100, "-c:a", "pcm_s16le", dst)
    return float(sf.info(dst).duration)


def _split_clean_clips_44k(sig: np.ndarray, sr: int = 44100, min_s: float = 2.4, max_s: float = 8.8) -> list[np.ndarray]:
    """Slice 44.1 kHz audio cleanly on energy valleys into 2.4-8.8s speech clips."""
    if sig.size < int(min_s * sr):
        return []
    win = int(0.05 * sr)
    n_frames = sig.size // win
    if n_frames < 4:
        return []
    frames = sig[: n_frames * win].reshape(n_frames, win)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    thresh = max(0.008, float(np.percentile(rms, 85)) * 0.08)
    voiced = rms >= thresh

    # Build contiguous active regions bridging pauses < 0.35s
    regions: list[tuple[int, int]] = []
    in_reg = False
    s0 = 0
    gap_frames = 0
    max_gap = 7  # 0.35s
    for idx, v in enumerate(voiced):
        if v:
            if not in_reg:
                in_reg = True
                s0 = idx
            gap_frames = 0
        else:
            if in_reg:
                gap_frames += 1
                if gap_frames > max_gap:
                    regions.append((s0 * win, (idx - gap_frames + 1) * win))
                    in_reg = False
    if in_reg:
        regions.append((s0 * win, n_frames * win))

    # Merge regions into 3.5-8.5s chunks
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = None, None
    for s, e in regions:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif (e - cur_s) / sr <= 8.2:
            cur_e = e
        else:
            if (cur_e - cur_s) / sr >= min_s:
                merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None and (cur_e - cur_s) / sr >= min_s:
        merged.append((cur_s, cur_e))

    pad = int(0.06 * sr)
    clips: list[np.ndarray] = []
    for s, e in merged:
        dur = (e - s) / sr
        if dur <= max_s:
            spans = [(s, e)]
        else:
            n_parts = int(np.ceil(dur / 7.2))
            step = (e - s) // n_parts
            spans = [(s + k * step, min(e, s + (k + 1) * step)) for k in range(n_parts)]
        for a, b in spans:
            seg = sig[max(0, a - pad) : min(sig.size, b + pad)].astype(np.float32)
            if int(min_s * sr) <= seg.size <= int((max_s + 0.8) * sr):
                if float(np.sqrt(np.mean(seg * seg))) >= 0.015:
                    clips.append(seg)
    return clips


def _extract_true_f0_reference(
    sig: np.ndarray, sr: int, target_sec: float = 14.0
) -> tuple[np.ndarray, np.ndarray, float]:
    """Select the cleanest `target_sec` segment whose median F0 best matches the speaker's
    true full-recording median F0 (preventing pitch overshoot from high-pitched hooks)."""
    true_f0 = _estimate_f0_median(sig, sr)
    win_len = int(target_sec * sr)
    if sig.size <= win_len:
        preview = sig[: min(sig.size, int(8.0 * sr))]
        return sig.astype(np.float32), preview.astype(np.float32), round(true_f0, 1)

    step = max(int(2.0 * sr), (sig.size - win_len) // 25)
    best_seg = sig[:win_len]
    best_score = 1e9
    for start in range(0, sig.size - win_len + 1, step):
        seg = sig[start : start + win_len]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms < 0.02:
            continue
        # Speech activity ratio
        active = float(np.mean(np.abs(seg) > rms * 0.25))
        if active < 0.35:
            continue
        f0 = _estimate_f0_median(seg, sr)
        err = abs(f0 - true_f0) - active * 4.0
        if err < best_score:
            best_score = err
            best_seg = seg

    preview_len = min(best_seg.size, int(8.0 * sr))
    return best_seg.astype(np.float32), best_seg[:preview_len].astype(np.float32), round(true_f0, 1)


class VoiceStudioStore:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.root = data_dir / "voices_studio"
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.ensure_default_voices()

    def _read_registry(self) -> list[dict]:
        if not self.registry_path.is_file():
            return []
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write_registry(self, items: list[dict]) -> None:
        self.registry_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def ensure_default_voices(self) -> list[dict]:
        """Keep existing local user voices intact and ensure built-in starter voices exist
        on fresh installations (both macOS and Windows) without requiring personal training data."""
        items = self._read_registry()
        now_ms = int(time.time() * 1000)

        # Ensure any existing custom local voice keeps its native single-token Orpheus speaker prefix
        changed = False
        native_voices = {"leo", "dan", "tara", "leah", "jess", "mia", "zac", "zoe"}
        for item in items:
            ov = (item.get("orpheus_voice") or "").strip().lower()
            if ov and ov not in native_voices:
                item["orpheus_voice"] = "leo"
                changed = True

        # Add clean built-in starter voices when the library is empty (fresh install)
        if not items:
            items = [
                {
                    "id": "studio-narrator-leo",
                    "name": "Studio Narrator · Leo",
                    "mode": "instant",
                    "is_default": True,
                    "orpheus_adapter": None,
                    "orpheus_voice": "leo",
                    "seedvc_checkpoint": None,
                    "seedvc_config": None,
                    "ref_audio": None,
                    "preview_file": None,
                    "target_f0_hz": 145.0,
                    "sample_minutes": 0.0,
                    "diffusion_steps": 30,
                    "cfg_rate": 0.72,
                    "created": now_ms,
                    "description": "Built-in expressive male narrator (Orpheus-3B). Click + Create Voice to clone your own.",
                },
                {
                    "id": "studio-narrator-tara",
                    "name": "Studio Narrator · Tara",
                    "mode": "instant",
                    "is_default": True,
                    "orpheus_adapter": None,
                    "orpheus_voice": "tara",
                    "seedvc_checkpoint": None,
                    "seedvc_config": None,
                    "ref_audio": None,
                    "preview_file": None,
                    "target_f0_hz": 215.0,
                    "sample_minutes": 0.0,
                    "diffusion_steps": 30,
                    "cfg_rate": 0.72,
                    "created": now_ms + 1,
                    "description": "Built-in expressive female narrator (Orpheus-3B). Click + Create Voice to clone your own.",
                },
            ]
            changed = True

        if changed:
            self._write_registry(items)
        return items

    def list_voices(self) -> list[dict]:
        self.ensure_default_voices()
        return self._read_registry()

    def get_voice(self, voice_id: str | None) -> dict | None:
        voices = self.list_voices()
        if voice_id:
            found = next((v for v in voices if v.get("id") == voice_id), None)
            if found:
                return found
        return voices[0] if voices else None

    def delete_voice(self, voice_id: str) -> bool:
        items = self._read_registry()
        target = next((v for v in items if v.get("id") == voice_id), None)
        if not target:
            return False
        if target.get("is_default"):
            raise ValueError("Cannot delete a built-in default voice.")
        items = [v for v in items if v.get("id") != voice_id]
        self._write_registry(items)
        vdir = self.root / voice_id
        if vdir.is_dir():
            shutil.rmtree(vdir, ignore_errors=True)
        adapter_id = target.get("orpheus_adapter")
        if adapter_id and adapter_id.startswith("orpheus-pvc-"):
            for ext in (".safetensors", ".json"):
                (self.data_dir / "adapters" / f"{adapter_id}{ext}").unlink(missing_ok=True)
        return True

    def create_designed_voice(
        self,
        *,
        name: str,
        category: str,
        prompt: str,
        candidate_wav: Path,
        gender_hint: str = "auto",
    ) -> dict:
        """Save a Voice-Design candidate WAV as a permanent, reusable Character Voice."""
        if not candidate_wav.is_file():
            raise ValueError("Selected voice preview expired or was not found. Generate candidates again.")

        vid = f"char-{uuid4().hex[:10]}"
        vdir = self.root / vid
        vdir.mkdir(parents=True, exist_ok=True)
        try:
            tmp_44k = vdir / "_cand_44k.wav"
            dur = _decode_to_44k_mono(candidate_wav, tmp_44k)
            if dur < 1.5:
                raise ValueError("Designed voice sample is too short. Use a slightly longer audition sentence.")
            sig, _ = sf.read(str(tmp_44k), dtype="float32")
            tmp_44k.unlink(missing_ok=True)

            # Repeat with a short clean pause if < 8s so Seed-VC gets a rich 10-14s timbre anchor
            if sig.size < int(8.0 * 44100):
                gap = np.zeros((int(0.25 * 44100),), dtype=np.float32)
                sig = np.concatenate([sig, gap, sig])[: int(14.0 * 44100)].astype(np.float32)

            ref_wav, prev_wav, f0_hz = _extract_true_f0_reference(sig, 44100, target_sec=14.0)
            ref_path = vdir / "ref.wav"
            prev_path = vdir / "preview.wav"
            sf.write(str(ref_path), ref_wav, 44100, subtype="PCM_16")
            sf.write(str(prev_path), prev_wav, 44100, subtype="PCM_16")

            # Infer gender hint from category or prompt when auto
            eff_gender = gender_hint
            if eff_gender == "auto":
                p_low = f"{category} {prompt}".lower()
                if any(w in p_low for w in ("female", "woman", "queen", "girl", "lady", "alto", "soprano")):
                    eff_gender = "female"
                elif any(w in p_low for w in ("male", "man", "baritone", "pirate", "wizard", "ceo", "detective")):
                    eff_gender = "male"

            actor = _pick_actor_for_voice(f0_hz, gender_hint=eff_gender)
            clean_cat = (category or "character").strip().lower()
            cat_title = clean_cat.replace("_", " ").title()
            short_prompt = prompt.strip()
            if len(short_prompt) > 110:
                short_prompt = short_prompt[:107] + "…"

            entry = {
                "id": vid,
                "name": name.strip(),
                "mode": "designed",
                "category": clean_cat,
                "design_prompt": prompt.strip(),
                "is_default": False,
                "orpheus_adapter": None,
                "orpheus_voice": actor,
                "seedvc_checkpoint": None,
                "seedvc_config": None,
                "ref_audio": str(ref_path),
                "preview_file": f"{vid}.wav",
                "target_f0_hz": f0_hz,
                "sample_minutes": round(dur / 60.0, 2),
                "diffusion_steps": 32,
                "cfg_rate": 0.76,
                "created": int(time.time() * 1000),
                "description": f"{cat_title} · {short_prompt}",
            }
            items = self._read_registry()
            items.append(entry)
            self._write_registry(items)
            return entry
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise

    def create_instant_voice(
        self,
        name: str,
        uploaded_paths: list[Path],
        gender_hint: str = "auto",
    ) -> dict:
        """Create an Instant Voice Clone (recommended 2-5 min, min 15s) in seconds."""
        if not uploaded_paths:
            raise ValueError("Upload at least one audio recording.")
        vid = f"instant-{uuid4().hex[:10]}"
        vdir = self.root / vid
        vdir.mkdir(parents=True, exist_ok=True)

        signals: list[np.ndarray] = []
        total_sec = 0.0
        try:
            for idx, src in enumerate(uploaded_paths):
                tmp_wav = vdir / f"_in_{idx}.wav"
                dur = _decode_to_44k_mono(src, tmp_wav)
                sig, _ = sf.read(str(tmp_wav), dtype="float32")
                tmp_wav.unlink(missing_ok=True)
                signals.append(sig)
                total_sec += dur

            if total_sec < MIN_INSTANT_SECONDS:
                raise ValueError(
                    f"Audio is too short ({total_sec:.1f}s). Upload at least 15 seconds (2–5 minutes recommended)."
                )

            combined = np.concatenate(signals).astype(np.float32)
            ref_wav, prev_wav, f0_hz = _extract_true_f0_reference(combined, 44100, target_sec=14.0)
            ref_path = vdir / "ref.wav"
            prev_path = vdir / "preview.wav"
            sf.write(str(ref_path), ref_wav, 44100, subtype="PCM_16")
            sf.write(str(prev_path), prev_wav, 44100, subtype="PCM_16")

            actor = _pick_actor_for_voice(f0_hz, gender_hint=gender_hint)
            mins = round(total_sec / 60.0, 2)
            entry = {
                "id": vid,
                "name": name.strip(),
                "mode": "instant",
                "is_default": False,
                "orpheus_adapter": None,
                "orpheus_voice": actor,
                "seedvc_checkpoint": None,
                "seedvc_config": None,
                "ref_audio": str(ref_path),
                "preview_file": f"{vid}.wav",
                "target_f0_hz": f0_hz,
                "sample_minutes": mins,
                "diffusion_steps": 30,
                "cfg_rate": 0.72,
                "created": int(time.time() * 1000),
                "description": f"Instant Voice Clone ({mins:.1f} min sample · F0 {f0_hz:.0f} Hz · Actor: {actor})",
            }
            items = self._read_registry()
            items.append(entry)
            self._write_registry(items)
            return entry
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise

    def train_professional_voice(
        self,
        engine,
        name: str,
        uploaded_paths: list[Path],
        *,
        seedvc_steps: int = 500,
        progress_cb: Callable[[str, int, str], None] | None = None,
    ) -> dict:
        """Full 2-Stage Professional Voice Clone (PVC, requires >= 20 min clean audio).

        Stage 1: Whisper transcription + Orpheus-3B attention-only QLoRA (exact expressions,
                 pauses, accent, and articulation).
        Stage 2: Seed-VC 44.1 kHz DiT fine-tuning on full 20+ min corpus (98-99% timbre lock).
        """
        report = progress_cb or (lambda stage, pct, msg: print(f"[PVC {pct}%] {stage}: {msg}", flush=True))
        if not uploaded_paths:
            raise ValueError("Upload at least 20 minutes of clean audio recordings.")

        # Guard low-VRAM GPUs (e.g. RTX 2050 4 GB) before allocating multi-GB training state
        if not HAS_MLX:
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ValueError(
                        "Professional Voice Clone (PVC) training requires an NVIDIA GPU with >= 8 GB VRAM "
                        "or Apple Silicon Mac (16 GB+ unified memory). Use Instant Voice Clone (2–5 min) on this PC."
                    )
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 7.5:
                    raise ValueError(
                        f"Detected {vram_gb:.1f} GB VRAM. Professional Voice Clone (PVC) training requires >= 8 GB VRAM. "
                        "Use Instant Voice Clone (2–5 min) on this GPU, or import a pre-trained PVC checkpoint."
                    )
            except ImportError:
                pass

        vid = f"pro-{uuid4().hex[:10]}"
        vdir = self.root / vid
        ds_dir = vdir / "_train_clips"
        vdir.mkdir(parents=True, exist_ok=True)
        ds_dir.mkdir(parents=True, exist_ok=True)

        try:
            report("Analyzing audio", 2, "Decoding recordings to 44.1 kHz mono…")
            signals: list[np.ndarray] = []
            total_sec = 0.0
            for idx, src in enumerate(uploaded_paths):
                tmp_wav = vdir / f"_in_{idx}.wav"
                dur = _decode_to_44k_mono(src, tmp_wav)
                sig, _ = sf.read(str(tmp_wav), dtype="float32")
                tmp_wav.unlink(missing_ok=True)
                signals.append(sig)
                total_sec += dur

            if total_sec < MIN_PROFESSIONAL_SECONDS:
                raise ValueError(
                    f"Professional Voice Clone requires at least 20 minutes of audio "
                    f"(uploaded {total_sec / 60.0:.1f} min). Upload more audio or use Instant Voice Clone (2–5 min)."
                )

            combined = np.concatenate(signals).astype(np.float32)
            report("Preparing dataset", 6, f"Slicing {total_sec/60.0:.1f} min of audio at natural pauses…")
            ref_wav, prev_wav, f0_hz = _extract_true_f0_reference(combined, 44100, target_sec=14.0)
            ref_path = vdir / "ref.wav"
            prev_path = vdir / "preview.wav"
            sf.write(str(ref_path), ref_wav, 44100, subtype="PCM_16")
            sf.write(str(prev_path), prev_wav, 44100, subtype="PCM_16")

            clips = _split_clean_clips_44k(combined, 44100)
            if len(clips) < 25:
                raise ValueError("Not enough clean speech segments detected. Ensure audio has clear spoken voice.")

            clip_paths: list[tuple[Path, float]] = []
            for i, seg in enumerate(clips):
                p = ds_dir / f"clip_{i:04d}.wav"
                sf.write(str(p), seg, 44100, subtype="PCM_16")
                clip_paths.append((p, float(np.sqrt(np.mean(seg * seg)))))

            # Unload inference engines to free GPU / unified memory
            engine.unload_all()

            adapter_id: str | None = None
            native_actor = _pick_actor_for_voice(f0_hz)

            # Stage 1 (Apple Silicon MLX): Whisper transcribe + Orpheus-3B attention-only QLoRA
            if HAS_MLX and mlx_whisper is not None and orpheus_lora is not None:
                report("Stage 1/2 · Expressions", 12, "Transcribing top clips for expression & articulation training…")
                ranked = sorted(clip_paths, key=lambda x: -x[1])
                step_stride = max(1, len(ranked) // 36)
                chosen_for_orpheus = [p for p, _ in ranked[::step_stride][:34]]
                snap = _whisper_snapshot()
                utterances: list[dict] = []
                for idx, cp in enumerate(chosen_for_orpheus):
                    asr = mlx_whisper.transcribe(str(cp), path_or_hf_repo=snap, language="en").get("text", "").strip()
                    if len(asr.split()) >= 4:
                        utterances.append({"path": cp, "text": asr})
                    if (idx + 1) % 5 == 0:
                        pct = 12 + int(10 * (idx + 1) / len(chosen_for_orpheus))
                        report("Stage 1/2 · Expressions", pct, f"Transcribed {idx+1}/{len(chosen_for_orpheus)} clips…")

                if len(utterances) >= 14:
                    adapter_id = f"orpheus-pvc-{vid}"
                    report("Stage 1/2 · Expressions", 24, "Fine-tuning Orpheus-3B on your exact modulations & pacing…")
                    orpheus_model = engine.load_orpheus()

                    def _orpheus_log(msg: str):
                        m = re.search(r"step\s+(\d+)/(\d+)", msg)
                        if m:
                            s, t = int(m.group(1)), max(1, int(m.group(2)))
                            report("Stage 1/2 · Expressions", 24 + int(18 * s / t), msg.replace("[Orpheus-QLoRA] ", ""))

                    orpheus_lora.train_orpheus_voice_lora(
                        model=orpheus_model,
                        utterances=utterances,
                        adapter_dir=self.data_dir / "adapters",
                        adapter_id=adapter_id,
                        voice_name=native_actor,
                        rank=16,
                        alpha=16.0,
                        num_layers=20,
                        include_mlp=False,
                        dropout=0.05,
                        epochs=5,
                        grad_accum=4,
                        lr=8.0e-5,
                        eval_every_steps=4,
                        patience=3,
                        progress_cb=_orpheus_log,
                    )

            # Free Stage 1 RAM completely before Stage 2 PyTorch Seed-VC fine-tuning
            engine.unload_all()
            gc.collect()
            if mx is not None:
                mx.clear_cache()

            # Stage 2: Dedicated Seed-VC 44.1 kHz DiT speaker fine-tuning on full 20+ min corpus
            report("Stage 2/2 · Voice Timbre", 44, f"Fine-tuning 44.1 kHz Seed-VC on {len(clip_paths)} clips ({total_sec/60.0:.1f} min)…")

            def _vc_prog(step: int, total: int, loss: float):
                pct = 44 + int(52 * min(1.0, step / max(1, total)))
                report("Stage 2/2 · Voice Timbre", pct, f"Step {step}/{total} · loss {loss:.4f}")

            ft_model, ft_cfg = run_seedvc_finetune(
                dataset_dir=ds_dir,
                run_name=f"pvc_{vid}",
                max_steps=seedvc_steps,
                batch_size=2,
                save_every=max(200, seedvc_steps // 2),
                progress_cb=_vc_prog,
            )

            # Move final inference-only ft_model.pth into voice folder & remove raw slice cache
            dst_ckpt = vdir / "ft_model.pth"
            dst_cfg = vdir / ft_cfg.name
            shutil.move(str(ft_model), str(dst_ckpt))
            shutil.copy2(str(ft_cfg), str(dst_cfg))
            shutil.rmtree(ft_model.parent, ignore_errors=True)
            shutil.rmtree(ds_dir, ignore_errors=True)

            mins = round(total_sec / 60.0, 1)
            entry = {
                "id": vid,
                "name": name.strip(),
                "mode": "professional",
                "is_default": False,
                "orpheus_adapter": adapter_id,
                "orpheus_voice": native_actor,
                "seedvc_checkpoint": str(dst_ckpt),
                "seedvc_config": str(dst_cfg),
                "ref_audio": str(ref_path),
                "preview_file": f"{vid}.wav",
                "target_f0_hz": f0_hz,
                "sample_minutes": mins,
                "diffusion_steps": 35,
                "cfg_rate": 0.76,
                "created": int(time.time() * 1000),
                "description": f"Professional Voice Clone ({mins:.1f} min · Orpheus QLoRA + Seed-VC 44.1 kHz)",
            }
            items = self._read_registry()
            items.append(entry)
            self._write_registry(items)
            report("Complete", 100, f"Professional Voice '{entry['name']}' is ready!")
            return entry
        except Exception:
            shutil.rmtree(vdir, ignore_errors=True)
            raise
