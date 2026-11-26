"""Prompt-Driven Character & Human Voice Design Engine for YouClone.

Generates 3 distinct brand-new synthetic voice candidates from a natural-language
character prompt (no reference audio or training required), applies optional character
formant/pitch sculpting (for Cartoon, Creature, Gritty, or Ethereal voices), and locks
the chosen candidate into a reusable Studio Character Voice (`ref.wav` + `preview.wav`).

Zero GPU Overlap:
  - Unloads Orpheus-3B & Seed-VC before loading `Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`.
  - Unloads `VoiceDesign-8bit` immediately after generating the 3 candidate previews so
    0 MB of GPU memory remains occupied during normal speech synthesis.
"""

from __future__ import annotations

import gc
import tempfile
import time
from math import gcd
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from huggingface_hub import snapshot_download
from scipy.signal import resample_poly

from . import dsp

try:
    import mlx.core as mx
    from mlx_audio.tts import load as mlx_load_tts

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    mlx_load_tts = None  # type: ignore[assignment]
    HAS_MLX = False

VOICE_DESIGN_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit"

TEXTURE_FILTERS: dict[str, tuple[float, str]] = {
    # (default_extra_semitones, extra_ffmpeg_eq_filter)
    "natural": (0.0, ""),
    "cartoon": (0.0, "highpass=f=130,treble=g=3.5:f=3600,acompressor=threshold=-18dB:ratio=2.8:attack=8:release=90"),
    "deep_creature": (0.0, "bass=g=5.0:f=115,lowpass=f=14500,acompressor=threshold=-16dB:ratio=3.2:attack=12:release=140"),
    "gritty": (0.0, "highpass=f=95,equalizer=f=2400:t=q:w=1.2:g=3.5,acompressor=threshold=-19dB:ratio=3.5:attack=5:release=80"),
    "ethereal": (0.0, "highpass=f=110,treble=g=4.0:f=5200,aecho=0.8:0.82:38|64:0.16|0.10"),
}


def check_voice_design_cached() -> bool:
    if not HAS_MLX:
        return True
    try:
        snap = snapshot_download(
            VOICE_DESIGN_REPO,
            allow_patterns=["config.json", "*.safetensors"],
            local_files_only=True,
        )
        return any(Path(snap).rglob("*.safetensors"))
    except Exception:
        return False


def ensure_voice_design_downloaded() -> str:
    if not HAS_MLX:
        return VOICE_DESIGN_REPO
    return snapshot_download(
        VOICE_DESIGN_REPO,
        ignore_patterns=["*.md", "*.png", "*.jpg"],
        max_workers=4,
    )


def _apply_character_sculpt(
    wav: np.ndarray,
    sr: int,
    *,
    pitch_semitones: float = 0.0,
    texture: str = "natural",
    work_dir: Path,
) -> tuple[np.ndarray, int]:
    """Apply true vocal-tract formant + pitch shift and character EQ at 44.1 kHz."""
    target_sr = 44100
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr

    semi = float(np.clip(pitch_semitones, -6.0, 6.0))
    tex_key = texture if texture in TEXTURE_FILTERS else "natural"
    _, eq_chain = TEXTURE_FILTERS[tex_key]

    filters: list[str] = []
    if abs(semi) >= 0.08:
        ratio = 2.0 ** (semi / 12.0)
        # asetrate + aresample shifts both fundamental pitch AND vocal-tract formants
        # (essential for convincing cartoon mice, goblins, or deep monsters), while
        # inverse atempo preserves natural speaking speed!
        inv_tempo = float(np.clip(1.0 / ratio, 0.52, 1.95))
        filters.append(f"asetrate={int(round(target_sr * ratio))}")
        filters.append(f"aresample={target_sr}")
        filters.append(f"atempo={inv_tempo:.5f}")

    if eq_chain:
        filters.append(eq_chain)

    filters.append("highpass=f=70,loudnorm=I=-15.0:TP=-1.2:LRA=11")
    af = ",".join(filters)

    in_path = work_dir / f"_vd_in_{uuid4().hex[:8]}.wav"
    out_path = work_dir / f"_vd_out_{uuid4().hex[:8]}.wav"
    try:
        sf.write(str(in_path), wav, target_sr, subtype="FLOAT")
        dsp.ffmpeg("-i", in_path, "-af", af, "-ar", target_sr, "-ac", 1, "-c:a", "pcm_s16le", out_path)
        sculpted, out_sr = sf.read(str(out_path), dtype="float32")
        if sculpted.ndim > 1:
            sculpted = sculpted.mean(axis=1)
        return sculpted.astype(np.float32), int(out_sr)
    finally:
        in_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


def _trim_edges(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-4:
        return audio
    thresh = max(1e-4, peak * 0.015)
    idx = np.flatnonzero(np.abs(audio) >= thresh)
    if idx.size:
        s = max(0, int(idx[0]) - int(0.03 * sr))
        e = min(audio.size, int(idx[-1]) + int(0.07 * sr))
        audio = audio[s:e]
    fade = min(int(0.015 * sr), audio.size // 8)
    if fade > 2:
        r = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        audio = audio.copy()
        audio[:fade] *= r
        audio[-fade:] *= r[::-1]
    return audio.astype(np.float32)


def generate_voice_design_candidates(
    engine,
    *,
    prompt: str,
    sample_text: str,
    pitch_semitones: float = 0.0,
    texture: str = "natural",
    seed: int = 108,
    takes_dir: Path,
    num_candidates: int = 3,
) -> list[dict]:
    """Synthesize 3 distinct character voice variations from `prompt` and save to `takes_dir`."""
    clean_prompt = prompt.strip()
    clean_text = sample_text.strip()
    if len(clean_prompt) < 4:
        raise ValueError("Describe the voice you want to design (e.g. tone, age, pitch, character vibe).")
    if len(clean_text) < 6:
        raise ValueError("Enter a short preview sentence (at least a few words) to audition the voice.")

    takes_dir.mkdir(parents=True, exist_ok=True)
    # Free Stage 1 Orpheus & Stage 2 Seed-VC completely before loading VoiceDesign (0 GPU overlap!)
    engine.unload_all()

    variant_specs = [
        ("Option 1 · Balanced", 0.74, 0.88),
        ("Option 2 · Expressive", 0.82, 0.91),
        ("Option 3 · Bold Character", 0.89, 0.93),
    ][: max(1, min(3, num_candidates))]

    raw_outputs: list[tuple[str, np.ndarray, int, int]] = []

    if HAS_MLX and mx is not None and mlx_load_tts is not None:
        mx.set_default_device(mx.gpu)
        try:
            mx.set_cache_limit(256 * 1024 * 1024)
        except Exception:
            pass

        ensure_voice_design_downloaded()
        vd_model = mlx_load_tts(VOICE_DESIGN_REPO)
        sr = int(getattr(vd_model, "sample_rate", 24000) or 24000)
        try:
            for idx, (v_label, temp, top_p) in enumerate(variant_specs):
                c_seed = (int(seed) + idx * 137 + int(time.time() * 10) % 97) % (2**31 - 1)
                mx.random.seed(c_seed)
                chunks: list[np.ndarray] = []
                for res in vd_model.generate_voice_design(
                    text=clean_text,
                    instruct=clean_prompt,
                    language="English",
                    temperature=temp,
                    top_p=top_p,
                    repetition_penalty=1.12,
                    max_tokens=900,
                    verbose=False,
                ):
                    chunks.append(np.asarray(res.audio, dtype=np.float32))
                    res_sr = getattr(res, "sample_rate", None)
                    if res_sr:
                        sr = int(res_sr)
                mx.synchronize()
                mx.clear_cache()
                if chunks:
                    wav = _trim_edges(np.concatenate(chunks), sr)
                    raw_outputs.append((v_label, wav, sr, c_seed))
        finally:
            del vd_model
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
    else:
        # Windows / Linux fallback: pair distinct expressive Orpheus actors + formant sculpting
        p_low = clean_prompt.lower()
        is_fem = any(w in p_low for w in ("female", "woman", "girl", "queen", "witch", "lady", "alto", "soprano", "she"))
        is_high = any(w in p_low for w in ("mouse", "squeak", "tiny", "goblin", "high", "child", "kid", "cartoon"))
        actors = (
            ["leah", "tara", "mia"]
            if is_fem
            else (["zac", "leo", "jess"] if is_high else ["dan", "leo", "zac"])
        )
        for idx, (v_label, temp, _top_p) in enumerate(variant_specs):
            c_seed = (int(seed) + idx * 137) % (2**31 - 1)
            wav_48k = engine.synthesize_studio_speech(
                text=clean_text,
                voice_profile={"orpheus_voice": actors[idx % len(actors)], "ref_audio": None},
                temperature=min(0.64, temp - 0.16),
                seed=c_seed,
            )
            raw_outputs.append((v_label, _trim_edges(wav_48k, 48000), 48000, c_seed))
        engine.unload_all()

    if not raw_outputs:
        raise RuntimeError("Voice Design model produced no audio. Try a slightly longer preview sentence.")

    candidates: list[dict] = []
    with tempfile.TemporaryDirectory(dir=takes_dir.parent) as tmp:
        work_dir = Path(tmp)
        for v_label, wav, sr, c_seed in raw_outputs:
            sculpted, out_sr = _apply_character_sculpt(
                wav,
                sr,
                pitch_semitones=pitch_semitones,
                texture=texture,
                work_dir=work_dir,
            )
            cid = str(uuid4())
            out_file = takes_dir / f"{cid}.wav"
            sf.write(str(out_file), sculpted, out_sr, subtype="PCM_16")
            dur = round(len(sculpted) / float(out_sr), 2)
            candidates.append(
                {
                    "id": cid,
                    "label": v_label,
                    "seconds": dur,
                    "seed": c_seed,
                    "url": f"/media/takes/{cid}.wav",
                }
            )

    return candidates
