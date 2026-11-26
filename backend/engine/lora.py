import json
import math
import random
import re
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from uuid import uuid4

import numpy as np
import soundfile as sf

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    HAS_MLX = True
except ImportError:
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    HAS_MLX = False

from . import dsp
from .transcriber import transcribe

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
}


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper around a frozen nn.Linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        out_dims, in_dims = base.weight.shape
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.lora_a = mx.random.normal((in_dims, rank), dtype=mx.float32) * (1.0 / math.sqrt(rank))
        self.lora_b = mx.zeros((rank, out_dims), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        delta = ((x.astype(mx.float32) @ self.lora_a) @ self.lora_b) * self.scale
        return y + delta.astype(y.dtype)


def strip_omnivoice_lora(model) -> None:
    """Restore original nn.Linear modules on OmniVoice attention & codebook heads."""
    layers = getattr(getattr(model, "backbone", None), "layers", [])
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, attr, None)
            if isinstance(mod, LoRALinear):
                setattr(attn, attr, mod.base)

    heads = getattr(model, "audio_heads", None)
    if heads is not None:
        for i in range(len(heads)):
            if isinstance(heads[i], LoRALinear):
                heads[i] = heads[i].base


def inject_omnivoice_lora(model, rank: int = 8, alpha: float | None = None, num_layers: int = 8) -> int:
    """Inject LoRA adapters into top OmniVoice backbone attention projections (keeping audio_heads frozen)."""
    strip_omnivoice_lora(model)
    effective_alpha = float(rank) * 0.25 if alpha is None else float(alpha)
    layers = getattr(getattr(model, "backbone", None), "layers", [])
    start_idx = max(0, len(layers) - max(1, num_layers))
    injected = 0
    for layer in layers[start_idx:]:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, attr, None)
            if isinstance(mod, nn.Linear):
                setattr(attn, attr, LoRALinear(mod, rank=rank, alpha=effective_alpha))
                injected += 1
    return injected


def save_lora_adapter(model, adapter_dir: Path, metadata: dict) -> dict:
    """Save trainable LoRA weights (.safetensors) and metadata (.json) to adapter_dir."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_id = metadata["id"]
    weights = {k: v for k, v in tree_flatten(model.trainable_parameters()) if "lora_" in k}
    if not weights:
        raise ValueError("No LoRA adapter weights were found to save.")
    mx.eval(*weights.values())
    mx.save_safetensors(str(adapter_dir / f"{adapter_id}.safetensors"), weights)
    (adapter_dir / f"{adapter_id}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_lora_adapter(model, adapter_dir: Path, adapter_id: str) -> dict:
    """Inject LoRA modules matching saved metadata and load weights into OmniVoice."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", adapter_id or ""):
        raise ValueError("Invalid LoRA adapter identifier.")
    meta_path = adapter_dir / f"{adapter_id}.json"
    weights_path = adapter_dir / f"{adapter_id}.safetensors"
    if not meta_path.is_file() or not weights_path.is_file():
        raise ValueError(f"LoRA adapter '{adapter_id}' was not found on disk.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    rank = int(metadata.get("rank", 8))
    alpha = float(metadata.get("alpha", rank * 0.25))
    num_layers = int(metadata.get("num_layers", 8))
    inject_omnivoice_lora(model, rank=rank, alpha=alpha, num_layers=num_layers)
    loaded = mx.load(str(weights_path))
    model.load_weights(list(loaded.items()), strict=False)
    mx.eval(model.parameters())
    return metadata


def list_lora_adapters(adapter_dir: Path) -> list[dict]:
    """List all saved LoRA adapters sorted newest first."""
    if not adapter_dir.is_dir():
        return []
    items = []
    for meta_file in adapter_dir.glob("*.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            if (adapter_dir / f"{data['id']}.safetensors").is_file():
                items.append(data)
        except Exception:
            continue
    items.sort(key=lambda item: item.get("created", 0), reverse=True)
    return items


def _clean_segment_windows(audio_path: Path, raw_result: dict) -> list[dict]:
    """Normalize Whisper segments into 1.5s–9.5s training chunks with clean text."""
    info = sf.info(audio_path)
    total_dur = info.duration
    raw_segments = raw_result.get("segments") or []
    lang_code = (raw_result.get("language") or "en").lower()
    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.capitalize() if lang_code else "English")
    cleaned: list[dict] = []

    if not raw_segments and (raw_result.get("text") or "").strip():
        raw_segments = [{"start": 0.0, "end": min(total_dur, 8.5), "text": raw_result["text"].strip()}]

    for seg in raw_segments:
        text = " ".join(str(seg.get("text") or "").split())
        if len(text) < 4:
            continue
        start = max(0.0, float(seg.get("start", 0.0)))
        end = min(total_dur, float(seg.get("end", start + 5.0)))
        dur = end - start
        if dur < 1.5:
            continue
        if dur <= 9.5:
            cleaned.append({"start": start, "end": end, "text": text, "language": lang_code, "lang_name": lang_name})
        else:
            # Split long continuous segments evenly into ~7s chunks by word proportion
            words = text.split()
            n_chunks = max(2, math.ceil(dur / 7.5))
            step_s = dur / n_chunks
            words_per = max(1, math.ceil(len(words) / n_chunks))
            for idx in range(n_chunks):
                sub_words = words[idx * words_per : (idx + 1) * words_per]
                if not sub_words:
                    continue
                sub_start = start + idx * step_s
                sub_end = min(end, sub_start + step_s)
                if sub_end - sub_start >= 1.5:
                    cleaned.append({
                        "start": sub_start,
                        "end": sub_end,
                        "text": " ".join(sub_words),
                        "language": lang_code,
                        "lang_name": lang_name,
                    })
    return cleaned


def train_voice_lora(
    engine,
    workspace_root: Path,
    data_dir: Path,
    adapter_name: str,
    sources: list[str],
    rank: int = 8,
    steps: int = 60,
    lr: float = 1e-4,
    num_layers: int = 8,
    progress_cb: Callable[[int, int, float, str], None] | None = None,
    adapter_id: str | None = None,
) -> dict:
    """Slice workspace recordings, encode with OmniVoice codec, and run MLX LoRA fine-tuning."""
    safe_sources: list[Path] = []
    for src in sources:
        name = Path(src).name
        candidate = (workspace_root / name).resolve()
        if candidate.parent == workspace_root.resolve() and candidate.is_file():
            safe_sources.append(candidate)
    if not safe_sources:
        raise ValueError("Select at least one valid workspace audio recording to fine-tune.")

    model = engine.load("omni")
    if getattr(model, "audio_tokenizer", None) is None or getattr(model, "text_tokenizer", None) is None:
        raise ValueError("OmniVoice tokenizers are not loaded.")

    if progress_cb:
        progress_cb(0, steps, 0.0, "Transcribing & slicing workspace voice recordings…")

    examples: list[dict] = []
    best_ref: dict | None = None

    with TemporaryDirectory(dir=data_dir) as tmp_str:
        tmp_dir = Path(tmp_str)
        for src_path in safe_sources:
            offset_s = 7.30 if src_path.name == "english.mp3" else 0.0
            mono_24k = tmp_dir / f"{src_path.stem}_24k.wav"
            dsp.ffmpeg(
                "-ss", offset_s, "-i", src_path, "-t", 180, "-vn", "-ac", 1, "-ar", 24000,
                "-af", "afftdn=nr=6:nf=-48", "-c:a", "pcm_s16le", mono_24k,
            )
            asr = transcribe(mono_24k)
            segments = _clean_segment_windows(mono_24k, asr)
            if not segments:
                continue

            audio_24k, _ = sf.read(mono_24k, dtype="float32")
            for seg in segments:
                s_idx = max(0, int(seg["start"] * 24000))
                e_idx = min(len(audio_24k), int(seg["end"] * 24000))
                clip = audio_24k[s_idx:e_idx]
                if len(clip) < int(1.5 * 24000):
                    continue
                rms = float(np.sqrt(np.mean(clip ** 2)))
                if rms < 0.005:
                    continue
                if 0 < rms < 0.1:
                    clip = clip * (0.1 / rms)
                wav_mx = mx.array(clip)[None, :, None]
                tokens = model.audio_tokenizer.encode(wav_mx)[0]
                mx.eval(tokens)
                T = int(tokens.shape[0])
                if T < 12:
                    continue
                item = {
                    "tokens": tokens,
                    "text": seg["text"],
                    "language": seg["lang_name"],
                    "lang_code": seg["language"],
                    "duration": len(clip) / 24000.0,
                    "source": src_path.name,
                    "start": offset_s + seg["start"],
                }
                examples.append(item)
                # Prefer a clean 4–9s English clip as the paired Voice Vault reference sample
                score = (10.0 if seg["language"] == "en" else 2.0) + min(8.5, item["duration"])
                if best_ref is None or score > best_ref["score"]:
                    best_ref = {**item, "src_path": src_path, "score": score}

        if not examples or best_ref is None:
            raise ValueError("Could not extract any clean voiced segments (1.5s–9.5s) from the selected files.")

        # Save the best reference clip as a standard 44.1kHz sample in DATA/samples for the Voice Vault
        sample_id = str(uuid4())
        sample_dest = data_dir / "samples" / f"{sample_id}.wav"
        try:
            dsp.sanitize(best_ref["src_path"], sample_dest, best_ref["start"], min(9.5, best_ref["duration"]))
        except Exception:
            fallback_wav = dsp.resample(
                sf.read(tmp_dir / f"{best_ref['src_path'].stem}_24k.wav", dtype="float32")[0][
                    max(0, int((best_ref["start"] - (7.6 if best_ref["src_path"].name == "english.mp3" else 0.0)) * 24000)) :
                    int((best_ref["start"] - (7.6 if best_ref["src_path"].name == "english.mp3" else 0.0) + min(9.5, best_ref["duration"])) * 24000)
                ],
                24000,
            )
            sf.write(sample_dest, fallback_wav, dsp.RATE, subtype="PCM_16")
        sample_seconds = float(sf.info(sample_dest).duration)

    alpha = float(rank) * 0.25
    inject_omnivoice_lora(model, rank=rank, alpha=alpha, num_layers=num_layers)
    model.freeze()
    model.unfreeze(keys=["lora_a", "lora_b"])

    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0.01)
    mask_id = int(model.config.audio_mask_id)
    rng = random.Random(42)
    raw_cb_weights = [8.0, 8.0, 6.0, 6.0, 4.0, 4.0, 2.0, 2.0]

    def loss_fn(mdl, cond_input_ids, cond_audio_mask, target_tokens, loss_mask):
        logits = mdl(cond_input_ids, cond_audio_mask)
        T_len = target_tokens.shape[1]
        C_len = target_tokens.shape[2]
        pred_logits = logits[:, -T_len:, :, :mask_id].astype(mx.float32)
        flat_logits = pred_logits.reshape(-1, mask_id)
        flat_targets = mx.clip(target_tokens, 0, mask_id - 1).reshape(-1)
        ce = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none").reshape(1, T_len, C_len)
        mask_f = loss_mask.astype(mx.float32)
        layer_means = (ce * mask_f).sum(axis=(0, 1)) / mx.maximum(mask_f.sum(axis=(0, 1)), 1.0)
        cb_w = mx.array(raw_cb_weights[:C_len], dtype=mx.float32)
        cb_w = cb_w / mx.maximum(cb_w.sum(), 1.0)
        return (layer_means * cb_w).sum()

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []

    for step in range(1, steps + 1):
        ex = examples[(step - 1) % len(examples)]
        if step % len(examples) == 1 and step > 1:
            rng.shuffle(examples)

        use_ref = len(examples) > 1 and rng.random() < 0.7
        ref_ex = rng.choice([e for e in examples if e is not ex]) if use_ref else None

        style_ids, text_ids = model._tokenize_style_and_text(
            text=ex["text"],
            language=ex["language"],
            instruct="None",
            text_tokenizer=model.text_tokenizer,
            denoise=True,
            ref_text=ref_ex["text"] if ref_ex else None,
        )

        target_tokens = ex["tokens"]
        T_len = int(target_tokens.shape[0])
        C = int(model.config.num_audio_codebook)

        # Sample cosine-like masking ratio in [0.25, 1.0] matching OmniVoice NAR training
        mask_ratio = 0.25 + 0.75 * rng.random()
        np_mask = np.random.default_rng(step + 42).random((1, T_len, C)) < mask_ratio
        if not np_mask.any():
            np_mask[:, 0, :] = True
        frame_mask = mx.array(np_mask)
        # Ensure at least one token is masked
        masked_target = mx.where(frame_mask, mask_id, target_tokens[None, :, :])

        style_block = mx.broadcast_to(style_ids[None, :, None], (1, len(style_ids), C))
        text_block = mx.broadcast_to(text_ids[None, :, None], (1, len(text_ids), C))
        parts = [style_block, text_block]
        n_text = len(style_ids) + len(text_ids)
        if ref_ex is not None:
            parts.append(ref_ex["tokens"][None, :, :])
        parts.append(masked_target)

        cond_input_ids = mx.concatenate(parts, axis=1)
        L = cond_input_ids.shape[1]
        cond_audio_mask = mx.concatenate(
            [mx.zeros((1, n_text), dtype=mx.bool_), mx.ones((1, L - n_text), dtype=mx.bool_)],
            axis=1,
        )

        loss, grads = loss_and_grad(
            model,
            mx.stop_gradient(cond_input_ids),
            mx.stop_gradient(cond_audio_mask),
            mx.stop_gradient(target_tokens[None, :, :]),
            mx.stop_gradient(frame_mask),
        )
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        step_loss = float(loss.item())
        losses.append(step_loss)
        if progress_cb:
            progress_cb(
                step,
                steps,
                step_loss,
                f"Step {step}/{steps} · Loss {step_loss:.3f} · {len(examples)} clips",
            )

    # Restore eval mode and keep adapter active
    model.unfreeze()
    adapter_id = adapter_id or f"lora-{uuid4().hex[:12]}"
    avg_final_loss = sum(losses[-min(5, len(losses)):]) / min(5, len(losses))
    metadata = {
        "id": adapter_id,
        "name": adapter_name.strip() or "My Fine-Tuned Voice",
        "rank": rank,
        "alpha": alpha,
        "num_layers": num_layers,
        "steps": steps,
        "clips": len(examples),
        "sources": [p.name for p in safe_sources],
        "final_loss": round(avg_final_loss, 4),
        "sample_id": sample_id,
        "sample_transcript": best_ref["text"],
        "sample_language": best_ref["lang_code"],
        "sample_seconds": round(sample_seconds, 2),
        "created": int(time.time() * 1000),
    }
    save_lora_adapter(model, data_dir / "adapters", metadata)
    engine.active_lora = adapter_id
    return metadata
