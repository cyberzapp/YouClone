"""Local MLX QLoRA Fine-Tuning & Inference Adapter for Orpheus-3B (8-bit).

Trains float32 LoRA adapters on top of frozen 8-bit QuantizedLinear projections
inside `mlx-community/orpheus-3b-0.1-ft-8bit`, learning a speaker's expressive
cadence, accent, and rhythmic pacing while preserving Orpheus-3B's 3B-LLM semantic
voice acting & paralinguistic tags (`<sigh>`, `<chuckle>`, `<laugh>`, `<gasp>`).
"""

from __future__ import annotations

import gc
import json
import math
import random
import time
from math import gcd
from pathlib import Path
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import soundfile as sf
from mlx.utils import tree_flatten, tree_map
from scipy.signal import resample_poly

from mlx_audio.tts.models.llama.llama import encode_audio_to_codes

# Orpheus-3B special control token IDs (verified against mlx_audio llama.py)
SOH_TOKEN = 128259  # Start of human
EOT_TOKEN = 128009  # End of text
EOH_TOKEN = 128260  # End of human
SOA_TOKEN = 128261  # Start of AI
SOS_TOKEN = 128257  # Start of speech
EOS_TOKEN = 128258  # End of speech
EOA_TOKEN = 128262  # End of AI
AUDIO_CODE_OFFSET = 128266


def _linear_dims(layer: nn.Module) -> tuple[int, int]:
    """Return (in_dims, out_dims) for nn.QuantizedLinear or nn.Linear."""
    if isinstance(layer, nn.QuantizedLinear):
        out_dims = int(layer.weight.shape[0])
        in_dims = int(layer.weight.shape[1]) * (32 // int(layer.bits))
        return in_dims, out_dims
    if isinstance(layer, nn.Linear):
        out_dims, in_dims = int(layer.weight.shape[0]), int(layer.weight.shape[1])
        return in_dims, out_dims
    raise TypeError(f"Unsupported base layer for LoRA: {type(layer).__name__}")


class OrpheusQLoRALinear(nn.Module):
    """Trainable float32 LoRA adapter wrapping a frozen QuantizedLinear/Linear."""

    def __init__(
        self,
        base: nn.Module,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base = base
        in_dims, out_dims = _linear_dims(base)
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        self.dropout_p = float(dropout)
        self.lora_a = mx.random.normal((in_dims, self.rank), dtype=mx.float32) * (
            1.0 / math.sqrt(self.rank)
        )
        self.lora_b = mx.zeros((self.rank, out_dims), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        x_f32 = x.astype(mx.float32)
        if self.training and self.dropout_p > 0.0:
            keep = 1.0 - self.dropout_p
            mask = (mx.random.uniform(shape=x_f32.shape) < keep).astype(mx.float32) / keep
            x_f32 = x_f32 * mask
        delta = ((x_f32 @ self.lora_a) @ self.lora_b) * self.scale
        return y + delta.astype(y.dtype)


def strip_orpheus_lora(model) -> None:
    """Restore original frozen projections on every Llama transformer layer."""
    layers = getattr(getattr(model, "model", None), "layers", [])
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(attn, attr, None)
                if isinstance(mod, OrpheusQLoRALinear):
                    setattr(attn, attr, mod.base)
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            for attr in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(mlp, attr, None)
                if isinstance(mod, OrpheusQLoRALinear):
                    setattr(mlp, attr, mod.base)


def inject_orpheus_lora(
    model,
    rank: int = 16,
    alpha: float = 32.0,
    num_layers: int = 28,
    include_mlp: bool = True,
    dropout: float = 0.05,
) -> int:
    """Freeze all base weights and inject QLoRA adapters into Orpheus-3B layers."""
    strip_orpheus_lora(model)
    model.freeze()

    layers = model.model.layers
    start_idx = max(0, len(layers) - max(1, int(num_layers)))
    injected = 0

    for layer in layers[start_idx:]:
        attn = layer.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, attr, None)
            if isinstance(mod, (nn.QuantizedLinear, nn.Linear)):
                wrapped = OrpheusQLoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout)
                wrapped.base.freeze()
                setattr(attn, attr, wrapped)
                injected += 1

        if include_mlp:
            mlp = layer.mlp
            for attr in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(mlp, attr, None)
                if isinstance(mod, (nn.QuantizedLinear, nn.Linear)):
                    wrapped = OrpheusQLoRALinear(mod, rank=rank, alpha=alpha, dropout=dropout)
                    wrapped.base.freeze()
                    setattr(mlp, attr, wrapped)
                    injected += 1

    return injected


def collect_orpheus_lora_weights(model) -> dict[str, mx.array]:
    weights = {
        k: v
        for k, v in tree_flatten(model.model.trainable_parameters())
        if "lora_" in k
    }
    if weights:
        mx.eval(*weights.values())
    return weights


def save_orpheus_lora(
    model,
    adapter_dir: Path,
    adapter_id: str,
    metadata: dict,
) -> dict:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    weights = collect_orpheus_lora_weights(model)
    if not weights:
        raise ValueError("No trainable Orpheus QLoRA weights found.")
    mx.save_safetensors(str(adapter_dir / f"{adapter_id}.safetensors"), weights)
    (adapter_dir / f"{adapter_id}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def load_orpheus_lora(model, adapter_dir: Path, adapter_id: str) -> dict:
    meta_path = adapter_dir / f"{adapter_id}.json"
    weights_path = adapter_dir / f"{adapter_id}.safetensors"
    if not meta_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(
            f"Orpheus QLoRA adapter '{adapter_id}' not found in {adapter_dir}"
        )
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    inject_orpheus_lora(
        model,
        rank=int(metadata.get("rank", 16)),
        alpha=float(metadata.get("alpha", 32.0)),
        num_layers=int(metadata.get("num_layers", 28)),
        include_mlp=bool(metadata.get("include_mlp", True)),
        dropout=0.0,
    )
    loaded = mx.load(str(weights_path))
    model.model.load_weights(list(loaded.items()), strict=False)
    model.eval()
    mx.eval(model.model.parameters())
    return metadata


def _load_audio_slice_24k(
    path: Path,
    start_s: float | None = None,
    end_s: float | None = None,
    target_sr: int = 24000,
) -> np.ndarray:
    """Load mono audio slice resampled cleanly to 24 kHz for SNAC encoding."""
    info = sf.info(str(path))
    sr = int(info.samplerate)
    start_frame = max(0, int((start_s or 0.0) * sr))
    stop_frame = min(info.frames, int(end_s * sr)) if end_s is not None else info.frames
    if stop_frame <= start_frame:
        return np.zeros((0,), dtype=np.float32)

    sig, _ = sf.read(str(path), start=start_frame, stop=stop_frame, dtype="float32")
    if sig.ndim > 1:
        sig = sig.mean(axis=1)

    # Gentle 15 ms edge fade to eliminate hard slice clicks
    fade = min(int(0.015 * sr), sig.size // 8)
    if fade > 4:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        sig[:fade] *= ramp
        sig[-fade:] *= ramp[::-1]

    if sr != target_sr:
        g = gcd(sr, target_sr)
        sig = resample_poly(sig, target_sr // g, sr // g).astype(np.float32)

    # Peak-safe gentle normalization matching studio dialogue level (-1.0 dBFS ceiling)
    peak = float(np.max(np.abs(sig))) if sig.size else 0.0
    if peak > 0.95:
        sig = sig * (0.95 / peak)
    return sig.astype(np.float32)


def _encode_training_example(
    model,
    text: str,
    audio_24k: np.ndarray,
    voice_name: str = "leo",
) -> tuple[mx.array, int, int] | None:
    """Build exact Orpheus-3B sequence:
    [SOH] + tokenizer("leo: <text>") + [EOT, EOH] + [SOA, SOS] + snac_codes + [EOS, EOA]
    Returns (full_seq [1, L], prefix_len, n_audio_tokens).
    """
    if audio_24k.size < 24000:  # < 1.0s
        return None
    audio_mx = mx.array(audio_24k, dtype=mx.float32)
    codes = encode_audio_to_codes(audio_mx) + AUDIO_CODE_OFFSET  # [1, N]
    mx.eval(codes)
    n_audio = int(codes.shape[1])
    if n_audio < 21:
        return None

    prompt = f"{voice_name}: {text.strip()}"
    text_ids = model.tokenizer(prompt, return_tensors="mlx").input_ids.astype(mx.int32)
    start_tok = mx.array([[SOH_TOKEN]], dtype=mx.int32)
    end_toks = mx.array([[EOT_TOKEN, EOH_TOKEN]], dtype=mx.int32)
    prefix = mx.concatenate([start_tok, text_ids, end_toks], axis=1)  # [1, P]

    audio_start = mx.array([[SOA_TOKEN, SOS_TOKEN]], dtype=mx.int32)
    audio_end = mx.array([[EOS_TOKEN, EOA_TOKEN]], dtype=mx.int32)
    full_seq = mx.concatenate(
        [prefix, audio_start, codes.astype(mx.int32), audio_end], axis=1
    )
    mx.eval(full_seq)
    return full_seq, int(prefix.shape[1]), n_audio


def _compute_audio_ce_loss(
    llama_inner,
    full_seq: mx.array,
    prefix_len: int,
    eos_weight: float = 1.5,
) -> mx.array:
    """Causal cross-entropy loss strictly over [SOA, SOS, audio_codes..., EOS, EOA]."""
    x = full_seq[:, :-1]  # [1, L-1]
    targets = full_seq[0, prefix_len:]  # [M] = [SOA, SOS, codes..., EOS, EOA]
    hidden = llama_inner(x)  # [1, L-1, H]
    audio_hidden = hidden[:, prefix_len - 1 :, :]  # [1, M, H]
    logits = llama_inner.embed_tokens.as_linear(audio_hidden)[0].astype(mx.float32)  # [M, V]
    per_tok = nn.losses.cross_entropy(logits, targets, reduction="none")  # [M]

    if eos_weight != 1.0 and per_tok.shape[0] >= 4:
        # Gently emphasize EOS (index -2) and EOA (index -1) so utterances close cleanly
        n_tok = int(per_tok.shape[0])
        main_sum = mx.sum(per_tok[:-2])
        end_sum = mx.sum(per_tok[-2:])
        return (main_sum + eos_weight * end_sum) / ((n_tok - 2) + 2.0 * eos_weight)
    return mx.mean(per_tok)


def train_orpheus_suman_lora(
    model,
    utterances: list[dict],
    adapter_dir: Path,
    adapter_id: str = "orpheus-voice-v1",
    voice_name: str = "leo",
    rank: int = 16,
    alpha: float = 16.0,
    num_layers: int = 20,
    include_mlp: bool = False,
    dropout: float = 0.05,
    epochs: int = 4,
    grad_accum: int = 4,
    lr: float = 6.0e-5,
    weight_decay: float = 0.03,
    eos_weight: float = 1.2,
    eval_every_steps: int = 6,
    patience: int = 3,
    seed: int = 42,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Encode target audio clips to SNAC tokens, train QLoRA with held-out
    validation tracking and cosine LR schedule, and save the lowest-val checkpoint.

    Each utterance dict supports:
      - path: str | Path
      - text: str
      - start: float | None (optional slice start in seconds)
      - end: float | None (optional slice end in seconds)
      - weight: int (default 1; repeat count in training pool for priority viral clips)
      - is_val: bool (optional explicit validation holdout flag)
    """
    log = progress_cb or (lambda msg: print(msg, flush=True))
    t0 = time.time()
    log(f"[Orpheus-QLoRA] Encoding {len(utterances)} utterances via SNAC 24 kHz...")

    train_pool: list[tuple[mx.array, int, int, str]] = []
    val_items: list[tuple[mx.array, int, int, str]] = []
    auto_items: list[tuple[mx.array, int, int, str, int]] = []

    for i, utt in enumerate(utterances):
        wav_path = Path(utt["path"])
        if not wav_path.is_file():
            log(f"[Orpheus-QLoRA] Skipping missing file: {wav_path}")
            continue
        sig = _load_audio_slice_24k(
            wav_path,
            start_s=utt.get("start"),
            end_s=utt.get("end"),
            target_sr=24000,
        )
        if sig.size < 24000:
            continue
        enc = _encode_training_example(model, utt["text"], sig, voice_name=voice_name)
        if enc is None:
            continue
        full_seq, prefix_len, n_audio = enc
        label = f"{wav_path.name}[{utt.get('start', 0):.1f}-{utt.get('end', sig.size/24000):.1f}s]"
        if utt.get("is_val", False):
            val_items.append((full_seq, prefix_len, n_audio, label))
        else:
            auto_items.append((full_seq, prefix_len, n_audio, label, max(1, int(utt.get("weight", 1)))))
        if (i + 1) % 15 == 0:
            mx.clear_cache()

    if not val_items:
        # Hold out ~15% evenly spaced clips (min 4, max 8)
        n_hold = min(8, max(4, len(auto_items) // 7))
        stride = max(1, len(auto_items) // n_hold)
        val_set = set(range(0, len(auto_items), stride)[:n_hold])
        for idx, (seq, pl, na, lbl, w) in enumerate(auto_items):
            if idx in val_set:
                val_items.append((seq, pl, na, lbl))
            else:
                for _ in range(w):
                    train_pool.append((seq, pl, na, lbl))
    else:
        for seq, pl, na, lbl, w in auto_items:
            for _ in range(w):
                train_pool.append((seq, pl, na, lbl))

    if len(train_pool) < 8 or len(val_items) < 2:
        raise ValueError(
            f"Not enough valid clips to train Orpheus QLoRA (train={len(train_pool)}, val={len(val_items)})."
        )

    total_train_sec = sum(na for _, _, na, _ in train_pool) / 84.0
    total_val_sec = sum(na for _, _, na, _ in val_items) / 84.0
    log(
        f"[Orpheus-QLoRA] Encoded in {time.time()-t0:.1f}s | "
        f"train={len(train_pool)} weighted clips ({total_train_sec:.1f}s) | "
        f"val={len(val_items)} held-out clips ({total_val_sec:.1f}s)"
    )

    # Inject QLoRA adapters
    injected = inject_orpheus_lora(
        model,
        rank=rank,
        alpha=alpha,
        num_layers=num_layers,
        include_mlp=include_mlp,
        dropout=dropout,
    )
    n_trainable = sum(v.size for _, v in tree_flatten(model.model.trainable_parameters()))
    log(
        f"[Orpheus-QLoRA] Injected {injected} QLoRA projections "
        f"(r={rank}, alpha={alpha}, mlp={include_mlp}, {n_trainable/1e6:.2f}M float32 params)"
    )

    def eval_val() -> float:
        model.eval()
        losses = []
        for seq, pl, _, _ in val_items:
            lv = _compute_audio_ce_loss(model.model, seq, pl, eos_weight=eos_weight)
            mx.eval(lv)
            losses.append(float(lv.item()))
        mx.clear_cache()
        return float(np.mean(losses))

    base_val = eval_val()
    log(f"[Orpheus-QLoRA] Step   0 (Baseline) | val_loss={base_val:.4f}")

    steps_per_epoch = math.ceil(len(train_pool) / max(1, grad_accum))
    total_opt_steps = epochs * steps_per_epoch
    warmup_steps = min(15, max(5, total_opt_steps // 10))
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=weight_decay)

    def loss_fn(llama_inner, full_seq, prefix_len):
        return _compute_audio_ce_loss(llama_inner, full_seq, prefix_len, eos_weight=eos_weight)

    loss_and_grad = nn.value_and_grad(model.model, loss_fn)

    best_val = base_val
    best_step = 0
    best_epoch = 0
    best_weights = {k: mx.array(v) for k, v in collect_orpheus_lora_weights(model).items()}
    mx.eval(*best_weights.values())

    rng = random.Random(seed)
    opt_step = 0
    no_improve = 0
    stop_early = False
    t_train = time.time()

    def _check_val(ep_idx: int, recent_losses: list[float], lr_now: float) -> bool:
        nonlocal best_val, best_step, best_epoch, best_weights, no_improve
        val_loss = eval_val()
        tr_avg = float(np.mean(recent_losses)) if recent_losses else 0.0
        marker = ""
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_step = opt_step
            best_epoch = ep_idx
            best_weights = {
                k: mx.array(v) for k, v in collect_orpheus_lora_weights(model).items()
            }
            mx.eval(*best_weights.values())
            no_improve = 0
            marker = " *BEST*"
        else:
            no_improve += 1
        log(
            f"[Orpheus-QLoRA] Ep {ep_idx:2d}/{epochs} step {opt_step:3d}/{total_opt_steps} | "
            f"lr={lr_now:.2e} train={tr_avg:.4f} val={val_loss:.4f} "
            f"peak={mx.get_peak_memory()/1e9:.2f}GB ({time.time()-t_train:.0f}s){marker}"
        )
        model.train()
        return no_improve >= patience

    for ep in range(1, epochs + 1):
        if stop_early:
            break
        model.train()
        order = list(range(len(train_pool)))
        rng.shuffle(order)
        ep_losses: list[float] = []

        accum_grads = None
        accum_count = 0

        for pos, idx in enumerate(order):
            seq, pl, _, _ = train_pool[idx]
            lv, grads = loss_and_grad(model.model, seq, pl)
            mx.eval(lv, grads)
            lv_f = float(lv.item())
            ep_losses.append(lv_f)
            accum_count += 1

            if accum_grads is None:
                accum_grads = grads
            else:
                accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
                mx.eval(accum_grads)

            is_update = (accum_count >= grad_accum) or (pos == len(order) - 1)
            if is_update:
                opt_step += 1
                if opt_step <= warmup_steps:
                    cur_lr = lr * (opt_step / float(warmup_steps))
                else:
                    prog = (opt_step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
                    cur_lr = lr * (0.10 + 0.90 * 0.5 * (1.0 + math.cos(math.pi * prog)))
                optimizer.learning_rate = cur_lr

                inv_n = 1.0 / float(accum_count)
                avg_grads = tree_map(lambda g: g * inv_n, accum_grads)

                flat_g = [g for _, g in tree_flatten(avg_grads)]
                g_norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in flat_g))
                clip = mx.minimum(mx.array(1.0, dtype=mx.float32), 1.0 / (g_norm + 1e-6))
                clipped = tree_map(lambda g: (g.astype(mx.float32) * clip).astype(g.dtype), avg_grads)

                optimizer.update(model.model, clipped)
                mx.eval(model.model.trainable_parameters(), optimizer.state)

                accum_grads = None
                accum_count = 0
                mx.clear_cache()

                is_end_ep = pos == len(order) - 1
                if (eval_every_steps > 0 and opt_step % eval_every_steps == 0) or is_end_ep:
                    if _check_val(ep, ep_losses[-(eval_every_steps * grad_accum) :], cur_lr):
                        log(f"[Orpheus-QLoRA] Early stopping at step {opt_step} (best step {best_step}, val={best_val:.4f})")
                        stop_early = True
                        break

    # Restore best validation checkpoint and switch to eval mode
    model.model.load_weights(list(best_weights.items()), strict=False)
    model.eval()
    mx.eval(model.model.parameters())

    metadata = {
        "id": adapter_id,
        "name": f"{voice_name.capitalize()} Expressive Voice (Orpheus-3B QLoRA)",
        "model": "orpheus_3b",
        "voice_name": voice_name,
        "rank": rank,
        "alpha": alpha,
        "num_layers": num_layers,
        "include_mlp": include_mlp,
        "train_clips": len(train_pool),
        "val_clips": len(val_items),
        "epochs": epochs,
        "total_opt_steps": total_opt_steps,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "baseline_val_loss": round(base_val, 4),
        "best_val_loss": round(best_val, 4),
        "train_seconds": round(time.time() - t_train, 1),
        "created": int(time.time() * 1000),
    }
    save_orpheus_lora(model, adapter_dir, adapter_id, metadata)
    log(
        f"[Orpheus-QLoRA] Saved best adapter (ep {best_epoch}, step {best_step}, "
        f"val {base_val:.4f} -> {best_val:.4f}) in {time.time()-t_train:.1f}s -> "
        f"{adapter_dir / (adapter_id + '.safetensors')}"
    )
    gc.collect()
    mx.clear_cache()
    return metadata
