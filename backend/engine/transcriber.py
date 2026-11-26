from __future__ import annotations

from huggingface_hub import snapshot_download

from ..models import WHISPER, WHISPER_FASTER

try:
    import mlx_whisper

    HAS_MLX_WHISPER = True
except ImportError:
    mlx_whisper = None  # type: ignore[assignment]
    HAS_MLX_WHISPER = False


def transcribe(path, *, words: bool = False):
    if HAS_MLX_WHISPER and mlx_whisper is not None:
        # Resolve only the inference files before handing mlx-whisper a local path.
        model_path = snapshot_download(
            WHISPER, allow_patterns=["config.json", "weights.safetensors"], local_files_only=True,
        )
        return mlx_whisper.transcribe(
            str(path),
            path_or_hf_repo=model_path,
            word_timestamps=words,
            condition_on_previous_text=False,
            verbose=None,
        )

    # Cross-platform Windows / Linux fallback using faster-whisper (CPU int8, 0 MB VRAM)
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_FASTER, device="cpu", compute_type="int8")
    segments_iter, info = model.transcribe(
        str(path),
        word_timestamps=words,
        condition_on_previous_text=False,
    )
    segments = []
    full_parts = []
    for seg in segments_iter:
        text = seg.text.strip()
        full_parts.append(text)
        entry: dict = {"start": float(seg.start), "end": float(seg.end), "text": text}
        if words and getattr(seg, "words", None):
            entry["words"] = [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in seg.words
            ]
        segments.append(entry)
    return {
        "text": " ".join(full_parts).strip(),
        "language": getattr(info, "language", "en") or "en",
        "segments": segments,
    }
