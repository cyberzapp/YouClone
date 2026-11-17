import mlx_whisper
from huggingface_hub import snapshot_download

from ..models import WHISPER


def transcribe(path, *, words=False):
    # Resolve only the inference files before handing mlx-whisper a local path.
    model_path = snapshot_download(
        WHISPER, allow_patterns=["config.json", "weights.safetensors"], local_files_only=True,
    )
    return mlx_whisper.transcribe(
        str(path), path_or_hf_repo=model_path, word_timestamps=words,
        condition_on_previous_text=False, verbose=None,
    )
