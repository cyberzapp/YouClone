from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import snapshot_download
from mlx_audio.tts import load

from .models import MODELS, WHISPER


def download(model):
    print(f"Downloading {model}", flush=True)
    snapshot_download(model, ignore_patterns=["*.md", "*.png", "*.jpg"], max_workers=4)


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(download, (*MODELS.values(), WHISPER)))
    for repo in MODELS.values():
        print(f"Validating {repo}", flush=True)
        model = load(repo)
        del model
