import sys
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download, snapshot_download

from .models import CORE_MODELS, MODELS, ORPHEUS_GGUF_FILE, ORPHEUS_GGUF_REPO, WHISPER, WHISPER_FASTER

try:
    from mlx_audio.tts import load as _mlx_load

    HAS_MLX = True
except ImportError:
    _mlx_load = None
    HAS_MLX = False


def download(model):
    print(f"Downloading {model}", flush=True)
    snapshot_download(model, ignore_patterns=["*.md", "*.png", "*.jpg"], max_workers=4)


if __name__ == "__main__":
    if HAS_MLX and _mlx_load is not None:
        args = sys.argv[1:]
        if "--all" in args:
            selected_keys = list(MODELS.keys())
        elif args:
            selected_keys = [key for key in args if key in MODELS]
        else:
            selected_keys = list(CORE_MODELS)

        repos = [MODELS[key] for key in selected_keys]
        targets = (*repos, WHISPER) if not args or "--all" in args else tuple(repos)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(download, targets))
        for repo in repos:
            print(f"Validating {repo}", flush=True)
            model = _mlx_load(repo)
            del model
    else:
        print(f"Downloading Orpheus-3B Q4_K_M GGUF ({ORPHEUS_GGUF_REPO}/{ORPHEUS_GGUF_FILE})...", flush=True)
        hf_hub_download(repo_id=ORPHEUS_GGUF_REPO, filename=ORPHEUS_GGUF_FILE)
        print("Downloading SNAC 24 kHz vocoder (hubertsiuzdak/snac_24khz)...", flush=True)
        snapshot_download("hubertsiuzdak/snac_24khz", ignore_patterns=["*.md", "*.png", "*.jpg"], max_workers=4)
        print(f"Warming faster-whisper ({WHISPER_FASTER})...", flush=True)
        try:
            from faster_whisper import WhisperModel

            fw = WhisperModel(WHISPER_FASTER, device="cpu", compute_type="int8")
            del fw
        except Exception as exc:
            print(f"Note: faster-whisper warmup deferred ({exc})", flush=True)
        print("Windows/CUDA Stage 1 models ready.", flush=True)
