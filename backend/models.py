MODELS = {
    "orpheus_seedvc": "mlx-community/orpheus-3b-0.1-ft-8bit",
}

CORE_MODELS = ("orpheus_seedvc",)

MODEL_INFO = {
    "orpheus_seedvc": {
        "label": "YouClone Studio (Orpheus-3B + Seed-VC 44kHz)",
        "sample_rate": 48000,
        "cloning": True,
    },
}

# Apple Silicon MLX Whisper model
WHISPER = "mlx-community/whisper-large-v3-turbo"

# Windows / Linux CUDA models (4-bit GGUF fits 4 GB VRAM GPUs like RTX 2050)
ORPHEUS_GGUF_REPO = "isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF"
ORPHEUS_GGUF_FILE = "orpheus-3b-0.1-ft-q4_k_m.gguf"
WHISPER_FASTER = "small.en"
