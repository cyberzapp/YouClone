# YouClone

**Local, private AI voice cloning & expressive speech studio** for macOS (Apple Silicon) and Windows (NVIDIA CUDA).

YouClone runs a **2-stage neural voice pipeline** completely on your own machine — zero cloud fees, zero voice uploads, and zero subscriptions:

1. **Stage 1 — Expressive Voice Acting (`Orpheus-3B`)**: Generates natural pacing, emphasis, pauses, and paralinguistic cues (`<sigh>`, `<chuckle>`, `<laugh>`, `<gasp>`), verified beat-by-beat by a local **Whisper accuracy gate** to eliminate mispronunciations.
2. **Stage 2 — 44.1 kHz Timbre Lock (`Seed-VC F0-Conditioned BigVGAN`)**: Converts the expressive performance into your exact cloned voice while preserving pitch dynamics, breath, and clarity — mastered to 48 kHz broadcast audio.

---

## Features

- **Two-Tier Voice Cloning**:
  - **Instant Voice Clone (2–5 min audio, min 15s)**: Extracts a true median-F0 vocal reference in seconds — works on both Apple Silicon Macs and 4 GB+ NVIDIA GPUs.
  - **Professional Voice Clone / PVC (20+ min audio)**: Full 2-stage local fine-tuning (Orpheus-3B attention QLoRA + Seed-VC 44.1 kHz DiT) for studio-grade timbre, cadence, and articulation matching.
- **Zero GPU Overlap Architecture**: Automatically unloads Stage 1 before running Whisper verification, and unloads Whisper before Stage 2 Seed-VC — keeping peak memory low and preventing GPU timeouts.
- **Production Tools**: Speech-to-Speech Voice Changer (44.1 kHz), Sound Effects Synthesizer, Local Whisper Transcription & Timed Dubbing, A/B Voice Compare, Vocal Isolator, Directorial AI, Pronunciation Aliases, and Broadcast / Podcast / Cinematic mastering presets.
- **Native Desktop App**: Clean Electron + Next.js desktop interface with waveform scrubbing, region A/B looping, take history, and one-click WAV / MP3 / FLAC / M4A + SRT subtitle export.

---

## Hardware Requirements

| Platform | Minimum Spec | Supported Modes |
| :--- | :--- | :--- |
| **macOS (Apple Silicon)** | M1 / M2 / M3 / M4 / M5 · **16 GB+ Unified Memory** · ~12 GB free disk | Full speech synthesis, Instant Voice Clone, **and** 2-stage PVC Training |
| **Windows 10 / 11 (NVIDIA CUDA)** | 6-core CPU (e.g., Ryzen 7 7445HS) · **RTX 2050 4 GB VRAM+** · **16 GB RAM** · ~12 GB free disk | Full speech synthesis (4-bit Q4_K_M Orpheus + fp16 Seed-VC) & **Instant Voice Clone** (2–5 min). *PVC fine-tuning requires ≥ 8 GB VRAM.* |
| **Windows / Linux (CPU-only)** | Modern 8-core CPU · 16 GB RAM | Works via `-CpuOnly` flag (slower generation) |

---

## Windows Setup (NVIDIA GPU — RTX 2050 4 GB & Up)

### 1. Install Prerequisites (One Time)
Open **PowerShell** (regular user is fine) and install Git, Node.js LTS, and FFmpeg using `winget`:

```powershell
winget install Git.Git
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg
```

> Make sure you have up-to-date **NVIDIA Game Ready / Studio Drivers** installed (`nvidia-smi` should work in PowerShell). Close and reopen PowerShell after installing prerequisites so `git`, `node`, and `ffmpeg` are on your `PATH`.

### 2. Clone & Run the Automated Installer
```powershell
git clone https://github.com/cyberzapp/YouClone.git
cd YouClone
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

What `setup_windows.ps1` does automatically:
1. Installs [`uv`](https://docs.astral.sh/uv/) (if missing) and Node/Electron packages.
2. Creates a Python 3.12 `.venv` with **PyTorch CUDA 12.4**, **`llama-cpp-python` (CUDA)**, **`snac`**, and **`faster-whisper`**.
3. Sets up the isolated **Seed-VC 44.1 kHz** worker environment inside `.voxm5\seed-vc`.
4. Downloads the 4-bit **Orpheus-3B Q4_K_M GGUF** (~2.1 GB), **SNAC 24 kHz**, and **Whisper** models locally into `.voxm5\huggingface`. *(Seed-VC's 44.1 kHz checkpoints download automatically on your first voice-clone generation.)*
5. Builds the UI and launches **YouClone**!

### 3. Daily Offline Launch (Windows)
After the first setup completes, launch YouClone anytime (100% offline) with:

```powershell
cd YouClone
.\setup_windows.ps1 -Offline
```

> **Tip for 4 GB VRAM GPUs (RTX 2050 / RTX 3050 4 GB):**
> - Close heavy GPU apps (games, video editors, Chrome with 40 tabs) while generating long scripts.
> - Use **Instant Voice Clone** (`+ Create Voice` → upload **2–5 minutes** of clean speech) for fast, realistic cloning.
> - Local **Professional Voice Clone (PVC)** training is automatically blocked below 8 GB VRAM to prevent CUDA out-of-memory crashes; if you train a PVC checkpoint on a Mac or 8 GB+ GPU, you can copy it over and synthesize with it on 4 GB VRAM.

---

## macOS Setup (Apple Silicon M1–M5)

### 1. Install Prerequisites
```bash
brew install ffmpeg node uv
```

### 2. Clone, Setup & Launch
```bash
git clone https://github.com/cyberzapp/YouClone.git
cd YouClone
./setup_mac_m5.sh
```

- **Daily offline launch**: `./setup_mac_m5.sh --offline`
- **Build standalone macOS `/Applications/YouClone.app`**:
  ```bash
  npm run package:mac
  ```

---

## Quick Start: Cloning Your Voice

1. Open YouClone and wait for the topbar status dot to turn green (**Ready**).
2. Click **`+ Create Voice`** in the top-right toolbar.
3. Choose:
   - **Instant Voice Clone**: Upload **2–5 minutes** of clean, quiet-room speech (WAV/MP3/M4A/FLAC). Ready in seconds.
   - **Professional Voice Clone (PVC)** *(Apple Silicon 16 GB+ or NVIDIA 8 GB+ VRAM)*: Upload **20+ minutes** of clean expressive speech for deep local fine-tuning.
4. Select your new voice in the composer bar, paste any script (supports `<sigh>`, `<chuckle>`, `<laugh>`, `<gasp>`, and `[pause 0.5s]`), and press **`Generate Speech`** (`⌘↵` / `Ctrl+Enter`).

---

## Privacy & Data Safety

- **100% Local**: Your voice recordings, trained weights, scripts, and generated audio stay strictly on your own device (`~/Library/Application Support/YouClone` on macOS, `%APPDATA%\YouClone` on Windows).
- **Strict `.gitignore`**: All audio/video media (`*.wav`, `*.mp3`, `*.mp4`, etc.), model weights & LoRAs (`*.safetensors`, `*.pth`, `*.gguf`, `*.pt`), local SQLite databases, `.voxm5/`, `.venv/`, `.env*`, and local archive folders are ignored so personal voice data and secrets are never pushed to GitHub.

---

## Troubleshooting

- **Windows `Set-ExecutionPolicy` blocked `setup_windows.ps1`**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window before `.\setup_windows.ps1`.
- **Windows `better-sqlite3` rebuild error**: Install [Visual Studio Build Tools ("Desktop development with C++")](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and rerun `npm run rebuild`.
- **First Seed-VC generation takes extra time on Windows**: On the very first voice-cloned generation, Seed-VC downloads its ~1.6 GB 44.1 kHz weights (`DiT`, `BigVGAN`, `RMVPE`, `CAM++`) into `.voxm5\seed-vc\checkpoints\`. Keep internet connected for that first run; every run after that works offline.
