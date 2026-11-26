# YouClone

**Local, private AI voice cloning, character voice design & expressive speech studio** for macOS (Apple Silicon) and Windows (NVIDIA CUDA).

YouClone runs a **2-stage neural voice pipeline** completely on your own hardware — zero cloud fees, zero voice uploads, and zero subscriptions:

1. **Stage 1 — Expressive Voice Acting (`Orpheus-3B`)**: Generates natural pacing, dramatic emphasis, pauses, and paralinguistic cues (`<sigh>`, `<chuckle>`, `<laugh>`, `<gasp>`), verified beat-by-beat by a local **Whisper accuracy gate** to prevent mispronunciations or hallucinated words.
2. **Stage 2 — 44.1 kHz Timbre Lock (`Seed-VC F0-Conditioned BigVGAN`)**: Locks the expressive performance onto any cloned or designed character voice while preserving pitch dynamics, breath, and studio clarity — mastered to 48 kHz broadcast audio.

---

## Features

- **Three-Mode Voice Creation Studio (`+ Create Voice`)**:
  - **Instant Voice Clone (2–5 min audio, min 15s)**: Extracts a true median-F0 vocal reference in seconds — works on both Apple Silicon Macs and 4 GB+ NVIDIA GPUs.
  - **Professional Voice Clone / PVC (20+ min audio)**: Full 2-stage local fine-tuning (Orpheus-3B attention QLoRA + Seed-VC 44.1 kHz DiT) for maximum likeness, accent, and articulation matching.
  - **Voice Design (Prompt-to-Character, Zero Audio Needed)**: Invent brand-new synthetic voices for mysterious villains, cartoon creatures, movie-trailer narrators, or fictional personas from a text prompt. Audition **3 distinct variations**, sculpt vocal formants & pitch (`-5` to `+5` semitones), and save any voice permanently to your **Character Voices** library.
- **16 Built-In Character Archetypes Across 4 Categories**:
  - **Mysterious & Villain**: *Shadow Mastermind*, *Noir Detective*, *Ancient Oracle*, *Cold Sci-Fi Villain*
  - **Cartoon & Creature**: *Little Mouse*, *Angry Pirate*, *Mischievous Goblin*, *Colossal Titan*
  - **Male Characters**: *Movie Trailer*, *British CEO*, *Grizzled Cowboy*, *Wise Old Wizard*
  - **Female Characters**: *Southern Storyteller*, *Cyberpunk AI*, *Regal Queen*, *Late-Night Host*
- **Zero GPU Overlap Architecture**: Unloads Stage 1 before running Whisper verification, unloads Whisper before Stage 2 Seed-VC, and unloads Voice Design models immediately after preview generation — preventing GPU memory contention on 16 GB Macs and 4 GB NVIDIA GPUs.
- **Full Production Suite**: Speech-to-Speech Voice Changer (44.1 kHz), Sound Effects Synthesizer, Local Whisper Transcription & Timed Dubbing, A/B Voice Compare, Vocal Isolator, Directorial AI, Pronunciation Aliases, and Broadcast / Podcast / Cinematic mastering presets.
- **Native Desktop App**: Minimal SF-style icon interface with waveform scrubbing, region A/B looping, take history, and one-click WAV / MP3 / FLAC / M4A + SRT subtitle export.

---

## Hardware Requirements

| Platform | Minimum Spec | Supported Modes |
| :--- | :--- | :--- |
| **macOS (Apple Silicon)** | M1 / M2 / M3 / M4 / M5 · **16 GB+ Unified Memory** · ~15 GB free disk | Full speech synthesis, Instant Voice Clone, **Voice Design** (`Qwen3-TTS-1.7B-VoiceDesign-8bit`), **and** 2-stage PVC Training |
| **Windows 10 / 11 (NVIDIA CUDA)** | 6-core CPU (e.g., Ryzen 7 7445HS) · **RTX 2050 4 GB VRAM+** · **16 GB RAM** · ~12 GB free disk | Full speech synthesis (4-bit Q4_K_M Orpheus + fp16 Seed-VC), **Instant Voice Clone**, & **Voice Design** (Multi-Actor + Formant Sculptor). *PVC training requires ≥ 8 GB VRAM.* |
| **Windows / Linux (CPU-only)** | Modern 8-core CPU · 16 GB RAM | Supported via `-CpuOnly` flag (slower generation) |

---

## Windows Setup (NVIDIA GPU — RTX 2050 4 GB & Up)

### 1. Install Prerequisites (One Time)
Open **PowerShell** and install Git, Node.js LTS, and FFmpeg using `winget`:

```powershell
winget install Git.Git
winget install OpenJS.NodeJS.LTS
winget install Gyan.FFmpeg
```

> Ensure up-to-date **NVIDIA Game Ready / Studio Drivers** are installed (`nvidia-smi` should work in PowerShell). Close and reopen PowerShell after installing prerequisites so `git`, `node`, and `ffmpeg` are on your `PATH`.

### 2. Clone & Run First-Time Setup
```powershell
git clone https://github.com/cyberzapp/YouClone.git
cd YouClone
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup_windows.ps1
```

### 3. Daily Launch & One-Command Updates (Windows)
- **Launch offline anytime** (after setup):
  ```powershell
  cd YouClone
  .\setup_windows.ps1 -Offline
  ```
- **Update an existing Windows installation to the latest GitHub release** (keeps your existing `.venv`, downloaded AI models, and saved voices intact):
  ```powershell
  cd YouClone
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\setup_windows.ps1 -Update
  ```

> **Tips for 4 GB VRAM GPUs (RTX 2050 / RTX 3050 4 GB):**
> - Close heavy GPU apps (games, video editors) while generating long scripts.
> - Use **Instant Voice Clone** (2–5 min audio) and **Voice Design** freely on 4 GB VRAM.
> - Local **Professional Voice Clone (PVC)** training is automatically guarded below 8 GB VRAM to prevent CUDA out-of-memory crashes.

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
- **Build standalone `/Applications/YouClone.app`**:
  ```bash
  npm run package:mac
  ```
- **Update an existing macOS installation**:
  ```bash
  git pull origin main
  npm install && npm run build
  npm run package:mac
  ```

---

## Using Voice Creation & Character Voice Design

1. Open YouClone and wait for the topbar status indicator to show **Ready**.
2. Click **`+ Create Voice`** in the top-right toolbar to open the 3-mode Voice Creation Studio:
   - **Instant Voice Clone**: Upload **2–5 minutes** of clean speech in a quiet room to clone a real voice in seconds.
   - **Professional Voice Clone (PVC)**: Upload **20+ minutes** of clean expressive speech for deep 2-stage local fine-tuning (*Apple Silicon 16 GB+ or NVIDIA 8 GB+ VRAM*).
   - **Voice Design (`Create New Character`)**:
     1. Pick a category (*Mysterious*, *Cartoon & Creature*, *Male*, *Female*) and click any of the **16 starter archetypes** — or write your own description (age, gender, pitch, rasp, accent, mood).
     2. Adjust **Character Vocal Sculpt** (*Natural Human*, *Cartoon / Animated*, *Gritty / Raspy*, *Colossal Bass / Creature*, *Ethereal*) and the **Formant & Pitch Tweak** slider (`-5.0` to `+5.0` semitones).
     3. Click **`Generate 3 Voice Options`** to audition 3 unique voice variations (`Balanced`, `Expressive`, `Bold Character`).
     4. Name your character, pick a category, and click **`Save Character Voice`** — locking that character's timbre permanently so it stays consistent across full-length scripts!
3. Select any Cloned or Character Voice in the composer bar, paste your script (supports `<sigh>`, `<chuckle>`, `<laugh>`, `<gasp>`, and `[pause 0.5s]`), and press **`Generate Speech`** (`⌘↵` / `Ctrl+Enter`).

---

## Privacy & Data Safety

- **100% On-Device**: Voice recordings, designed character references, trained checkpoints, scripts, and rendered audio never leave your computer (`~/Library/Application Support/YouClone` on macOS, `%APPDATA%\YouClone` on Windows).
- **Strict `.gitignore`**: All audio/video media (`*.wav`, `*.mp3`, `*.mp4`, etc.), model weights & LoRAs (`*.safetensors`, `*.pth`, `*.gguf`, `*.pt`, `*.bin`), SQLite databases, `.voxm5/`, `.venv/`, `.env*`, and local archive folders are ignored so personal voice data and credentials are never committed to GitHub.

---

## Troubleshooting

- **Windows `Set-ExecutionPolicy` blocked `setup_windows.ps1`**: Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window first.
- **Windows `better-sqlite3` rebuild error**: Install [Visual Studio Build Tools ("Desktop development with C++")](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and rerun `npm run rebuild`.
- **First Seed-VC generation takes extra time**: On the very first voice-cloned or character-locked generation on a new PC, Seed-VC downloads its ~1.6 GB 44.1 kHz weights (`DiT`, `BigVGAN`, `RMVPE`, `CAM++`) into `.voxm5/seed-vc/checkpoints/`. Keep internet connected for that first run; subsequent runs work completely offline.
