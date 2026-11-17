# YouClone

YouClone is an Apple Silicon desktop studio for local voice cloning, voice design, and dialogue production.

## Start

Install the prerequisites with Homebrew, then run setup from this directory.

```bash
brew install uv node ffmpeg
./setup_mac_m5.sh
```

The first run downloads JavaScript and Python dependencies and four checkpoints, including Whisper (about 9.6 GB of model files). Setup can run pnpm through npm when pnpm is missing. Once setup finishes, launch without network access with `./setup_mac_m5.sh --offline`.

When present in the project folder, the files `English_voice2.wav`, `Hindi_voice.wav`, and `english.mp3` appear under **Workspace recordings**. Click a file to audition it in the waveform. In **Add sample**, choose the recording, select a section with one speaker, and click **Clean and transcribe**. Review the transcript and save the clip under a profile and delivery style. To add another delivery to a profile, use its existing name. The originals remain in the workspace. The MP3 defaults to a 7.6-second start to skip its leading silence.

## Models

| Model | Guidance |
| --- | --- |
| OmniVoice | Uses a recording and its transcript. Select a clip of up to 10 seconds. The checkpoint ID is `mlx-community/OmniVoice-bfloat16`. Inline `[laughter]` and `[sigh]` are supported by MLX-Audio's OmniVoice tokenizer. |
| ZONOS3-Mini | Streams 44.1 kHz audio. Use a reference of up to 60 seconds or ask the vault to cache a 192-value speaker embedding during preparation. |
| Qwen3 VoiceDesign | Generates a voice from a natural-language description. Combine delivery, pitch, and speaking style in the prompt. |

Use `[pause 0.5s]` or `<break time="0.8s" />` to set a pause at a specific point. The optional streaming audition plays raw generated PCM. Saved audio receives the configured speed, joins, pauses, and Studio Polish.

## Shortcuts

| Shortcut | Action |
| --- | --- |
| ⌘ Enter | Generate or regenerate the focused block |
| ⌘ Shift Enter | Generate the nonempty blocks |
| Space | Play or pause the waveform outside editable fields |
| ⌘ R | Open the voice vault recorder |
| ⌘ [ / ] | Move between saved takes |
| ⌘ E | Export the master as WAV to the watch folder |

Set `VOXM5_WATCH_FOLDER` before launching to customize quick exports. The default is `~/Music/YouClone Exports`. Set `VOXM5_MODEL_IDLE` to the desired timeout in seconds or `0` to retain loaded models for the entire session. SQLite, prepared references, and rendered audio live under Electron’s app-data directory. Downloaded checkpoints live under `.voxm5/huggingface`.

## Source layout

```text
setup_mac_m5.sh
backend/
  download.py
  models.py
  server.py
  engine/
    dsp.py
    mlx_tts.py
    transcriber.py
desktop/
  main.cjs
  preload.cjs
  store.cjs
src/
  app/
    page.tsx
    timeline.tsx
    vault.tsx
    studio.css
  db/
    schema.ts
```
