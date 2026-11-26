import asyncio
import hmac
import json
import logging
import os
import queue
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, field_validator

from .engine import dsp, lora
from .engine.mlx_tts import Engine
from .engine.transcriber import transcribe
from .engine.voice_design import check_voice_design_cached, generate_voice_design_candidates
from .engine.voice_studio import MIN_PROFESSIONAL_SECONDS, VoiceStudioStore
from .models import MODEL_INFO, MODELS

DATA = Path(os.environ["VOXM5_DATA"]).resolve()
TOKEN = os.environ["VOXM5_TOKEN"]
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DEFAULT = WORKSPACE_ROOT / ".voxm5" / "default_voice"
for folder in ("samples", "takes", "exports", "adapters", "voices_studio"):
    (DATA / folder).mkdir(parents=True, exist_ok=True)

engine = Engine()
voice_store = VoiceStudioStore(DATA)

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1 GB (supports 20-60+ min 48kHz WAV uploads for PVC)


def _ensure_default_voice():
    if BUNDLE_DEFAULT.is_dir():
        for folder in ("samples", "adapters"):
            src_dir = BUNDLE_DEFAULT / folder
            if not src_dir.is_dir():
                continue
            for item in src_dir.iterdir():
                if item.is_file():
                    dest = DATA / folder / item.name
                    if not dest.exists():
                        shutil.copy2(item, dest)
        manifest_src = BUNDLE_DEFAULT / "manifest.json"
        manifest_dest = DATA / "default_voice.json"
        if manifest_src.is_file() and not manifest_dest.exists():
            shutil.copy2(manifest_src, manifest_dest)
    voice_store.ensure_default_voices()


def _load_default_manifest() -> dict:
    for candidate in (DATA / "default_voice.json", BUNDLE_DEFAULT / "manifest.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


_ensure_default_voice()
worker = ThreadPoolExecutor(max_workers=1)
logger = logging.getLogger("uvicorn.error")
prewarm_error = None

lora_state = {
    "running": False,
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "message": "",
    "error": None,
    "last_adapter": None,
}

pvc_state = {
    "running": False,
    "stage": "",
    "percent": 0,
    "message": "",
    "error": None,
    "voice": None,
}


async def run(function, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(worker, partial(function, *args, **kwargs))


@asynccontextmanager
async def lifespan(_):
    async def maintain():
        global prewarm_error
        try:
            def _warm():
                _ensure_default_voice()
                engine.load("orpheus_seedvc")
            await run(_warm)
        except Exception:
            prewarm_error = "Orpheus-3B checkpoint is unavailable. Check .voxm5/huggingface and restart."
            logger.exception("Orpheus-3B prewarm failed")
        while True:
            await asyncio.sleep(25)
            await run(engine.evict)

    task = asyncio.create_task(maintain())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    engine.seed_vc.shutdown()
    worker.shutdown(wait=True, cancel_futures=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def authorized(token: str | None):
    return hmac.compare_digest((token or "").encode(), TOKEN.encode())


def authenticate(authorization: str = Header(default="")):
    if not authorized(authorization.removeprefix("Bearer ")):
        raise HTTPException(403, "Unauthorized")


ModelKey = Literal["orpheus_seedvc"]


class Settings(BaseModel):
    model: ModelKey = "orpheus_seedvc"
    text: str = Field(min_length=1, max_length=40000)
    studio_voice_id: str = Field(default="", max_length=120)
    sample: UUID | None = None
    transcript: str = Field(default="", max_length=20000)
    language: str = Field(default="", max_length=50)
    style: str = Field(default="", max_length=1000)
    embedding: list[float] | None = Field(default=None, min_length=192, max_length=192)
    temperature: float = Field(default=0.58, ge=0, le=2)
    top_p: float = Field(default=0.86, gt=0, le=1)
    repetition_penalty: float = Field(default=1.18, ge=1, le=2)
    guidance: float = Field(default=2.0, ge=0, le=8)
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    seed: int = Field(default=202, ge=0, le=2**32 - 1)
    comma_pause: float = Field(default=0.22, ge=0, le=3)
    period_pause: float = Field(default=0.34, ge=0, le=3)
    polish: bool = True
    num_steps: int = Field(default=35, ge=15, le=60)
    class_temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    position_temperature: float = Field(default=5.0, ge=0.0, le=10.0)
    lora_adapter: str = Field(default="", max_length=120)
    stability: float = Field(default=55, ge=0, le=100)
    similarity_boost: float = Field(default=75, ge=0, le=100)
    style_exaggeration: float = Field(default=45, ge=0, le=100)
    speaker_boost: bool = True
    effect_preset: Literal["broadcast", "podcast", "cinematic", "warm", "bright", "raw"] = "broadcast"
    pause_fill: Literal["silence", "room_tone"] = "silence"
    trim_silence: bool = True
    filter_mode: Literal["none", "phone", "radio", "robot", "megaphone"] = "none"
    space_mode: Literal["none", "room", "hall", "cathedral"] = "none"
    distance_mode: Literal["near", "medium", "far"] = "near"
    ambience_mode: Literal["none", "vinyl", "rain", "tape", "cafe"] = "none"
    watermark: bool = False

    @field_validator("model", mode="before")
    @classmethod
    def coerce_legacy_model(cls, value):
        return "orpheus_seedvc"

    @field_validator("embedding")
    @classmethod
    def finite_embedding(cls, value):
        if value is not None and not np.isfinite(value).all():
            raise ValueError("Embedding must contain finite values.")
        return value


@app.get("/health", dependencies=[Depends(authenticate)])
def health():
    return {"ready": True, "modelWarning": prewarm_error}


async def save_upload(file: UploadFile, destination: Path, max_bytes: int = MAX_UPLOAD_BYTES) -> int:
    size = 0
    with destination.open("wb") as handle:
        while chunk := await file.read(2 * 1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit.")
            handle.write(chunk)
    return size


@app.post("/transcribe", dependencies=[Depends(authenticate)])
async def transcribe_media(file: UploadFile = File()):
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            media = Path(temp) / "input"
            await save_upload(file, media)
            result = await run(transcribe, media)
            segments = [
                {"start": row["start"], "end": row["end"], "text": row["text"].strip()}
                for row in result["segments"]
                if row["text"].strip()
            ]
            return {"text": result["text"].strip(), "language": result["language"], "segments": segments}
    except Exception as exc:
        logger.exception("Media transcription failed")
        raise HTTPException(422, "Could not transcribe this recording. " + str(exc)) from exc


@app.post("/sample", dependencies=[Depends(authenticate)])
async def prepare(
    file: UploadFile = File(),
    start: float = Form(0, ge=0),
    duration: float = Form(14, ge=1, le=60),
    cache_embedding: bool = Form(False),
):
    ident = uuid4()
    destination = DATA / "samples" / f"{ident}.wav"
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            source = Path(temp) / "input"
            await save_upload(file, source)
            await run(dsp.sanitize, source, destination, start, duration)
            result = await run(transcribe, destination)
            return {
                "id": ident,
                "embedding": None,
                "transcript": result["text"].strip(),
                "language": result["language"],
                "seconds": sf.info(destination).duration,
            }
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Sample preparation failed")
        raise HTTPException(422, str(exc)) from exc


# ==================== Studio Voices (Instant & Professional Clone) ====================


@app.get("/studio-voices", dependencies=[Depends(authenticate)])
async def list_studio_voices():
    voices = await asyncio.to_thread(voice_store.list_voices)
    return {
        "voices": voices,
        "training": pvc_state,
        "seedvc_ready": engine.seed_vc.is_available(),
        "voice_design_cached": check_voice_design_cached(),
    }


class VoiceDesignPreviewRequest(BaseModel):
    prompt: str = Field(min_length=4, max_length=600)
    sample_text: str = Field(
        default="Listen closely... not everything hidden in the dark wants to be found.",
        min_length=6,
        max_length=450,
    )
    pitch_semitones: float = Field(default=0.0, ge=-6.0, le=6.0)
    texture: Literal["natural", "cartoon", "deep_creature", "gritty", "ethereal"] = "natural"
    seed: int = Field(default=108, ge=0, le=2**31 - 1)


@app.post("/studio-voices/design/preview", dependencies=[Depends(authenticate)])
async def preview_voice_design_endpoint(req: VoiceDesignPreviewRequest):
    if pvc_state["running"]:
        raise HTTPException(409, "Wait for Professional Voice Clone training to finish before designing voices.")
    try:
        async with inference_lock:
            candidates = await asyncio.to_thread(
                generate_voice_design_candidates,
                engine,
                prompt=req.prompt,
                sample_text=req.sample_text,
                pitch_semitones=req.pitch_semitones,
                texture=req.texture,
                seed=req.seed,
                takes_dir=DATA / "takes",
                num_candidates=3,
            )
        return {"candidates": candidates, "voice_design_cached": True}
    except Exception as exc:
        logger.exception("Voice design preview failed")
        raise HTTPException(422, str(exc)) from exc


class VoiceDesignSaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    category: str = Field(default="mysterious", max_length=40)
    prompt: str = Field(min_length=4, max_length=600)
    candidate_id: UUID
    gender: str = Field(default="auto", max_length=20)


@app.post("/studio-voices/design/save", dependencies=[Depends(authenticate)])
async def save_voice_design_endpoint(req: VoiceDesignSaveRequest):
    cand_path = DATA / "takes" / f"{req.candidate_id}.wav"
    try:
        entry = await asyncio.to_thread(
            voice_store.create_designed_voice,
            name=req.name,
            category=req.category,
            prompt=req.prompt,
            candidate_wav=cand_path,
            gender_hint=req.gender,
        )
        return {"voice": entry, "voices": voice_store.list_voices()}
    except Exception as exc:
        logger.exception("Saving designed voice failed")
        raise HTTPException(422, str(exc)) from exc


@app.post("/studio-voices/instant", dependencies=[Depends(authenticate)])
async def create_instant_voice_endpoint(
    name: str = Form(..., min_length=1, max_length=80),
    gender: str = Form("auto"),
    files: list[UploadFile] = File(...),
):
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            staged: list[Path] = []
            total_bytes = 0
            for idx, upload in enumerate(files):
                ext = Path(upload.filename or f"clip_{idx}.wav").suffix or ".wav"
                dst = Path(temp) / f"upload_{idx}{ext}"
                total_bytes += await save_upload(upload, dst, max_bytes=MAX_UPLOAD_BYTES - total_bytes)
                staged.append(dst)
            entry = await asyncio.to_thread(voice_store.create_instant_voice, name, staged, gender)
            return {"voice": entry, "voices": voice_store.list_voices()}
    except Exception as exc:
        logger.exception("Instant voice creation failed")
        raise HTTPException(422, str(exc)) from exc


@app.post("/studio-voices/professional", dependencies=[Depends(authenticate)])
async def create_professional_voice_endpoint(
    name: str = Form(..., min_length=1, max_length=80),
    steps: int = Form(500, ge=200, le=1200),
    files: list[UploadFile] = File(...),
):
    if pvc_state["running"]:
        raise HTTPException(409, "A Professional Voice Clone training job is already running.")

    staging_dir = DATA / "voices_studio" / f"_stage_{uuid4().hex[:8]}"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    try:
        total_bytes = 0
        for idx, upload in enumerate(files):
            ext = Path(upload.filename or f"clip_{idx}.wav").suffix or ".wav"
            dst = staging_dir / f"upload_{idx}{ext}"
            total_bytes += await save_upload(upload, dst, max_bytes=MAX_UPLOAD_BYTES - total_bytes)
            staged.append(dst)

        # Quick upfront duration validation (< 20 min rejected immediately!)
        total_sec = 0.0
        for p in staged:
            try:
                total_sec += float(sf.info(str(p)).duration)
            except Exception:
                # Fallback ffprobe/ffmpeg decode length check
                probe_wav = staging_dir / f"_probe_{p.stem}.wav"
                await asyncio.to_thread(dsp.ffmpeg, "-i", p, "-vn", "-ac", 1, "-ar", 16000, "-c:a", "pcm_s16le", probe_wav)
                total_sec += float(sf.info(str(probe_wav)).duration)
                probe_wav.unlink(missing_ok=True)

        if total_sec < MIN_PROFESSIONAL_SECONDS:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise HTTPException(
                422,
                f"Professional Voice Clone requires at least 20 minutes of audio "
                f"(you uploaded {total_sec / 60.0:.1f} min). Add more audio or use Instant Voice Clone (2–5 min).",
            )
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc

    pvc_state.update({
        "running": True,
        "stage": "Queued",
        "percent": 1,
        "message": f"Starting Professional Voice Clone for '{name.strip()}' ({total_sec/60.0:.1f} min)…",
        "error": None,
        "voice": None,
    })

    def _progress(stage: str, percent: int, message: str):
        pvc_state.update({
            "running": True,
            "stage": stage,
            "percent": int(percent),
            "message": message,
            "error": None,
        })

    async def _bg_train():
        try:
            async with inference_lock:
                entry = await run(
                    voice_store.train_professional_voice,
                    engine,
                    name.strip(),
                    staged,
                    seedvc_steps=steps,
                    progress_cb=_progress,
                )
            pvc_state.update({
                "running": False,
                "stage": "Complete",
                "percent": 100,
                "message": f"'{entry['name']}' is ready to use!",
                "error": None,
                "voice": entry,
            })
        except Exception as exc:
            logger.exception("Professional Voice Clone training failed")
            pvc_state.update({
                "running": False,
                "stage": "Failed",
                "percent": 0,
                "message": "Training failed",
                "error": str(exc),
                "voice": None,
            })
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    asyncio.create_task(_bg_train())
    return {"started": True, "training": pvc_state}


@app.delete("/studio-voices/{voice_id}", dependencies=[Depends(authenticate)])
async def delete_studio_voice_endpoint(voice_id: str):
    try:
        removed = await asyncio.to_thread(voice_store.delete_voice, voice_id)
        if not removed:
            raise HTTPException(404, "Voice not found.")
        return {"deleted": voice_id, "voices": voice_store.list_voices()}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


# ==================== Audio Finalization & Generation ====================


def _finalize_master_audio(master: np.ndarray, settings: Settings) -> tuple[UUID, float]:
    ident = uuid4()
    dest = DATA / "takes" / f"{ident}.wav"
    if settings.trim_silence:
        master = dsp.trim_speech_padding_np(master, dsp.RATE)
    has_fx = (
        settings.filter_mode != "none"
        or settings.space_mode != "none"
        or settings.distance_mode != "near"
        or settings.ambience_mode != "none"
    )
    with TemporaryDirectory(dir=DATA) as temp:
        raw = Path(temp) / "raw.wav"
        result = Path(temp) / "finished.wav"
        sf.write(raw, master, dsp.RATE, subtype="FLOAT")

        if settings.polish and not has_fx and settings.effect_preset == "broadcast":
            # Clean transparent 48 kHz studio broadcast master
            dsp.ffmpeg(
                "-i", raw,
                "-af", "highpass=f=80,loudnorm=I=-14.0:TP=-1.0:LRA=11",
                "-ar", dsp.RATE, "-ac", 1, "-c:a", "pcm_s16le", result,
            )
        elif settings.polish or has_fx or settings.effect_preset != "raw":
            dsp.polish(
                raw,
                result,
                effect_preset=settings.effect_preset if settings.polish else "raw",
                speaker_boost=settings.speaker_boost,
                filter_mode=settings.filter_mode,
                space_mode=settings.space_mode,
                distance_mode=settings.distance_mode,
                ambience_mode=settings.ambience_mode,
                seed=settings.seed,
            )
        else:
            sf.write(result, master, dsp.RATE, subtype="PCM_16")

        if settings.watermark:
            sig, _ = sf.read(result, dtype="float32")
            sf.write(result, dsp.embed_watermark_np(sig, dsp.RATE), dsp.RATE, subtype="PCM_16")
            dsp.append_wav_provenance_tag(result, model_id=settings.model)
        result.replace(dest)
    return ident, sf.info(dest).duration


def _resolve_voice_profile(settings: Settings) -> dict:
    # Prefer explicitly chosen Studio Voice (Instant or Professional)
    if settings.studio_voice_id:
        profile = voice_store.get_voice(settings.studio_voice_id)
        if profile:
            return profile
    # Fallback to legacy Vault sample if provided
    if settings.sample:
        sample_wav = DATA / "samples" / f"{settings.sample}.wav"
        if sample_wav.is_file():
            return {
                "id": str(settings.sample),
                "name": "Custom Sample",
                "mode": "instant",
                "orpheus_adapter": None,
                "orpheus_voice": "leo",
                "seedvc_checkpoint": None,
                "seedvc_config": None,
                "ref_audio": str(sample_wav),
                "diffusion_steps": settings.num_steps,
                "cfg_rate": 0.72,
            }
    default_v = voice_store.get_voice(None)
    if not default_v:
        raise ValueError("No studio voice profile is available. Create a voice in Voice Creation.")
    return default_v


inference_lock = asyncio.Lock()


@app.websocket("/generate")
async def generate(socket: WebSocket):
    if not authorized(socket.query_params.get("token")):
        await socket.close(code=1008)
        return
    await socket.accept()
    try:
        settings = Settings.model_validate(await socket.receive_json())
        if pvc_state["running"]:
            await socket.send_json({"type": "error", "message": "Wait for Professional Voice training to finish."})
            await socket.close()
            return

        voice_profile = _resolve_voice_profile(settings)
        if not engine.check_model_cached("orpheus_seedvc"):
            raise ValueError("Orpheus-3B checkpoint is not cached in .voxm5/huggingface.")

        ev_queue: queue.Queue = queue.Queue()
        done_flag = threading.Event()
        result_box: dict = {}

        def _synth_worker():
            try:
                master = engine.synthesize_studio_speech(
                    text=settings.text,
                    voice_profile=voice_profile,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    repetition_penalty=settings.repetition_penalty,
                    stability=settings.stability,
                    similarity_boost=settings.similarity_boost,
                    style_exaggeration=settings.style_exaggeration,
                    period_pause=settings.period_pause,
                    diffusion_steps=settings.num_steps,
                    seed=settings.seed,
                    progress_cb=lambda ev: ev_queue.put(ev),
                )
                if abs(settings.speed - 1.0) > 0.015:
                    master = dsp.tempo(master, float(np.clip(settings.speed, 0.7, 1.3)))
                ident, duration = _finalize_master_audio(master, settings)
                result_box["done"] = {"type": "done", "id": str(ident), "seconds": duration}
            except Exception as exc:
                logger.exception("Generation failed")
                result_box["error"] = str(exc)
            finally:
                done_flag.set()

        async with inference_lock:
            fut = asyncio.get_running_loop().run_in_executor(worker, _synth_worker)
            while not done_flag.is_set() or not ev_queue.empty():
                try:
                    ev = ev_queue.get_nowait()
                    await socket.send_json(ev)
                except queue.Empty:
                    await asyncio.sleep(0.12)
            await fut
            if "error" in result_box:
                await socket.send_json({"type": "error", "message": result_box["error"]})
            elif "done" in result_box:
                await socket.send_json(result_box["done"])
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if not isinstance(exc, ValueError):
            logger.exception("Generation ended")
        with suppress(Exception):
            await socket.send_json({"type": "error", "message": str(exc)})
    finally:
        with suppress(Exception):
            await socket.close()


@app.get("/models/status", dependencies=[Depends(authenticate)])
async def models_status():
    def _collect():
        return {
            "models": [
                {
                    "key": key,
                    "repo": repo,
                    "label": MODEL_INFO[key]["label"],
                    "sample_rate": MODEL_INFO[key]["sample_rate"],
                    "cloning": MODEL_INFO[key]["cloning"],
                    "cached": engine.check_model_cached(key),
                    "loaded": key in engine.models,
                }
                for key, repo in MODELS.items()
            ],
            "adapters": lora.list_lora_adapters(DATA / "adapters"),
            "active_lora": engine.active_lora,
            "default_voice": _load_default_manifest(),
            "studio_voices": voice_store.list_voices(),
            "pvc_training": pvc_state,
        }

    return await asyncio.to_thread(_collect)


class DownloadModelRequest(BaseModel):
    model: ModelKey


@app.post("/models/download", dependencies=[Depends(authenticate)])
async def download_model_endpoint(req: DownloadModelRequest):
    try:
        repo = await asyncio.to_thread(engine.download_model, req.model)
        return {"model": req.model, "repo": repo, "cached": engine.check_model_cached(req.model)}
    except Exception as exc:
        logger.exception("Model download failed")
        raise HTTPException(422, str(exc)) from exc


@app.get("/lora/status", dependencies=[Depends(authenticate)])
def get_lora_status():
    return {
        **lora_state,
        "adapters": lora.list_lora_adapters(DATA / "adapters"),
        "active_lora": engine.active_lora,
    }


class LoraTrainRequest(BaseModel):
    name: str = Field(default="My Fine-Tuned Voice", min_length=1, max_length=100)
    sources: list[str] = Field(min_length=1, max_length=10)
    rank: int = Field(default=16, ge=4, le=32)
    steps: int = Field(default=60, ge=10, le=300)
    learning_rate: float = Field(default=8e-5, gt=1e-6, le=5e-3)


@app.post("/lora/train", dependencies=[Depends(authenticate)])
async def train_lora_endpoint(req: LoraTrainRequest):
    raise HTTPException(422, "Use Create Voice → Professional Voice Clone (20+ min) for full 2-stage training.")


@app.post("/convert", dependencies=[Depends(authenticate)])
async def convert_voice(
    file: UploadFile = File(),
    studio_voice_id: str = Form(""),
    model: str = Form("orpheus_seedvc"),
    sample: UUID | None = Form(None),
    transcript: str = Form(""),
    language: str = Form(""),
    style: str = Form(""),
    match_timing: bool = Form(True),
    effect_preset: Literal["broadcast", "podcast", "cinematic", "warm", "bright", "raw"] = Form("broadcast"),
):
    """Direct 44.1 kHz Seed-VC Speech-to-Speech Voice Changer (preserves 100% of acting & timing)."""
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            source_media = Path(temp) / "source_input"
            clean_wav = Path(temp) / "source_clean.wav"
            await save_upload(file, source_media)
            await run(dsp.ffmpeg, "-i", source_media, "-vn", "-ac", 1, "-ar", 44100, "-c:a", "pcm_s16le", clean_wav)
            source_duration = sf.info(clean_wav).duration

            settings = Settings(
                studio_voice_id=studio_voice_id,
                sample=sample,
                text="Speech-to-Speech conversion",
                effect_preset=effect_preset,
                polish=effect_preset != "raw",
            )
            voice_profile = _resolve_voice_profile(settings)

            def _convert():
                master = engine.convert_speech_to_speech(
                    clean_wav,
                    voice_profile,
                    diffusion_steps=int(voice_profile.get("diffusion_steps") or 35),
                    cfg_rate=float(voice_profile.get("cfg_rate") or 0.75),
                )
                ident, final_seconds = _finalize_master_audio(master, settings)
                return {
                    "id": str(ident),
                    "seconds": final_seconds,
                    "source_seconds": round(source_duration, 2),
                    "text": f"Converted to {voice_profile.get('name', 'Studio Voice')}",
                    "language": language or "en",
                }

            async with inference_lock:
                return await run(_convert)
    except Exception as exc:
        logger.exception("Voice conversion failed")
        raise HTTPException(422, str(exc)) from exc


@app.post("/isolate", dependencies=[Depends(authenticate)])
async def isolate_audio(file: UploadFile = File()):
    ident = uuid4()
    dest = DATA / "takes" / f"{ident}.wav"
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            source = Path(temp) / "input"
            await save_upload(file, source)
            duration = await run(dsp.isolate_vocals, source, dest)
            return {"id": str(ident), "seconds": round(duration, 2)}
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Voice isolation failed")
        raise HTTPException(422, str(exc)) from exc


class SfxRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    duration: float = Field(default=5.0, ge=1.0, le=30.0)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


@app.post("/sfx", dependencies=[Depends(authenticate)])
async def generate_sfx(req: SfxRequest):
    try:
        def _build_sfx():
            ident = uuid4()
            dest = DATA / "takes" / f"{ident}.wav"
            signal = dsp.generate_sfx_audio(req.prompt, duration=req.duration, seed=req.seed)
            sf.write(dest, signal, dsp.RATE, subtype="PCM_16")
            return {
                "id": str(ident),
                "seconds": round(sf.info(dest).duration, 2),
                "prompt": req.prompt.strip(),
                "seed": req.seed,
            }

        return await run(_build_sfx)
    except Exception as exc:
        logger.exception("SFX generation failed")
        raise HTTPException(422, str(exc)) from exc


@app.post("/watermark/detect", dependencies=[Depends(authenticate)])
async def detect_watermark(file: UploadFile = File()):
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            source = Path(temp) / "input"
            await save_upload(file, source)
            return await run(dsp.detect_watermark_file, source)
    except Exception as exc:
        logger.exception("Watermark detection failed")
        raise HTTPException(422, "Could not analyze audio watermark: " + str(exc)) from exc


class DirectionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


@app.post("/tools/direction", dependencies=[Depends(authenticate)])
def analyze_direction(req: DirectionRequest):
    return dsp.parse_direction(req.text)


class RateFitRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    slot_seconds: float = Field(gt=0.1, le=3600)
    language: str = Field(default="en", max_length=50)


@app.post("/tools/rate-fit", dependencies=[Depends(authenticate)])
def analyze_rate_fit(req: RateFitRequest):
    return dsp.rate_fit_analysis(req.text, req.slot_seconds, req.language)


class ClipTiming(BaseModel):
    start: float = Field(ge=0, le=3600)
    end: float = Field(gt=0, le=3600)


class Export(BaseModel):
    takes: list[UUID] = Field(min_length=1, max_length=500)
    timing: list[ClipTiming] | None = Field(default=None, min_length=1, max_length=500)
    format: Literal["wav", "flac", "mp3", "m4a"] = "wav"
    subtitles: bool = False
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    polish: bool = False
    effect_preset: Literal["broadcast", "podcast", "cinematic", "warm", "bright", "raw"] = "broadcast"
    pause_fill: Literal["silence", "room_tone"] = "silence"
    pause: float = Field(default=0.34, ge=0, le=3)


def render_export(settings: Export):
    ident = uuid4()
    output = DATA / "exports" / f"{ident}.{settings.format}"
    with TemporaryDirectory(dir=DATA) as temp:
        master = Path(temp) / "master.wav"
        finished = Path(temp) / "finished.wav"
        clips = [sf.read(DATA / "takes" / f"{take}.wav", dtype="float32")[0] for take in settings.takes]
        if settings.timing is None:
            mixed = clips[0]
            for idx, clip in enumerate(clips[1:], 1):
                mixed = dsp.join(mixed, clip, settings.pause, pause_fill=settings.pause_fill, seed=3611 + idx)
        else:
            if len(settings.timing) != len(clips):
                raise ValueError("Each take must have a matching time range.")
            mixed = dsp.timed_mix(clips, [(cue.start, cue.end) for cue in settings.timing])
        start = round((settings.start or 0) * dsp.RATE)
        stop = round(settings.end * dsp.RATE) if settings.end is not None else len(mixed)
        if stop <= start or start >= len(mixed):
            raise ValueError("Select a region within the audio.")
        sf.write(master, mixed[start:stop], dsp.RATE, subtype="FLOAT")
        if settings.polish:
            dsp.polish(master, finished, effect_preset=settings.effect_preset)
            master = finished
        args = {
            "wav": ["-c:a", "pcm_s16le"],
            "flac": ["-c:a", "flac", "-sample_fmt", "s16"],
            "mp3": ["-c:a", "libmp3lame", "-q:a", 2],
            "m4a": ["-c:a", "aac", "-b:a", "192k"],
        }[settings.format]
        staged = Path(temp) / output.name
        dsp.ffmpeg("-i", master, *args, staged)
        if settings.subtitles:
            text = dsp.subtitles(transcribe(master, words=True))
            output.with_suffix(".srt").write_text(text, encoding="utf-8")
        staged.replace(output)
    return {"file": output.name, "subtitles": output.with_suffix(".srt").name if settings.subtitles else None}


@app.post("/export", dependencies=[Depends(authenticate)])
async def export(settings: Export):
    try:
        return await run(render_export, settings)
    except Exception as exc:
        logger.exception("Export failed")
        raise HTTPException(422, str(exc)) from exc
