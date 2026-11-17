import asyncio
import hmac
import logging
import os
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

from .engine import dsp
from .engine.mlx_tts import Engine
from .engine.transcriber import transcribe

DATA = Path(os.environ["VOXM5_DATA"]).resolve()
TOKEN = os.environ["VOXM5_TOKEN"]
for folder in ("samples", "takes", "exports"):
    (DATA / folder).mkdir(parents=True, exist_ok=True)
engine = Engine()
worker = ThreadPoolExecutor(max_workers=1)
logger = logging.getLogger("uvicorn.error")
prewarm_error = None


async def run(function, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(worker, partial(function, *args, **kwargs))


@asynccontextmanager
async def lifespan(_):
    async def maintain():
        global prewarm_error
        try:
            await run(engine.load, "omni")
        except Exception:
            prewarm_error = "OmniVoice is unavailable. Check the local checkpoint and restart after setup completes."
            logger.exception("OmniVoice prewarm failed")
        while True:
            await asyncio.sleep(20)
            await run(engine.evict)

    task = asyncio.create_task(maintain())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    worker.shutdown(wait=True, cancel_futures=True)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


def authorized(token: str | None):
    return hmac.compare_digest((token or "").encode(), TOKEN.encode())


def authenticate(authorization: str = Header(default="")):
    if not authorized(authorization.removeprefix("Bearer ")):
        raise HTTPException(403, "Unauthorized")


class Settings(BaseModel):
    model: Literal["omni", "zonos", "qwen"] = "omni"
    text: str = Field(min_length=1, max_length=40000)
    sample: UUID | None = None
    transcript: str = Field(default="", max_length=20000)
    language: str = Field(default="", max_length=50)
    style: str = Field(default="", max_length=1000)
    embedding: list[float] | None = Field(default=None, min_length=192, max_length=192)
    temperature: float = Field(default=.7, ge=0, le=2)
    top_p: float = Field(default=.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.05, ge=1, le=2)
    guidance: float = Field(default=2, ge=0, le=8)
    speed: float = Field(default=1, ge=.7, le=1.3)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    comma_pause: float = Field(default=.25, ge=0, le=3)
    period_pause: float = Field(default=.45, ge=0, le=3)
    polish: bool = True

    @field_validator("embedding")
    @classmethod
    def finite_embedding(cls, value):
        if value is not None and not np.isfinite(value).all():
            raise ValueError("Embedding must contain finite values.")
        return value


@app.get("/health", dependencies=[Depends(authenticate)])
def health():
    return {"ready": True, "modelWarning": prewarm_error}


@app.post("/sample", dependencies=[Depends(authenticate)])
async def prepare(file: UploadFile = File(), start: float = Form(0, ge=0),
                  duration: float = Form(10, ge=1, le=60), cache_embedding: bool = Form(False)):
    ident = uuid4()
    destination = DATA / "samples" / f"{ident}.wav"
    try:
        with TemporaryDirectory(dir=DATA) as temp:
            source = Path(temp) / "input"
            size = 0
            with source.open("wb") as handle:
                while data := await file.read(1024 * 1024):
                    size += len(data)
                    if size > 150 * 1024 * 1024:
                        raise ValueError("Recording exceeds 150 MB.")
                    handle.write(data)
            await run(dsp.sanitize, source, destination, start, duration)
            result = await run(transcribe, destination)
            embedding = await run(engine.speaker_embedding, destination) if cache_embedding else None
            return {"id": ident, "embedding": embedding, "transcript": result["text"].strip(),
                    "language": result["language"], "seconds": sf.info(destination).duration}
    except Exception as exc:
        destination.unlink(missing_ok=True)
        logger.exception("Sample preparation failed")
        raise HTTPException(422, str(exc)) from exc


def generation(settings):
    source = DATA / "samples" / f"{settings.sample}.wav"
    if settings.model == "omni":
        if not source.is_file() or not settings.transcript.strip():
            raise ValueError("OmniVoice requires a saved reference and its transcript.")
        if sf.info(source).duration > 10.01:
            raise ValueError("Select a reference of 10 seconds or less for OmniVoice.")
    if settings.model == "zonos" and not (settings.embedding or source.is_file()):
        raise ValueError("ZONOS3 requires a saved reference or a speaker embedding.")
    if settings.model == "qwen" and not settings.style.strip():
        raise ValueError("Describe the voice you want Qwen to design.")
    parts = dsp.script_parts(settings.text, settings.comma_pause, settings.period_pause)
    if not any(text for text, _ in parts):
        raise ValueError("Enter text to synthesize.")
    master = np.empty(0, dtype=np.float32)
    previous_pause = 0
    for index, (text, pause) in enumerate(parts):
        if not text:
            previous_pause = pause
            continue
        yield {"type": "progress", "current": index + 1, "total": len(parts)}
        clips = []
        for pcm in engine.generate(settings, text, source, (settings.seed + index) % 2**32):
            clips.append(pcm)
            # Stream preview PCM. The file receives final stitching and mastering.
            yield np.asarray(np.clip(pcm, -1, 1) * 32767, dtype="<i2").tobytes()
        if not clips:
            raise ValueError("The model produced no audio.")
        clip = dsp.tempo(np.concatenate(clips), settings.speed)
        if not master.size and previous_pause:
            master = np.zeros(round(previous_pause * dsp.RATE), dtype=np.float32)
            previous_pause = 0
        master = dsp.join(master, clip, previous_pause)
        previous_pause = pause
    ident = uuid4()
    dest = DATA / "takes" / f"{ident}.wav"
    with TemporaryDirectory(dir=DATA) as temp:
        raw = Path(temp) / "raw.wav"
        result = Path(temp) / "finished.wav"
        if settings.polish:
            sf.write(raw, master, dsp.RATE, subtype="FLOAT")
            dsp.polish(raw, result)
        else:
            sf.write(result, master, dsp.RATE, subtype="PCM_16")
        result.replace(dest)
    yield {"type": "done", "id": str(ident), "seconds": sf.info(dest).duration}


inference_lock = asyncio.Lock()


@app.websocket("/generate")
async def generate(socket: WebSocket):
    if not authorized(socket.query_params.get("token")):
        await socket.close(code=1008)
        return
    await socket.accept()
    iterator = None
    try:
        settings = Settings.model_validate(await socket.receive_json())
        async with inference_lock:
            iterator = generation(settings)
            while (item := await run(next, iterator, None)) is not None:
                if isinstance(item, bytes):
                    await socket.send_bytes(item)
                else:
                    await socket.send_json(item)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        if not isinstance(exc, ValueError):
            logger.exception("Generation ended")
        with suppress(Exception):
            await socket.send_json({"type": "error", "message": str(exc)})
    finally:
        if iterator:
            await run(iterator.close)
        with suppress(Exception):
            await socket.close()


class Export(BaseModel):
    takes: list[UUID] = Field(min_length=1, max_length=500)
    format: Literal["wav", "flac", "mp3"] = "wav"
    subtitles: bool = False
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    polish: bool = False
    pause: float = Field(default=.45, ge=0, le=3)


def render_export(settings):
    ident = uuid4()
    output = DATA / "exports" / f"{ident}.{settings.format}"
    with TemporaryDirectory(dir=DATA) as temp:
        master = Path(temp) / "master.wav"
        finished = Path(temp) / "finished.wav"
        clips = [sf.read(DATA / "takes" / f"{take}.wav", dtype="float32")[0] for take in settings.takes]
        mixed = clips[0]
        for clip in clips[1:]:
            mixed = dsp.join(mixed, clip, settings.pause)
        start = round((settings.start or 0) * dsp.RATE)
        stop = round(settings.end * dsp.RATE) if settings.end is not None else len(mixed)
        if stop <= start or start >= len(mixed):
            raise ValueError("Select a region within the audio.")
        sf.write(master, mixed[start:stop], dsp.RATE, subtype="FLOAT")
        if settings.polish:
            dsp.polish(master, finished)
            master = finished
        args = {"wav": ["-c:a", "pcm_s16le"], "flac": ["-c:a", "flac", "-sample_fmt", "s16"],
                "mp3": ["-c:a", "libmp3lame", "-q:a", 2]}[settings.format]
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
