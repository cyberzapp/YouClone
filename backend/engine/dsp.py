import json
import math
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

RATE = 44100
BREAK = re.compile(r'<break\s+time=["\'](\d+(?:\.\d+)?)(ms|s)["\']\s*/>|\[pause\s+(\d+(?:\.\d+)?)(ms|s)\]', re.I)


def ffmpeg(*args):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y", *map(str, args)],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip().splitlines()[-1])
    return result.stderr


def sanitize(source, dest, start, duration):
    ffmpeg("-ss", start, "-i", source, "-t", duration, "-vn", "-ac", 1, "-ar", RATE,
           "-af", "afftdn=nr=6:nf=-48", "-c:a", "pcm_s16le", dest)
    signal, _ = sf.read(dest, dtype="float32")
    active = np.flatnonzero(np.abs(signal) >= 0.01)
    if not active.size:
        raise ValueError("No speech above the −40 dBFS trim threshold.")
    pad = RATE // 30
    signal = signal[max(0, active[0] - pad):active[-1] + pad + 1]
    if len(signal) < RATE:
        raise ValueError("Select at least one second of speech.")
    sf.write(dest, signal, RATE, subtype="PCM_16")


def resample(signal, rate):
    factor = math.gcd(RATE, rate)
    return resample_poly(signal, RATE // factor, rate // factor) if rate != RATE else signal


def script_parts(text, comma=.25, period=.45):
    items = []
    parts = BREAK.split(text)
    for index in range(0, len(parts), 5):
        remaining = parts[index].strip()
        while remaining:
            limit = min(250, len(remaining))
            boundaries = list(re.finditer(r'[.!?।,;:](?=\s|$)', remaining[:limit + 1]))
            natural = [match.end() for match in boundaries if 150 <= match.end() <= limit]
            if limit < len(remaining):
                space = remaining.rfind(" ", 0, limit + 1)
                limit = natural[-1] if natural else (space if space > 0 else limit)
            segment, remaining = remaining[:limit].strip(), remaining[limit:].strip()
            if segment:
                pause = comma if segment[-1] in ",;:" else period if segment[-1] in ".!?।" else 0
                items.append((segment, pause))
        if index + 4 < len(parts):
            first, first_unit, second, second_unit = parts[index + 1:index + 5]
            unit = first_unit or second_unit
            pause = min(10, float(first or second) / (1000 if unit.lower() == "ms" else 1))
            if items:
                items[-1] = (items[-1][0], pause)
            else:
                items.append(("", pause))
    return items


def join(left, right, pause=0):
    if not right.size:
        return left
    if not left.size:
        return right
    if pause:
        edge = min(round(.004 * RATE), len(left), len(right))
        fade = np.linspace(0, 1, edge, dtype=np.float32)
        tail = left[-edge:] * fade[::-1]
        head = right[:edge] * fade
        return np.concatenate((left[:-edge], tail, np.zeros(round(pause * RATE), np.float32), head, right[edge:]))
    # Search for zero crossings close to each edge before overlapping the clips.
    edge = round(.004 * RATE)
    a = np.flatnonzero(np.diff(np.signbit(left[-edge:])))
    b = np.flatnonzero(np.diff(np.signbit(right[:edge])))
    end = len(left) - edge + int(a[-1]) + 1 if len(left) >= edge and a.size else len(left)
    start = int(b[0]) + 1 if b.size else 0
    left, right = left[:end], right[start:]
    overlap = min(round(.035 * RATE), len(left) // 2, len(right) // 2)
    if overlap < 2:
        return np.concatenate((left, right))
    angle = np.linspace(0, np.pi / 2, overlap, dtype=np.float32)
    return np.concatenate((left[:-overlap], left[-overlap:] * np.cos(angle)
                           + right[:overlap] * np.sin(angle), right[overlap:]))


def polish(source, dest):
    chain = ("highpass=f=80,treble=f=4500:g=1.5,"
             "acompressor=threshold=0.125:ratio=2.2:attack=12:release=100,"
             "deesser=i=0.2:m=0.35:f=0.55,")
    norm = "loudnorm=I=-14:TP=-1:LRA=11"
    log = ffmpeg("-i", source, "-af", chain + norm + ":print_format=json", "-f", "null", "-")
    data = json.loads(re.findall(r'\{[^{}]*"input_i"[^{}]*\}', log)[-1])
    values = [data[key] for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
    if all(math.isfinite(float(v)) for v in values):
        fields = ("measured_I", "measured_TP", "measured_LRA", "measured_thresh", "offset")
        norm += ":" + ":".join(f"{field}={value}" for field, value in zip(fields, values))
    ffmpeg("-i", source, "-af", chain + norm, "-ar", RATE, "-ac", 1, "-c:a", "pcm_s16le", dest)


def tempo(signal, speed):
    if speed == 1:
        return signal
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1",
         "-i", "pipe:0", "-af", f"atempo={speed}", "-f", "f32le", "pipe:1"],
        input=np.asarray(signal, dtype="<f4").tobytes(), capture_output=True, check=True, timeout=120,
    )
    return np.frombuffer(result.stdout, dtype="<f4")


def subtitles(result):
    def timecode(seconds):
        minutes, ms = divmod(max(0, round(seconds * 1000)), 60_000)
        hours, minutes = divmod(minutes, 60)
        sec, ms = divmod(ms, 1000)
        return f"{hours:02}:{minutes:02}:{sec:02},{ms:03}"

    words = [word for segment in result["segments"] for word in segment.get("words", [])]
    return "\n".join(
        f"{index}\n{timecode(word['start'])} --> {timecode(word['end'])}\n{word['word'].strip()}\n"
        for index, word in enumerate(words, 1)
    )
