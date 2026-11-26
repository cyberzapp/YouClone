import hashlib
import json
import math
import re
import struct
import subprocess
import wave

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal
from scipy.signal import resample_poly

RATE = 44100
BREAK = re.compile(r'<break\s+time=["\'](\d+(?:\.\d+)?)(ms|s)["\']\s*/>|\[pause(?:\s+(\d+(?:\.\d+)?)(ms|s))?\]', re.I)
INLINE_PHONETIC = re.compile(r"\[\[([^\[\]\n]+)\]\]")

# 16-bit provenance signature ("YC" = 0x59 0x43 = 0101_1001 0100_0011)
WATERMARK_BITS = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1]
WATERMARK_MAGIC = b"YCWMv1:"

EFFECT_PRESETS = {
    "broadcast": {
        "label": "Broadcast",
        "icon": "waveform",
        "description": "Radio/podcast standard — even volume, warm and clear.",
    },
    "podcast": {
        "label": "Podcast",
        "icon": "mic",
        "description": "Close-mic, intimate — gated, tight compression, no reverb.",
    },
    "cinematic": {
        "label": "Cinematic",
        "icon": "director",
        "description": "Film-quality — spacious room reflections and gentle dynamics.",
    },
    "warm": {
        "label": "Warm",
        "icon": "warm",
        "description": "Boosted low-mids, subtle saturation, cozy feel.",
    },
    "bright": {
        "label": "Bright",
        "icon": "sparkle",
        "description": "Crisp high-end, presence boost, airy feel.",
    },
    "raw": {
        "label": "Raw",
        "icon": "bypass",
        "description": "No mastering — model output as-is.",
    },
}

TAXONOMY = {
    "energy": ["calm", "relaxed", "steady", "energetic", "urgent", "frantic"],
    "emotion": ["happy", "sad", "angry", "surprised", "fearful", "neutral", "warm", "cold", "hopeful", "resigned"],
    "pace": ["slow", "measured", "conversational", "quick", "rushed"],
    "intimacy": ["whispered", "intimate", "conversational", "projected", "announcing"],
    "formality": ["casual", "neutral", "formal", "clinical"],
}

_KEYWORD_HINTS = {
    "urgent": ("energy", "urgent"),
    "urgency": ("energy", "urgent"),
    "frantic": ("energy", "frantic"),
    "energetic": ("energy", "energetic"),
    "upbeat": ("energy", "energetic"),
    "excited": ("energy", "energetic"),
    "calm": ("energy", "calm"),
    "relaxed": ("energy", "relaxed"),
    "steady": ("energy", "steady"),
    "rushed": ("pace", "rushed"),
    "quick": ("pace", "quick"),
    "fast": ("pace", "quick"),
    "slow": ("pace", "slow"),
    "measured": ("pace", "measured"),
    "surprised": ("emotion", "surprised"),
    "shocked": ("emotion", "surprised"),
    "angry": ("emotion", "angry"),
    "furious": ("emotion", "angry"),
    "sad": ("emotion", "sad"),
    "melancholy": ("emotion", "sad"),
    "happy": ("emotion", "happy"),
    "cheerful": ("emotion", "happy"),
    "fearful": ("emotion", "fearful"),
    "terrified": ("emotion", "fearful"),
    "scared": ("emotion", "fearful"),
    "warm": ("emotion", "warm"),
    "cold": ("emotion", "cold"),
    "hopeful": ("emotion", "hopeful"),
    "resigned": ("emotion", "resigned"),
    "whisper": ("intimacy", "whispered"),
    "whispered": ("intimacy", "whispered"),
    "whispering": ("intimacy", "whispered"),
    "intimate": ("intimacy", "intimate"),
    "close": ("intimacy", "intimate"),
    "projected": ("intimacy", "projected"),
    "announcing": ("intimacy", "announcing"),
    "announcer": ("intimacy", "announcing"),
    "casual": ("formality", "casual"),
    "conversational": ("pace", "conversational"),
    "formal": ("formality", "formal"),
    "clinical": ("formality", "clinical"),
}


def parse_direction(text: str) -> dict:
    """Parse natural-language acting direction into VoiceStudio's 5D taxonomy."""
    tokens: dict[str, list[str]] = {}
    lower = (text or "").lower()
    for kw, (dim, val) in _KEYWORD_HINTS.items():
        if re.search(rf"\b{re.escape(kw)}\b", lower):
            tokens.setdefault(dim, [])
            if val not in tokens[dim]:
                tokens[dim].append(val)
    for dim, allowed in TAXONOMY.items():
        for val in allowed:
            if re.search(rf"\b{re.escape(val)}\b", lower):
                tokens.setdefault(dim, [])
                if val not in tokens[dim]:
                    tokens[dim].append(val)
    order = ["emotion", "energy", "pace", "intimacy", "formality"]
    terms: list[str] = []
    for dim in order:
        for val in tokens.get(dim, []):
            if val not in terms:
                terms.append(val)
    instruct_prompt = ", ".join(terms) if terms else text.strip()
    energy = set(tokens.get("energy", []))
    pace = set(tokens.get("pace", []))
    emotion = set(tokens.get("emotion", []))
    if "urgent" in energy or "frantic" in energy or "rushed" in pace or "quick" in pace:
        rate_bias = 1.12
    elif "calm" in energy or "relaxed" in energy or "slow" in pace or "measured" in pace:
        rate_bias = 0.92
    else:
        rate_bias = 1.0
    stability = 35 if ("surprised" in emotion or "angry" in emotion or "fearful" in emotion or "frantic" in energy) else (
        70 if ("calm" in energy or "steady" in energy or "clinical" in tokens.get("formality", [])) else 50
    )
    style_exaggeration = 45 if ("energetic" in energy or "urgent" in energy or "announcing" in tokens.get("intimacy", [])) else 15
    return {
        "tokens": tokens,
        "instruct_prompt": instruct_prompt,
        "translate_hint": f"Deliver this with a {instruct_prompt} tone." if instruct_prompt else "",
        "rate_bias": rate_bias,
        "recommended_stability": stability,
        "recommended_style": style_exaggeration,
        "taxonomy": TAXONOMY,
    }


def rate_fit_analysis(text: str, slot_seconds: float, language: str = "en") -> dict:
    """Analyze whether a script line fits into a target dubbing time slot."""
    clean = re.sub(r"\[[^\]]*\]|<[^>]*>", "", text or "").strip()
    words = [w for w in re.split(r"\s+", clean) if w]
    chars = len(re.sub(r"\s+", "", clean))
    # Approximate natural speaking duration across languages (~14 chars/s or ~2.6 words/s)
    cjk = sum(1 for ch in clean if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff" or "\uac00" <= ch <= "\ud7af")
    if cjk > len(clean) * 0.3:
        estimated_seconds = max(0.3, cjk / 4.2)
    else:
        estimated_seconds = max(0.3, max(len(words) / 2.6, chars / 14.5))
    ratio = estimated_seconds / max(0.1, slot_seconds)
    clamped_speed = min(1.3, max(0.7, round(ratio, 2)))
    if 0.9 <= ratio <= 1.1:
        verdict = "optimal"
        suggestion = "Fits naturally within the target slot without speed distortion."
    elif ratio < 0.9:
        verdict = "short"
        suggestion = f"Line is shorter than the {slot_seconds:.1f}s slot. Add a [pause {(slot_seconds - estimated_seconds):.1f}s] or slow speed to {clamped_speed:.2f}×."
    elif ratio <= 1.35:
        verdict = "tight"
        suggestion = f"Line runs ~{estimated_seconds:.1f}s for a {slot_seconds:.1f}s slot. Set Speed to {clamped_speed:.2f}× or trim {max(1, round((estimated_seconds - slot_seconds) * 2.5))} words."
    else:
        target_words = max(1, int(slot_seconds * 2.6))
        trimmed = " ".join(words[:target_words]) + ("…" if len(words) > target_words else "")
        verdict = "overflow"
        suggestion = f"Line needs ~{estimated_seconds:.1f}s ({ratio:.1f}× slot). Shorten to ~{target_words} words: \"{trimmed}\""
    return {
        "word_count": len(words),
        "char_count": chars,
        "slot_seconds": round(slot_seconds, 2),
        "estimated_seconds": round(estimated_seconds, 2),
        "required_speed": round(ratio, 2),
        "recommended_speed": clamped_speed,
        "verdict": verdict,
        "suggestion": suggestion,
    }


def ffmpeg(*args):
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y", *map(str, args)],
        capture_output=True, text=True, timeout=3600,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip().splitlines()[-1])
    return result.stderr


def trim_passage_to_pauses_np(
    signal: np.ndarray,
    sample_rate: int = RATE,
    *,
    pad_lead_s: float = 0.0,
    pad_trail_s: float = 0.0,
) -> np.ndarray:
    """Trim leading and trailing silence below -38 dBFS with 30ms margin, 20ms cosine fades, and optional padding."""
    if signal.size == 0:
        return signal
    floor = 10.0 ** (-38.0 / 20.0)
    voiced = np.flatnonzero(np.abs(signal) >= floor)
    if not voiced.size:
        return signal
    margin = int(sample_rate * 0.030)
    start = max(0, int(voiced[0]) - margin)
    end = min(len(signal), int(voiced[-1]) + 1 + margin)
    trimmed = signal[start:end].astype(np.float32, copy=True)
    fade_n = min(int(sample_rate * 0.020), len(trimmed) // 4)
    if fade_n >= 2:
        t = np.linspace(0.0, math.pi, fade_n, dtype=np.float32)
        fade_in = 0.5 * (1.0 - np.cos(t))
        fade_out = 0.5 * (1.0 + np.cos(t))
        trimmed[:fade_n] *= fade_in
        trimmed[-fade_n:] *= fade_out
    lead_n = max(0, int(round(sample_rate * pad_lead_s)))
    trail_n = max(0, int(round(sample_rate * pad_trail_s)))
    if lead_n or trail_n:
        trimmed = np.concatenate((
            np.zeros(lead_n, dtype=np.float32),
            trimmed,
            np.zeros(trail_n, dtype=np.float32),
        ))
    return trimmed


def trim_detached_tail_np(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """Trim detached leading/trailing micro-artifacts separated by >=200ms silence (preserving stop releases like -t/-p/-k)."""
    if audio.size < int(0.8 * sample_rate):
        return audio
    win = max(1, int(0.015 * sample_rate))
    n = len(audio) // win
    if n < 4:
        return audio
    frames = audio[:n * win].reshape(n, win)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak_rms = float(np.max(rms))
    if peak_rms < 1e-4:
        return audio
    thresh = max(10.0 ** (-37.0 / 20.0), peak_rms * 0.06)
    voiced = rms >= thresh
    if not np.any(voiced):
        return audio

    islands: list[tuple[int, int]] = []
    in_island = False
    start_i = 0
    for idx, flag in enumerate(voiced):
        if flag and not in_island:
            in_island = True
            start_i = idx
        elif not flag and in_island:
            in_island = False
            islands.append((start_i, idx))
    if in_island:
        islands.append((start_i, n))

    frame_dur = win / float(sample_rate)
    while len(islands) >= 2:
        _prev_start, prev_end = islands[-2]
        last_start, last_end = islands[-1]
        gap_s = (last_start - prev_end) * frame_dur
        last_dur_s = (last_end - last_start) * frame_dur
        last_rms = float(np.max(rms[last_start:last_end]))
        if gap_s >= 0.20 and (last_dur_s <= 0.28 or last_rms < peak_rms * 0.35):
            islands.pop()
        else:
            break

    while len(islands) >= 2:
        first_start, first_end = islands[0]
        next_start, _next_end = islands[1]
        gap_s = (next_start - first_end) * frame_dur
        first_dur_s = (first_end - first_start) * frame_dur
        first_rms = float(np.max(rms[first_start:first_end]))
        if gap_s >= 0.18 and (first_dur_s <= 0.24 or first_rms < peak_rms * 0.30):
            islands.pop(0)
        else:
            break

    start_sample = max(0, int((islands[0][0] - 2) * win))
    cut_sample = min(len(audio), int((islands[-1][1] + 2) * win))
    return audio[start_sample:cut_sample]


def sanitize(source, dest, start, duration):
    ffmpeg("-ss", start, "-i", source, "-t", duration, "-vn", "-ac", 1, "-ar", RATE,
           "-af", "afftdn=nr=6:nf=-48", "-c:a", "pcm_s16le", dest)
    signal, _ = sf.read(dest, dtype="float32")
    floor = 10.0 ** (-38.0 / 20.0)
    active = np.flatnonzero(np.abs(signal) >= floor)
    if not active.size:
        raise ValueError("No speech above the −38 dBFS trim threshold.")
    signal = trim_passage_to_pauses_np(signal, RATE, pad_lead_s=0.08, pad_trail_s=0.22)
    if len(signal) < RATE:
        raise ValueError("Select at least one second of speech.")
    sf.write(dest, signal, RATE, subtype="PCM_16")


def isolate_vocals(source, dest):
    """VoiceStudio / ElevenLabs Voice Isolator: spectral FFT denoising + voice bandpass + speech leveling."""
    ffmpeg(
        "-i", source, "-vn", "-ac", 1, "-ar", RATE,
        "-af", "highpass=f=85,lowpass=f=11500,afftdn=nr=18:nf=-42:tn=1,adeclick",
        "-c:a", "pcm_s16le", dest,
    )
    signal, _ = sf.read(dest, dtype="float32")
    if not signal.size:
        raise ValueError("Audio file contains no usable samples.")
    leveled = level_speech_np(signal, RATE, target_db=-18.0, amount=0.75)
    limited = peak_limit_np(leveled, RATE, threshold_db=-1.0)
    sf.write(dest, limited, RATE, subtype="PCM_16")
    return sf.info(dest).duration


def resample(signal, rate):
    factor = math.gcd(RATE, rate)
    return resample_poly(signal, RATE // factor, rate // factor) if rate != RATE else signal


_KEEP_UPPER = {
    "GTA", "FBI", "CIA", "USA", "NPC", "DLC", "RPG", "FPS", "HUD", "VIP",
    "CEO", "USB", "GPU", "CPU", "RAM", "SSD", "API", "TTS", "AI", "UI",
    "UX", "UK", "US", "EU", "TV", "PC", "PS", "II", "III", "IV", "VI",
    "VII", "VIII",
}
_ABBR_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|vs|Vs|etc|approx|fig|def)|e\.g|i\.e)\.$")


def normalize_script_text(text: str) -> str:
    """Normalize Markdown, paragraph breaks, prosodic punctuation, emphasis caps, and quotes before TTS chunking."""
    # Strip Markdown ATX headings (# Title), bold/italic markers (**, __, *), and inline backticks
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    cleaned = re.sub(r"\*\*|__|``|`", "", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\1", cleaned)
    # Preserve paragraph breaks while cleaning surrounding horizontal whitespace
    cleaned = re.sub(r"[ \t]*\r?\n+[ \t]*", "\n", cleaned)
    # Convert parenthetical em-dashes and double-hyphens to natural comma pauses
    cleaned = re.sub(r"[ \t]*(?:—|--)[ \t]*", ", ", cleaned)
    # Convert mid-sentence ellipses to rising comma pauses; sentence-final ellipses to clean periods
    cleaned = re.sub(r"[ \t]*(?:\.{2,}|…)[ \t]*(?=[a-z])", ", ", cleaned)
    cleaned = re.sub(r"[ \t]*(?:\.{2,}|…)[ \t]*", ". ", cleaned)
    # Strip decorative quotation marks around short 1-4 word phrases for cleaner tokenization
    cleaned = re.sub(r'["“”]([^"“”\n]{1,32})["“”]', r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    # Convert ALL-CAPS English emphasis words of 3+ letters (LISTEN -> listen, LOT -> lot)
    def _lower_emphasis(match: re.Match) -> str:
        word = match.group(0)
        if word in _KEEP_UPPER:
            return word
        return word.lower()

    cleaned = re.sub(r"\b[A-Z]{3,}\b", _lower_emphasis, cleaned)
    # Neutralize exclamation marks so non-autoregressive TTS doesn't emit trailing gasps/squeaks
    cleaned = re.sub(r"[!?]*\?[!?]*", "?", cleaned)
    cleaned = re.sub(r"!+", ".", cleaned)
    cleaned = re.sub(r"\.{2,}", ".", cleaned)
    # Resolve inline [[phonetic]] overrides by keeping their inner text
    return INLINE_PHONETIC.sub(r"\1", cleaned)


def _split_sentences(block: str) -> list[str]:
    """Split a paragraph into individual sentences and dramatic colon setups."""
    sentences: list[str] = []
    start = 0
    for match in re.finditer(r"(?:[.!?।]+|:)(?=\s|$)", block):
        end = match.end()
        candidate = block[start:end].strip()
        if candidate.endswith(".") and _ABBR_END.search(candidate):
            continue
        if candidate:
            sentences.append(candidate)
        start = end
    tail = block[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _split_long_sentence(sentence: str, max_chars: int = 200) -> list[str]:
    """Split an oversized (>220 char) single sentence at clause boundaries [,;:] near midpoint."""
    pieces: list[str] = []
    remaining = sentence.strip()
    while len(remaining) > 220:
        limit = min(max_chars, len(remaining))
        clauses = [m.end() for m in re.finditer(r"[,;:](?=\s|$)", remaining[:limit + 1])]
        mid_clauses = [pos for pos in clauses if pos >= 90]
        if mid_clauses:
            cut = mid_clauses[-1]
        elif clauses:
            cut = clauses[-1]
        else:
            space = remaining.rfind(" ", 0, limit + 1)
            cut = space if space > 0 else limit
        head, remaining = remaining[:cut].strip(), remaining[cut:].strip()
        if head:
            pieces.append(head)
    if remaining:
        pieces.append(remaining)
    return pieces


def script_parts(text, comma=.22, period=.36, *, autoregressive: bool = False):
    if autoregressive:
        # Preserve natural dramatic punctuation (—, ..., :, !, ?, quotes) and group sentences for rich prosody.
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
        cleaned = re.sub(r"\*\*|__|``|`", "", cleaned)
        cleaned = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]*\r?\n+[ \t]*", "\n", cleaned)
        cleaned = INLINE_PHONETIC.sub(r"\1", cleaned)
        items = []
        parts = BREAK.split(cleaned)
        max_ar_chars = 420
        for index in range(0, len(parts), 5):
            raw_block = parts[index].strip()
            if raw_block:
                paragraphs = [p.strip() for p in re.split(r"\n+", raw_block) if p.strip()]
                for p_idx, para in enumerate(paragraphs):
                    sents: list[str] = []
                    start = 0
                    for m in re.finditer(r"[.!?।]+(?:\s*[\"”'’])?(?=\s|$)", para):
                        end = m.end()
                        cand = para[start:end].strip()
                        if cand.endswith(".") and _ABBR_END.search(cand):
                            continue
                        if cand:
                            sents.append(cand)
                        start = end
                    tail = para[start:].strip()
                    if tail:
                        sents.append(tail)
                    current = ""
                    for s_idx, sent in enumerate(sents):
                        if not current:
                            current = sent
                        elif len(current) + 1 + len(sent) <= max_ar_chars:
                            current = f"{current} {sent}"
                        else:
                            items.append((current, period))
                            current = sent
                    if current:
                        is_last_para = p_idx == len(paragraphs) - 1
                        p_pause = period if is_last_para else max(period * 2.0, 0.72)
                        items.append((current, p_pause))
            if index + 4 < len(parts):
                first, first_unit, second, second_unit = parts[index + 1:index + 5]
                raw_val = first or second
                unit = (first_unit or second_unit or "s").lower()
                pause = min(10.0, float(raw_val) / (1000.0 if unit == "ms" else 1.0)) if raw_val else 0.35
                if items:
                    items[-1] = (items[-1][0], pause)
                else:
                    items.append(("", pause))
        return items

    text = normalize_script_text(text)
    items = []
    parts = BREAK.split(text)
    for index in range(0, len(parts), 5):
        raw_block = parts[index].strip()
        if raw_block:
            paragraphs = [p.strip() for p in re.split(r"\n+", raw_block) if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                units: list[str] = []
                for sent in _split_sentences(para):
                    if len(sent) > 220:
                        units.extend(_split_long_sentence(sent))
                    else:
                        units.append(sent)
                for u_idx, unit in enumerate(units):
                    is_last_in_para = (u_idx == len(units) - 1) and (p_idx < len(paragraphs) - 1)
                    if unit.endswith(":"):
                        clean_unit = unit[:-1].strip() + "."
                        pause = max(comma * 1.5, 0.36)
                    elif unit.endswith(("?", "!", "।")):
                        clean_unit = unit
                        pause = max(period * 1.18, 0.42) if unit.endswith("?") else period
                    elif unit[-1] in ",;":
                        clean_unit = unit
                        pause = comma
                    else:
                        clean_unit = unit
                        pause = max(period * 1.18, 0.42) if len(clean_unit.split()) <= 3 else period
                    if is_last_in_para:
                        pause = max(pause, period * 2.2, 0.82)
                    if clean_unit and clean_unit[0].islower():
                        clean_unit = clean_unit[0].upper() + clean_unit[1:]
                    items.append((clean_unit, pause))
        if index + 4 < len(parts):
            first, first_unit, second, second_unit = parts[index + 1:index + 5]
            raw_val = first or second
            unit = (first_unit or second_unit or "s").lower()
            pause = min(10.0, float(raw_val) / (1000.0 if unit == "ms" else 1.0)) if raw_val else 0.35
            if items:
                items[-1] = (items[-1][0], pause)
            else:
                items.append(("", pause))
    return items


def trim_speech_padding_np(signal: np.ndarray, sample_rate: int = RATE) -> np.ndarray:
    """Trim leading/trailing silence below -38 dBFS with 25ms context margin and 20ms cosine fade-in/out."""
    return trim_passage_to_pauses_np(signal, sample_rate)


def _room_tone_profile_np(signal: np.ndarray):
    """Extract magnitude spectrum and RMS level of the quietest 15% active frames for room-tone pause filling."""
    frame_len = 1024
    n = len(signal) // frame_len
    if n < 4:
        return None
    frames = signal[:n * frame_len].reshape(n, frame_len).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    live = rms > 10.0 ** (-90.0 / 20.0)
    if int(live.sum()) < 4:
        return None
    quiet = frames[live & (rms <= np.percentile(rms[live], 15))]
    if quiet.shape[0] < 2:
        return None
    level = float(np.sqrt((quiet ** 2).mean()))
    if level <= 0 or level > 10.0 ** (-30.0 / 20.0):
        return None
    window = np.hanning(frame_len)
    magnitude = np.abs(np.fft.rfft(quiet * window, axis=1)).mean(axis=0)
    return magnitude, level


def _generate_room_tone(profile, n_samples: int, seed: int = 3611) -> np.ndarray:
    if n_samples <= 0:
        return np.empty(0, dtype=np.float32)
    if profile is None:
        return np.zeros(n_samples, dtype=np.float32)
    magnitude, level = profile
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n_samples))
    shape = np.interp(
        np.linspace(0.0, 1.0, spectrum.shape[0]),
        np.linspace(0.0, 1.0, magnitude.shape[0]),
        magnitude,
    )
    noise = np.fft.irfft(spectrum * shape, n_samples)
    rms = float(np.sqrt((noise ** 2).mean())) if n_samples else 0.0
    scaled = noise * (level / rms) if rms > 0 else noise
    return scaled.astype(np.float32)


def join(left, right, pause=0, pause_fill="silence", seed=3611):
    if not right.size:
        return left
    if not left.size:
        return right
    if pause:
        edge = min(round(.004 * RATE), len(left), len(right))
        fade = np.linspace(0, 1, edge, dtype=np.float32)
        tail = left[-edge:] * fade[::-1]
        head = right[:edge] * fade
        gap_samples = round(pause * RATE)
        if pause_fill == "room_tone" and gap_samples > 0:
            profile = _room_tone_profile_np(np.concatenate((left, right)))
            gap = _generate_room_tone(profile, gap_samples, seed=seed)
        else:
            gap = np.zeros(gap_samples, np.float32)
        return np.concatenate((left[:-edge], tail, gap, head, right[edge:]))
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


def _box_mean(values: np.ndarray, width: int) -> np.ndarray:
    n = values.shape[0]
    csum = np.concatenate(([0.0], np.cumsum(values)))
    lo = np.clip(np.arange(n) - width // 2, 0, n)
    hi = np.clip(np.arange(n) + width - width // 2, 0, n)
    return (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)


def level_speech_np(
    signal: np.ndarray,
    sample_rate: int = RATE,
    *,
    target_db: float | None = -19.0,
    window_ms: float = 400.0,
    amount: float = 0.7,
    max_gain_db: float = 8.0,
    smooth_ms: float = 200.0,
) -> np.ndarray:
    """VoiceStudio EBU-style gated slow phrase leveler in pure NumPy."""
    if signal.size == 0:
        return signal
    hop = max(1, int(sample_rate * 0.01))
    n = len(signal)
    frames = -(-n // hop)
    padded = np.zeros(frames * hop, dtype=np.float64)
    padded[:n] = signal.astype(np.float64) ** 2
    frame_power = padded.reshape(frames, hop).mean(axis=1)

    short_power = _box_mean(frame_power, 5)
    short_db = 10.0 * np.log10(short_power + 1e-20)
    if not short_db.max() > -50.0:
        return signal
    floor_db = max(-70.0, float(short_db.max()) - 40.0)
    average_db = 10.0 * np.log10(float(short_power[short_db > floor_db].mean()) + 1e-20)
    speech = short_db > max(floor_db, average_db - 25.0)
    if not np.any(speech):
        return signal
    speech_db = 10.0 * np.log10(float(frame_power[speech].mean()) + 1e-20)

    win = max(1, round(window_ms / 10.0))
    speech_power = _box_mean(np.where(speech, frame_power, 0.0), win)
    speech_share = _box_mean(speech.astype(np.float64), win)
    level_db = 10.0 * np.log10(speech_power / np.maximum(speech_share, 1e-12) + 1e-20)

    gain_db = np.clip(amount * (speech_db - level_db), -max_gain_db, max_gain_db)
    if target_db is not None:
        gain_db += target_db - speech_db

    idx = np.where(speech, np.arange(frames), -1)
    idx = np.maximum.accumulate(idx)
    idx[idx < 0] = int(np.argmax(speech))
    gain_db = gain_db[idx]

    alpha = math.exp(-1.0 / max(1.0, smooth_ms / 10.0))
    for order in (slice(None), slice(None, None, -1)):
        g = gain_db[order].copy()
        for i in range(1, frames):
            g[i] = alpha * g[i - 1] + (1.0 - alpha) * g[i]
        gain_db[order] = g

    centers = np.arange(frames) * hop + hop / 2.0
    sample_gain = 10.0 ** (np.interp(np.arange(n), centers, gain_db) / 20.0)
    return (signal.astype(np.float64) * sample_gain).astype(np.float32)


def peak_limit_np(
    signal: np.ndarray,
    sample_rate: int = RATE,
    *,
    threshold_db: float = -1.0,
) -> np.ndarray:
    """Pure NumPy soft-knee lookahead ceiling limiter."""
    if signal.size == 0:
        return signal
    ceiling = 10.0 ** (threshold_db / 20.0)
    peak = float(np.max(np.abs(signal)))
    if peak <= ceiling:
        return signal
    win = max(1, int(sample_rate * 0.005))
    env = np.maximum(np.abs(signal), 1e-12)
    kernel = np.ones(win, dtype=np.float32) / win
    required_gain = np.minimum(1.0, ceiling / env)
    smoothed_gain = np.minimum(required_gain, np.convolve(required_gain, kernel, mode="same"))
    return (signal * smoothed_gain).astype(np.float32)


def synthesize_ambience(mode: str, n_samples: int, seed: int = 42) -> np.ndarray:
    """Generate procedural ambient beds (vinyl, rain, tape, cafe) for ElevenLabs-style Audio Effects."""
    if n_samples <= 0 or mode == "none" or not mode:
        return np.zeros(max(0, n_samples), dtype=np.float32)
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n_samples).astype(np.float32)
    if mode == "tape":
        b, a = sp_signal.butter(2, [200 / (RATE / 2), 6000 / (RATE / 2)], btype="band")
        bed = sp_signal.lfilter(b, a, white).astype(np.float32) * 0.012
    elif mode == "vinyl":
        b, a = sp_signal.butter(2, [400 / (RATE / 2), 5000 / (RATE / 2)], btype="band")
        hiss = sp_signal.lfilter(b, a, white).astype(np.float32) * 0.008
        crackle_mask = rng.random(n_samples) < 0.0008
        crackles = np.where(crackle_mask, rng.uniform(-0.08, 0.08, n_samples), 0.0).astype(np.float32)
        bed = hiss + crackles
    elif mode == "rain":
        b, a = sp_signal.butter(2, [300 / (RATE / 2), 4200 / (RATE / 2)], btype="band")
        rain = sp_signal.lfilter(b, a, white).astype(np.float32)
        mod = 0.8 + 0.2 * np.sin(np.linspace(0, n_samples / RATE * 1.7, n_samples, dtype=np.float32))
        bed = rain * mod * 0.022
    elif mode == "cafe":
        b, a = sp_signal.butter(3, [180 / (RATE / 2), 1800 / (RATE / 2)], btype="band")
        murmur = sp_signal.lfilter(b, a, white).astype(np.float32)
        t = np.linspace(0, n_samples / RATE, n_samples, dtype=np.float32)
        envelope = 0.65 + 0.35 * np.sin(2 * math.pi * 0.9 * t) * np.cos(2 * math.pi * 0.37 * t)
        bed = murmur * envelope * 0.025
    else:
        bed = np.zeros(n_samples, dtype=np.float32)
    return bed


def generate_sfx_audio(prompt: str, duration: float = 5.0, seed: int = 42) -> np.ndarray:
    """Synthesize procedural sound effects & soundscapes from prompt keywords and acoustic synthesis."""
    n = max(RATE, round(min(30.0, max(1.0, duration)) * RATE))
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, n / RATE, n, endpoint=False, dtype=np.float32)
    lower = (prompt or "").lower()
    out = np.zeros(n, dtype=np.float32)

    if any(k in lower for k in ("rain", "storm", "downpour", "water", "shower")):
        white = rng.standard_normal(n).astype(np.float32)
        b, a = sp_signal.butter(2, [250 / (RATE / 2), 5500 / (RATE / 2)], btype="band")
        rain = sp_signal.lfilter(b, a, white).astype(np.float32)
        droplets = np.where(rng.random(n) < 0.0015, rng.uniform(-0.35, 0.35, n), 0.0).astype(np.float32)
        b2, a2 = sp_signal.butter(2, [1200 / (RATE / 2), 4500 / (RATE / 2)], btype="band")
        out += rain * 0.22 + sp_signal.lfilter(b2, a2, droplets).astype(np.float32)

    if any(k in lower for k in ("thunder", "rumble", "explosion", "boom", "impact")):
        white = rng.standard_normal(n).astype(np.float32)
        b, a = sp_signal.butter(3, 140 / (RATE / 2), btype="low")
        rumble = sp_signal.lfilter(b, a, white).astype(np.float32)
        env = np.exp(-t / max(0.6, duration * 0.35)) * (0.6 + 0.4 * np.sin(2 * math.pi * 1.8 * t) ** 2)
        out += rumble * env * 2.5

    if any(k in lower for k in ("wind", "breeze", "gale", "howl", "air", "whoosh")):
        white = rng.standard_normal(n).astype(np.float32)
        b, a = sp_signal.butter(2, [150 / (RATE / 2), 900 / (RATE / 2)], btype="band")
        wind = sp_signal.lfilter(b, a, white).astype(np.float32)
        sweep = 0.5 + 0.5 * np.sin(2 * math.pi * (0.25 if "whoosh" not in lower else 0.8) * t - math.pi / 2)
        out += wind * sweep * 0.65

    if any(k in lower for k in ("riser", "cinematic", "suspense", "swell", "transition")):
        f0, f1 = 55.0, 440.0
        phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) * (t ** 2) / max(0.1, duration))
        synth = 0.45 * np.sin(phase) + 0.25 * np.sin(2.01 * phase) + 0.15 * np.sin(0.5 * phase)
        env = np.linspace(0.02, 1.0, n, dtype=np.float32) ** 2
        out += synth * env

    if any(k in lower for k in ("drone", "scifi", "sci-fi", "space", "ambient", "hum", "cyber")):
        drone = (
            0.35 * np.sin(2 * math.pi * 65.4 * t)
            + 0.25 * np.sin(2 * math.pi * 98.0 * t + 0.4 * np.sin(2 * math.pi * 0.3 * t))
            + 0.18 * np.sin(2 * math.pi * 130.8 * t + 0.7 * np.sin(2 * math.pi * 0.17 * t))
        )
        out += drone.astype(np.float32) * 0.7

    if any(k in lower for k in ("heart", "heartbeat", "pulse")):
        bpm = 72.0
        period = 60.0 / bpm
        phase_t = np.mod(t, period)
        lub = np.exp(-((phase_t - 0.12) ** 2) / 0.0012) * np.sin(2 * math.pi * 52.0 * phase_t)
        dub = 0.75 * np.exp(-((phase_t - 0.30) ** 2) / 0.0010) * np.sin(2 * math.pi * 64.0 * phase_t)
        out += (lub + dub).astype(np.float32) * 0.85

    if any(k in lower for k in ("type", "typing", "keyboard", "click", "steps", "footsteps")):
        rate_hz = 6.5 if "type" in lower or "key" in lower else 2.0
        n_hits = max(2, int(duration * rate_hz))
        positions = np.sort(rng.integers(0, max(1, n - int(0.04 * RATE)), size=n_hits))
        for pos in positions:
            length = min(int(0.03 * RATE), n - int(pos))
            burst = rng.standard_normal(length).astype(np.float32) * np.exp(-np.linspace(0, 12, length, dtype=np.float32))
            out[int(pos):int(pos) + length] += burst * 0.55

    if any(k in lower for k in ("vinyl", "crackle", "fire", "campfire")):
        out += synthesize_ambience("vinyl", n, seed=seed) * 6.0

    if any(k in lower for k in ("cafe", "crowd", "murmur", "room", "office")):
        out += synthesize_ambience("cafe", n, seed=seed) * 7.0

    if any(k in lower for k in ("radio", "static", "glitch", "interference", "white noise")):
        white = rng.standard_normal(n).astype(np.float32)
        b, a = sp_signal.butter(2, [400 / (RATE / 2), 3800 / (RATE / 2)], btype="band")
        static = sp_signal.lfilter(b, a, white).astype(np.float32)
        tone = 0.15 * np.sin(2 * math.pi * (800 + 300 * np.sin(2 * math.pi * 3.5 * t)) * t)
        out += (static * 0.3 + tone).astype(np.float32)

    if float(np.max(np.abs(out))) < 1e-5:
        # Fallback rich procedural texture keyed by prompt hash
        h = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
        base_f = 90.0 + (h % 220)
        white = rng.standard_normal(n).astype(np.float32)
        b, a = sp_signal.butter(2, [max(60.0, base_f * 0.5) / (RATE / 2), min(6000.0, base_f * 6.0) / (RATE / 2)], btype="band")
        textured = sp_signal.lfilter(b, a, white).astype(np.float32) * 0.25
        harmonic = 0.3 * np.sin(2 * math.pi * base_f * t + 0.5 * np.sin(2 * math.pi * 1.5 * t))
        out = (textured + harmonic).astype(np.float32)

    fade = min(int(0.06 * RATE), n // 4)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        out[:fade] *= ramp
        out[-fade:] *= ramp[::-1]
    return peak_limit_np(level_speech_np(out, RATE, target_db=-16.0), RATE, threshold_db=-1.0)


def build_ffmpeg_fx_chain(
    *,
    polish_enabled: bool = True,
    effect_preset: str = "broadcast",
    speaker_boost: bool = True,
    filter_mode: str = "none",
    space_mode: str = "none",
    distance_mode: str = "near",
) -> str:
    """Build combined VoiceStudio + ElevenLabs FFmpeg audio filter chain."""
    stages: list[str] = []

    # 1. VoiceStudio Effect Preset
    preset = (effect_preset or ("broadcast" if polish_enabled else "raw")).lower()
    if preset == "broadcast":
        stages.extend([
            "highpass=f=65",
            "acompressor=threshold=0.22:ratio=1.35:attack=25:release=160",
            "deesser=i=0.15:m=0.30:f=0.55",
        ])
    elif preset == "podcast":
        stages.extend([
            "highpass=f=100",
            "agate=threshold=0.01:release=200",
            "acompressor=threshold=0.1:ratio=3.8:attack=3:release=65",
            "equalizer=f=180:t=q:w=1.0:g=-1.0",
            "equalizer=f=1600:t=q:w=1.0:g=2.0",
            "treble=f=4200:g=1.5",
            "deesser=i=0.25:m=0.35:f=0.55",
        ])
    elif preset == "cinematic":
        stages.extend([
            "highpass=f=60",
            "acompressor=threshold=0.18:ratio=1.8:attack=10:release=150",
            "aecho=0.85:0.82:38|72:0.16|0.10",
        ])
    elif preset == "warm":
        stages.extend([
            "highpass=f=60",
            "bass=f=180:g=3.0",
            "equalizer=f=1200:t=q:w=1.0:g=1.0",
            "treble=f=4500:g=-1.0",
            "acompressor=threshold=0.16:ratio=2.0:attack=8:release=120",
            "aecho=0.92:0.88:28:0.08",
        ])
    elif preset == "bright":
        stages.extend([
            "highpass=f=80",
            "bass=f=180:g=-1.0",
            "treble=f=4000:g=4.0",
            "acompressor=threshold=0.2:ratio=2.5:attack=4:release=80",
        ])

    # 2. ElevenLabs Speaker Boost (subtle formant-preserving presence clarity)
    if speaker_boost and preset != "raw":
        stages.append("equalizer=f=2500:t=q:w=1.4:g=0.4")

    # 3. ElevenLabs Filter Mode
    f_mode = (filter_mode or "none").lower()
    if f_mode == "phone":
        stages.extend(["highpass=f=320", "lowpass=f=3400", "acompressor=threshold=0.08:ratio=4.5:attack=2:release=50"])
    elif f_mode == "radio":
        stages.extend(["highpass=f=450", "lowpass=f=2800", "equalizer=f=1200:t=q:w=2.0:g=5.0", "acompressor=threshold=0.06:ratio=6:attack=1:release=40"])
    elif f_mode == "robot":
        stages.extend(["aecho=0.8:0.88:11|23:0.45|0.35", "flanger=delay=4:depth=3:speed=2.5"])
    elif f_mode == "megaphone":
        stages.extend(["highpass=f=550", "lowpass=f=3800", "equalizer=f=1800:t=q:w=1.5:g=6.0"])

    # 4. ElevenLabs Space / Reverb Mode
    s_mode = (space_mode or "none").lower()
    if s_mode == "room":
        stages.append("aecho=0.88:0.85:24|48:0.14|0.08")
    elif s_mode == "hall":
        stages.append("aecho=0.84:0.82:55|110|165:0.22|0.15|0.09")
    elif s_mode == "cathedral":
        stages.append("aecho=0.80:0.78:85|170|260|350:0.28|0.22|0.16|0.10")

    # 5. ElevenLabs Mic Distance Mode
    d_mode = (distance_mode or "near").lower()
    if d_mode == "medium":
        stages.extend(["lowpass=f=9500", "aecho=0.9:0.85:18:0.08"])
    elif d_mode == "far":
        stages.extend(["highpass=f=150", "lowpass=f=5800", "aecho=0.85:0.80:35|75:0.22|0.14"])

    return ",".join(stages)


def polish(
    source,
    dest,
    *,
    effect_preset: str = "broadcast",
    speaker_boost: bool = True,
    filter_mode: str = "none",
    space_mode: str = "none",
    distance_mode: str = "near",
    ambience_mode: str = "none",
    seed: int = 42,
):
    signal, _ = sf.read(source, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    preset = (effect_preset or "broadcast").lower()
    if preset in ("broadcast", "podcast", "warm", "bright"):
        signal = level_speech_np(signal, RATE, target_db=-16.5, amount=0.28)
    if ambience_mode and ambience_mode != "none":
        signal = signal + synthesize_ambience(ambience_mode, len(signal), seed=seed)
    signal = peak_limit_np(signal, RATE, threshold_db=-0.5)
    sf.write(source, signal, RATE, subtype="FLOAT")

    chain = build_ffmpeg_fx_chain(
        polish_enabled=preset != "raw",
        effect_preset=preset,
        speaker_boost=speaker_boost,
        filter_mode=filter_mode,
        space_mode=space_mode,
        distance_mode=distance_mode,
    )
    if not chain:
        sf.write(dest, signal, RATE, subtype="PCM_16")
        return
    norm = "loudnorm=I=-12.0:TP=-0.3:LRA=14"
    full_chain = f"{chain},{norm}"
    log = ffmpeg("-i", source, "-af", full_chain + ":print_format=json", "-f", "null", "-")
    matches = re.findall(r'\{[^{}]*"input_i"[^{}]*\}', log)
    if matches:
        data = json.loads(matches[-1])
        values = [data[key] for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
        if all(math.isfinite(float(v)) for v in values):
            fields = ("measured_I", "measured_TP", "measured_LRA", "measured_thresh", "offset")
            norm += ":" + ":".join(f"{field}={value}" for field, value in zip(fields, values))
    ffmpeg("-i", source, "-af", f"{chain},{norm}", "-ar", RATE, "-ac", 1, "-c:a", "pcm_s16le", dest)


def embed_watermark_np(signal: np.ndarray, sample_rate: int = RATE) -> np.ndarray:
    """Embed imperceptible 16-bit spread-spectrum ultrasonic carrier (18.2–19.6 kHz) for provenance verification."""
    if signal.size < sample_rate // 2:
        return signal
    n = len(signal)
    t = np.arange(n, dtype=np.float64) / sample_rate
    carrier = np.zeros(n, dtype=np.float64)
    for idx, bit in enumerate(WATERMARK_BITS):
        freq = 18200.0 + idx * 85.0
        amp = 1.0 if bit else -1.0
        carrier += amp * np.sin(2.0 * math.pi * freq * t)
    carrier /= len(WATERMARK_BITS)
    # Scale to -58 dBFS (~0.00125) so it is completely inaudible to humans
    watermarked = signal.astype(np.float64) + carrier * 0.0012
    return np.clip(watermarked, -0.999, 0.999).astype(np.float32)


def append_wav_provenance_tag(wav_path, model_id: str = "youclone") -> None:
    """Append a lightweight RIFF trailing provenance tag so lossless files carry instant cryptographic attribution."""
    payload = WATERMARK_MAGIC + f"model={model_id};sr={RATE}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16].encode("ascii")
    footer = WATERMARK_MAGIC + digest
    with open(wav_path, "ab") as handle:
        handle.write(footer)


def detect_watermark_file(path) -> dict:
    """Check an audio file for YouClone's spread-spectrum ultrasonic signature and RIFF provenance footer."""
    raw_bytes = path.read_bytes()
    has_footer = WATERMARK_MAGIC in raw_bytes
    signal, sr = sf.read(path, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if len(signal) < sr // 2:
        return {
            "detected": bool(has_footer),
            "confidence": 0.99 if has_footer else 0.0,
            "signature": "YC (YouClone Provenance v1)" if has_footer else "None",
            "details": "Short clip verified via container provenance tag." if has_footer else "Clip is too short for spectral watermark analysis.",
        }
    if sr != RATE:
        signal = resample(signal, sr)
    n = min(len(signal), RATE * 15)
    segment = signal[:n].astype(np.float64)
    t = np.arange(n, dtype=np.float64) / RATE
    matched = 0
    correlations = []
    for idx, expected_bit in enumerate(WATERMARK_BITS):
        freq = 18200.0 + idx * 85.0
        ref = np.sin(2.0 * math.pi * freq * t)
        corr = float(np.dot(segment, ref)) / max(1, n)
        decoded_bit = 1 if corr >= 0 else 0
        if decoded_bit == expected_bit:
            matched += 1
        correlations.append(abs(corr))
    bit_accuracy = matched / len(WATERMARK_BITS)
    energy = float(np.mean(correlations))
    spectral_match = bit_accuracy >= 0.8125 and energy > 1e-6
    detected = bool(has_footer or spectral_match)
    confidence = round(min(0.995, max(0.98 if has_footer else 0.0, (bit_accuracy - 0.5) * 2.0 if spectral_match else bit_accuracy * 0.4)), 3)
    return {
        "detected": detected,
        "confidence": max(0.0, confidence),
        "bits_matched": f"{matched}/{len(WATERMARK_BITS)}",
        "signature": "YC (0x5943 · YouClone Studio)" if detected else "Unmarked / External Audio",
        "method": "Hybrid (RIFF Provenance + 18.2kHz Spread-Spectrum)" if has_footer and spectral_match else (
            "RIFF Provenance Tag" if has_footer else ("18.2kHz Spread-Spectrum Carrier" if spectral_match else "No signature found")
        ),
        "details": (
            f"Verified YouClone synthetic audio ({matched}/{len(WATERMARK_BITS)} carrier bands matched, confidence {max(0.0, confidence) * 100:.1f}%)."
            if detected
            else f"No YouClone watermark detected ({matched}/{len(WATERMARK_BITS)} random carrier alignment)."
        ),
    }


def tempo(signal, speed):
    if abs(speed - 1.0) < 0.005:
        return signal
    clamped = min(2.0, max(0.5, float(speed)))
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "f32le", "-ar", str(RATE), "-ac", "1",
         "-i", "pipe:0", "-af", f"atempo={clamped}", "-f", "f32le", "pipe:1"],
        input=np.asarray(signal, dtype="<f4").tobytes(), capture_output=True, check=True, timeout=120,
    )
    return np.frombuffer(result.stdout, dtype="<f4")


def timed_mix(clips, ranges):
    frames = [(round(start * RATE), round(end * RATE)) for start, end in ranges]
    if any(stop <= start for start, stop in frames):
        raise ValueError("Every timed block needs a positive duration.")
    if any(start < previous_end for (_, previous_end), (start, _) in zip(frames, frames[1:])):
        raise ValueError("Timed blocks must be in order and may not overlap.")
    mixed = np.zeros(frames[-1][1], dtype=np.float32)
    for clip, (start, stop) in zip(clips, frames):
        count = stop - start
        if len(clip) > count:
            speed = len(clip) / count
            if speed > 2:
                raise ValueError("A take is too long for its time range. Shorten the block and regenerate it.")
            clip = tempo(clip, speed)
        length = min(count, len(clip))
        mixed[start:start + length] = clip[:length]
    return mixed


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
