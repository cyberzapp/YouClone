"""Automatic Emotion, Breath & Voice Modulation Director for YouClone.

Analyzes scripts per sentence/beat (English & Hindi) to drive VoxCPM2's
Controllable Voice Cloning `(instruct)` conditioning automatically — with zero
manual setup required — while also honoring optional inline emotion tags like
`[surprised]`, `[fear]`, `[joy]`, `[furious]`, `[whisper]`, `[sad]`, `[calm]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import dsp

INLINE_EMOTION_RE = re.compile(
    r"\[(?P<tag>surprised|shocked|amazed|fear|fearful|scary|suspense|whisper|"
    r"happy|joy|joyful|excited|furious|angry|urgent|serious|sad|somber|"
    r"curious|dramatic|calm|explain|warm|breath|sigh)\]",
    re.I,
)

TAG_TO_EMOTION = {
    "surprised": "surprise",
    "shocked": "surprise",
    "amazed": "surprise",
    "fear": "fear_suspense",
    "fearful": "fear_suspense",
    "scary": "fear_suspense",
    "suspense": "fear_suspense",
    "whisper": "whisper",
    "happy": "joy_happy",
    "joy": "joy_happy",
    "joyful": "joy_happy",
    "excited": "joy_happy",
    "furious": "furious_intense",
    "angry": "furious_intense",
    "urgent": "furious_intense",
    "serious": "furious_intense",
    "sad": "sad_somber",
    "somber": "sad_somber",
    "curious": "curious_question",
    "dramatic": "dramatic_reveal",
    "calm": "warm_explain",
    "explain": "warm_explain",
    "warm": "warm_explain",
    "breath": "fear_suspense",
    "sigh": "sad_somber",
}

# Tuned specifically for OpenBMB VoxCPM2 Controllable Voice Cloning `(instruct)` syntax.
# Each profile has a subtle (balanced documentary) and expressive (strong acting) variant.
EMOTION_PROMPTS: dict[str, tuple[str, str]] = {
    "surprise": (
        "surprised and intrigued tone, expressive rising inflection, crisp word articulation and natural breath",
        "genuinely surprised and amazed tone, lively excited pace, expressive rising inflection and audible breath",
    ),
    "fear_suspense": (
        "slow, tense, suspenseful and grave tone, lowered pitch, deliberate pauses and subtle breath",
        "slow, fearful, hushed and tense tone, lower pitch, heavy suspense with deliberate pauses and audible breath",
    ),
    "whisper": (
        "soft, hushed, intimate near-whisper tone, slow deliberate pacing and audible breath",
        "quiet, eerie whispered tone, very slow tense pacing with pauses and audible breath",
    ),
    "joy_happy": (
        "warm, cheerful, smiling and upbeat tone, bright lively energy and expressive pacing",
        "warm, joyful, smiling and enthusiastic tone, bright energetic delivery, lively happy modulation",
    ),
    "furious_intense": (
        "serious, firm, intense and urgent tone, sharp emphatic articulation and strong vocal projection",
        "furious, intense, stern and urgent tone, sharp forceful articulation, strong vocal emphasis",
    ),
    "sad_somber": (
        "soft, reflective, melancholic and empathetic tone, slower gentle pacing with tender pauses",
        "soft, deeply melancholic and somber tone, slow heavy pauses with gentle sighing breath",
    ),
    "curious_question": (
        "curious, engaging and conversational tone, expressive rising inquisitive inflection",
        "highly curious, leaning-in intriguing tone, clear rising inflection and animated articulation",
    ),
    "dramatic_reveal": (
        "poised, dramatic storytelling tone, deliberate emphasis on key words and crisp articulation",
        "dramatic, impactful reveal tone, slower deliberate emphasis with theatrical pause and breath",
    ),
    "warm_explain": (
        "warm, clear, engaging narrator, natural pitch modulation, crisp articulation and subtle breaths",
        "expressive, warm, animated storytelling narrator, rich vocal modulation, clear articulation and natural breaths",
    ),
}

# Pause multiplier after a beat ends (gives fear/suspense & dramatic reveals breathing room)
EMOTION_PAUSE_BOOST: dict[str, float] = {
    "fear_suspense": 0.22,
    "whisper": 0.25,
    "dramatic_reveal": 0.16,
    "sad_somber": 0.18,
    "surprise": 0.08,
    "furious_intense": 0.06,
    "curious_question": 0.12,
    "joy_happy": 0.0,
    "warm_explain": 0.0,
}

# Lexicon cues (English + Hindi)
_CUES: dict[str, tuple[re.Pattern, float]] = {
    "surprise": (
        re.compile(
            r"\b(?:surpris\w*|shock\w*|unbelievabl\w*|extraordinary|didn't expect|never expected|"
            r"unexpected\w*|caught|turns out|wait a second|believe it or not|mind[- ]blowing|"
            r"plot twist|suddenly|lit up|incredible|astonish\w*|stunning|jaw[- ]dropping|"
            r"whoa|actually happened|you won't believe|not going to believe)\b|"
            r"(?:हैरान|चौंक|अविश्वसनीय|अचानक|कमाल हो गया)",
            re.I,
        ),
        2.2,
    ),
    "fear_suspense": (
        re.compile(
            r"\b(?:fear\w*|scary|scared|terrify\w*|terrified|horrif\w*|creepy|eerie|dark\w*|shadow\w*|"
            r"black box|hidden|secret\w*|silently|danger\w*|threat\w*|deadly|warning|beware|"
            r"interfere\w*|suppress\w*|trapped|haunt\w*|nightmare|chilling|sinister|ominous|"
            r"hiding|in control|watching you|stalking|panic|dread)\b|"
            r"(?:डर|खतरनाक|भयानक|रहस्य|छुपा|खतरा|अंधेरा)",
            re.I,
        ),
        2.1,
    ),
    "joy_happy": (
        re.compile(
            r"\b(?:happy|happily|joy\w*|wonderful|awesome|amazing|fantastic|brilliant|celebrat\w*|"
            r"congratulations|good news|best news|great news|finally|cracked it|won|victory|"
            r"triumph|delight\w*|thrilled|excited|exciting|love it|smile|smiling|laugh\w*|"
            r"proud|breakthrough|miracle|cheerful|blessed|dream come true)\b|"
            r"(?:खुश|खुशी|शानदार|बधाई|कमाल|जीत|मज़ा|अच्छी खबर)",
            re.I,
        ),
        2.2,
    ),
    "furious_intense": (
        re.compile(
            r"\b(?:furious\w*|angry|anger|outrage\w*|unacceptable|ridiculous|disgusting|betray\w*|"
            r"scam|fraud|lying|lies|liar|destroyed|ruined|worst|horrible|terrible|pathetic|"
            r"enough is enough|never again|shameful|corrupt\w*|illegal|reckless|infuriat\w*|"
            r"locked behind|ripped off|stealing|exploitation)\b|"
            r"(?:गुस्सा|अस्वीकार्य|धोखा|बर्बाद|शर्मनाक|झूठ)",
            re.I,
        ),
        2.4,
    ),
    "sad_somber": (
        re.compile(
            r"\b(?:sad\w*|tragic|tragedy|heartbreak\w*|grief|lost everything|lonely|tears|crying|"
            r"unfortunately|regret\w*|painful|devastat\w*|passed away|died|forgotten|hopeless|"
            r"broken|sorrow|mourn\w*)\b|"
            r"(?:दुख|दर्द|अफसोस|आंसू|अकेला)",
            re.I,
        ),
        2.1,
    ),
    "curious_question": (
        re.compile(
            r"\b(?:so how|what else|why would|have you ever|ever wondered|how does|what happens|"
            r"what if|think about|imagine if|could it be|did you know)\b|"
            r"(?:सोचो|क्या आप|कैसे|आखिर क्यों)",
            re.I,
        ),
        1.6,
    ),
}

_ABBR_END = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|No|vs|Vs|etc|approx|fig|def)|e\.g|i\.e)\.$")


@dataclass
class DirectedBeat:
    text: str
    pause: float
    emotion: str
    instruct: str
    explicit: bool = False


def classify_sentence(sentence: str, style_exaggeration: float = 45.0, user_style: str = "") -> tuple[str, str, str, bool]:
    """Return (clean_sentence, emotion_key, voxcpm_instruct, was_explicit_tag)."""
    explicit_emotion: str | None = None

    def _capture_tag(match: re.Match) -> str:
        nonlocal explicit_emotion
        tag = match.group("tag").lower()
        if explicit_emotion is None:
            explicit_emotion = TAG_TO_EMOTION.get(tag)
        return ""

    clean = INLINE_EMOTION_RE.sub(_capture_tag, sentence)
    clean = re.sub(r"[ \t]{2,}", " ", clean).strip()
    if not clean:
        return "", "warm_explain", "", False

    # If the user explicitly typed a custom style instruction in Voice Settings, honor it directly
    if user_style.strip() and not explicit_emotion:
        return clean, "custom", user_style.strip(), True

    if explicit_emotion:
        subtle, strong = EMOTION_PROMPTS[explicit_emotion]
        chosen = strong if style_exaggeration >= 30.0 else subtle
        return clean, explicit_emotion, chosen, True

    # Score semantic + punctuation signals
    scores: dict[str, float] = {k: 0.0 for k in _CUES}
    for emo, (pattern, weight) in _CUES.items():
        hits = pattern.findall(clean)
        if hits:
            scores[emo] += len(hits) * weight

    has_bang = "!" in clean
    has_q = clean.rstrip(' "\'”’').endswith("?")
    has_ellipsis = "..." in clean or "…" in clean
    has_colon_or_dash = ":" in clean or "—" in clean or " - " in clean
    word_count = len(clean.split())

    if has_q:
        scores["curious_question"] += 1.8
        if scores["fear_suspense"] > 0:
            scores["fear_suspense"] += 0.9
    if has_ellipsis:
        scores["fear_suspense"] += 1.6
    if has_bang:
        if scores["furious_intense"] > 0:
            scores["furious_intense"] += 1.8
        elif scores["joy_happy"] > 0:
            scores["joy_happy"] += 1.8
        elif scores["surprise"] > 0:
            scores["surprise"] += 1.8
        else:
            scores["surprise"] += 1.5
    if "?!" in clean or "!?" in clean:
        scores["surprise"] += 2.5

    best_emo, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score < 1.5:
        if word_count <= 7 and (has_colon_or_dash or clean.endswith(".")):
            best_emo = "dramatic_reveal"
        else:
            best_emo = "warm_explain"

    subtle, strong = EMOTION_PROMPTS[best_emo]
    # Use strong acting prompt when intensity is high OR multiple strong cues triggered
    use_strong = style_exaggeration >= 60.0 or best_score >= 4.0
    return clean, best_emo, (strong if use_strong else subtle), False


def split_emotional_beats(
    text: str,
    comma: float = 0.22,
    period: float = 0.42,
    *,
    style_exaggeration: float = 45.0,
    user_style: str = "",
    max_beat_chars: int = 340,
) -> list[DirectedBeat]:
    """Split text into emotion-coherent beats so every emotional turn gets its own vocal modulation."""
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    cleaned = re.sub(r"\*\*|__|``|`", "", cleaned)
    cleaned = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]*\r?\n+[ \t]*", "\n", cleaned)
    cleaned = dsp.INLINE_PHONETIC.sub(r"\1", cleaned)

    beats: list[DirectedBeat] = []
    parts = dsp.BREAK.split(cleaned)

    for index in range(0, len(parts), 5):
        raw_block = parts[index].strip()
        if raw_block:
            paragraphs = [p.strip() for p in re.split(r"\n+", raw_block) if p.strip()]
            for p_idx, para in enumerate(paragraphs):
                # Split into sentences while keeping inline [tags] attached to the following sentence
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

                # Classify each sentence and group consecutive sentences that share the SAME emotion
                cur_texts: list[str] = []
                cur_emo: str | None = None
                cur_inst: str = ""
                cur_explicit: bool = False

                def _flush(pause_s: float):
                    nonlocal cur_texts, cur_emo, cur_inst, cur_explicit
                    if not cur_texts:
                        return
                    emo_key = cur_emo or "warm_explain"
                    boost = EMOTION_PAUSE_BOOST.get(emo_key, 0.0) if pause_s > 0 else 0.0
                    beats.append(
                        DirectedBeat(
                            text=" ".join(cur_texts),
                            pause=pause_s + boost,
                            emotion=emo_key,
                            instruct=cur_inst,
                            explicit=cur_explicit,
                        )
                    )
                    cur_texts = []
                    cur_emo = None
                    cur_inst = ""
                    cur_explicit = False

                for s in sents:
                    clean_s, emo, inst, was_exp = classify_sentence(s, style_exaggeration, user_style)
                    if not clean_s:
                        continue
                    joined_len = sum(len(x) for x in cur_texts) + len(cur_texts) + len(clean_s)
                    # Start a new beat on ANY emotional shift or explicit tag or length cap
                    if cur_texts and (emo != cur_emo or was_exp or joined_len > max_beat_chars):
                        # Slightly longer breath pause when shifting emotions
                        shift_pause = max(period, 0.42) if emo != cur_emo else period
                        _flush(shift_pause)
                    if not cur_texts:
                        cur_emo = emo
                        cur_inst = inst
                        cur_explicit = was_exp
                    cur_texts.append(clean_s)

                is_last_para = p_idx == len(paragraphs) - 1
                _flush(0.0 if is_last_para else max(period * 1.45, 0.55))

        # Handle explicit <break time="..."/> or [pause ...] tags
        if index + 4 < len(parts):
            raw_val = parts[index + 1] or parts[index + 3] or "0.6"
            unit = (parts[index + 2] or parts[index + 4] or "s").lower()
            explicit_pause = float(raw_val) / (1000.0 if unit == "ms" else 1.0)
            if beats:
                beats[-1].pause = max(beats[-1].pause, explicit_pause)
            else:
                beats.append(DirectedBeat(text="", pause=explicit_pause, emotion="warm_explain", instruct=""))

    return beats
