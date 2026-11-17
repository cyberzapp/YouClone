import gc
import os
import re
import time
from collections import OrderedDict

import mlx.core as mx
import numpy as np
from mlx_audio.tts import load

from ..models import MODELS
from .dsp import resample

OMNI_CUE = re.compile(r"\[(?:laughter|sigh|confirmation-en|question-(?:en|ah|oh|ei|yi)|surprise-(?:ah|oh|wa|yo)|dissatisfaction-hnn)\]")


class Engine:
    def __init__(self):
        self.models = {}
        self.omni_prompts = OrderedDict()
        self.idle = max(0, float(os.getenv("VOXM5_MODEL_IDLE", "600")))

    def evict(self):
        expired = [key for key, (_, last) in self.models.items()
                   if self.idle and time.monotonic() - last > self.idle]
        for key in expired:
            del self.models[key]
        if "omni" in expired:
            self.omni_prompts.clear()
        if expired:
            gc.collect()
            mx.clear_cache()

    def load(self, key):
        mx.set_default_device(mx.gpu)
        if key not in self.models:
            model = load(MODELS[key])
            mx.eval(model.parameters())
        else:
            model, _ = self.models[key]
        self.models[key] = (model, time.monotonic())
        return model

    def speaker_embedding(self, source):
        model = self.load("zonos")
        return np.asarray(model.extract_speaker_embedding(source), dtype=np.float32).reshape(192).tolist()

    def omni_prompt(self, model, source, transcript):
        from mlx_audio.tts.models.omnivoice.utils import create_voice_clone_prompt

        identity = (source, source.stat().st_mtime_ns, transcript)
        if identity not in self.omni_prompts:
            tokens = create_voice_clone_prompt(
                str(source), tokenizer=model.audio_tokenizer, ref_text=transcript, max_duration_s=10,
            )
            mx.eval(tokens)
            self.omni_prompts[identity] = tokens
        self.omni_prompts.move_to_end(identity)
        if len(self.omni_prompts) > 32:
            self.omni_prompts.popitem(last=False)
        return self.omni_prompts[identity]

    def generate(self, settings, text, source, seed):
        key = settings.model
        model = self.load(key)
        mx.random.seed(seed)
        if key == "omni":
            from mlx_audio.tts.models.omnivoice.duration import RuleDurationEstimator

            prompt = self.omni_prompt(model, source, settings.transcript)
            # Measure the encoded reference so removed silence does not inflate its speaking time.
            tokens_per_second = model.sample_rate / 960
            spoken_text, cue_count = OMNI_CUE.subn("", text)
            duration = RuleDurationEstimator().estimate_duration(
                spoken_text, settings.transcript, prompt.shape[0],
            ) / tokens_per_second
            # Leave room for nonverbal cues. Bound unusual ratios from short or inaccurate transcripts.
            duration = min(max(1.5, len(spoken_text) / 6) + cue_count, duration + .5 * cue_count)
            options = dict(ref_tokens=prompt, ref_text=settings.transcript,
                           duration_s=max(.4, duration),
                           language=settings.language or "None",
                           guidance_scale=settings.guidance, class_temperature=settings.temperature,
                           instruct=settings.style or "None")
            results = model.generate(text=text, **options)
        elif key == "zonos":
            options = dict(stream=True, temperature=settings.temperature, top_p=settings.top_p,
                           speaker_cfg_scale=settings.guidance, seed=seed,
                           lang_code=settings.language or "en", repetition_penalty=settings.repetition_penalty)
            if settings.embedding:
                options["speaker_embedding"] = np.asarray(settings.embedding, dtype=np.float32)
            else:
                options["ref_audio"] = str(source)
            results = model.generate(text=text, **options)
        else:
            results = model.generate_voice_design(
                text=text, language=settings.language or "Auto", instruct=settings.style,
                temperature=settings.temperature, top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty, stream=True,
            )
        try:
            for result in results:
                audio = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                if audio.size:
                    yield resample(audio, result.sample_rate).astype(np.float32)
        finally:
            self.models[key] = (model, time.monotonic())
