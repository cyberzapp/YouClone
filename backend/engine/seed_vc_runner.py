"""Persistent Seed-VC 44.1 kHz F0-Conditioned Voice Conversion worker.

Runs inside `.voxm5/seed-vc/.venv/bin/python` (PyTorch MPS) and communicates with
YouClone's main Python 3.12 MLX server over JSON lines on stdin/stdout.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

SEED_VC_ROOT = Path(__file__).resolve().parent.parent.parent / ".voxm5" / "seed-vc"
os.chdir(str(SEED_VC_ROOT))
sys.path.insert(0, str(SEED_VC_ROOT))
os.environ["HF_HUB_CACHE"] = str(SEED_VC_ROOT / "checkpoints" / "hf_cache")
if (SEED_VC_ROOT / "checkpoints" / "models--Plachta--Seed-VC").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
import yaml

from hf_utils import load_custom_model_from_hf
from modules.audio import mel_spectrogram
from modules.bigvgan import bigvgan
from modules.campplus.DTDNN import CAMPPlus
from modules.commons import build_model, load_checkpoint, recursive_munch
from modules.rmvpe import RMVPE
from transformers import AutoFeatureExtractor, WhisperModel


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_cached_file(pattern: str) -> str | None:
    ckpt_root = SEED_VC_ROOT / "checkpoints"
    for p in ckpt_root.rglob(pattern):
        if p.is_file():
            return str(p)
    return None


def _crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = (
            chunk2[:overlap] * fade_in[: len(chunk2)]
            + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
        )
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def _adjust_f0_semitones(f0_seq: torch.Tensor, n_semitones: float) -> torch.Tensor:
    return f0_seq * (2.0 ** (float(n_semitones) / 12.0))


class SeedVC44kWorker:
    def __init__(self):
        self.device = _get_device()
        self.loaded = False
        self.active_ckpt: str | None = None
        self.model = None
        self.semantic_fn = None
        self.f0_fn = None
        self.vocoder_fn = None
        self.campplus_model = None
        self.to_mel = None
        self.sr = 44100
        self.hop_length = 512

    def unload(self):
        self.model = None
        self.semantic_fn = None
        self.f0_fn = None
        self.vocoder_fn = None
        self.campplus_model = None
        self.to_mel = None
        self.active_ckpt = None
        self.loaded = False
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def _resolve_base_ckpt(self) -> tuple[str, str]:
        cached_pth = _find_cached_file("DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth")
        cached_yml = str(SEED_VC_ROOT / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
        if cached_pth and os.path.isfile(cached_yml):
            return cached_pth, cached_yml
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        return load_custom_model_from_hf(
            "Plachta/Seed-VC",
            "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
            "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        )

    def _ensure_loaded(self, checkpoint_path: str | None, config_path: str | None):
        base_pth, base_yml = self._resolve_base_ckpt()
        target_ckpt = checkpoint_path if (checkpoint_path and os.path.isfile(checkpoint_path)) else base_pth
        target_cfg = config_path if (config_path and os.path.isfile(config_path)) else base_yml

        if self.loaded and self.active_ckpt == target_ckpt:
            return

        config = yaml.safe_load(open(target_cfg, "r"))
        model_params = recursive_munch(config["model_params"])
        model_params.dit_type = "DiT"
        self.sr = int(config["preprocess_params"]["sr"])
        self.hop_length = int(config["preprocess_params"]["spect_params"]["hop_length"])

        # Reload DiT weights only if checkpoint changed
        model = build_model(model_params, stage="DiT")
        model, _, _, _ = load_checkpoint(
            model, None, target_ckpt, load_only_params=True, ignore_modules=[], is_distributed=False
        )
        for key in model:
            model[key].eval()
            model[key].to(self.device)
        model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.model = model
        self.active_ckpt = target_ckpt

        if not self.loaded:
            rmvpe_path = _find_cached_file("rmvpe.pt") or load_custom_model_from_hf(
                "lj1995/VoiceConversionWebUI", "rmvpe.pt", None
            )
            f0_extractor = RMVPE(rmvpe_path, is_half=False, device=self.device)
            self.f0_fn = f0_extractor.infer_from_audio

            campplus_path = str(SEED_VC_ROOT / "campplus_cn_common.bin")
            if not os.path.isfile(campplus_path):
                campplus_path = _find_cached_file("campplus_cn_common.bin") or load_custom_model_from_hf(
                    "funasr/campplus", "campplus_cn_common.bin", config_filename=None
                )
            campplus = CAMPPlus(feat_dim=80, embedding_size=192)
            campplus.load_state_dict(torch.load(campplus_path, map_location="cpu"))
            campplus.eval().to(self.device)
            self.campplus_model = campplus

            bigvgan_name = model_params.vocoder.name
            bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
            bigvgan_model.remove_weight_norm()
            self.vocoder_fn = bigvgan_model.eval().to(self.device)

            whisper_name = model_params.speech_tokenizer.name
            whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
            del whisper_model.decoder
            whisper_fe = AutoFeatureExtractor.from_pretrained(whisper_name)

            def _semantic(waves_16k: torch.Tensor) -> torch.Tensor:
                ori_inputs = whisper_fe(
                    [waves_16k.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                ori_feats = whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(self.device)
                with torch.no_grad():
                    out = whisper_model.encoder(
                        ori_feats.to(whisper_model.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                return out.last_hidden_state.to(torch.float32)[:, : waves_16k.size(-1) // 320 + 1]

            self.semantic_fn = _semantic

            mel_args = {
                "n_fft": config["preprocess_params"]["spect_params"]["n_fft"],
                "win_size": config["preprocess_params"]["spect_params"]["win_length"],
                "hop_size": self.hop_length,
                "num_mels": config["preprocess_params"]["spect_params"]["n_mels"],
                "sampling_rate": self.sr,
                "fmin": config["preprocess_params"]["spect_params"].get("fmin", 0),
                "fmax": None
                if config["preprocess_params"]["spect_params"].get("fmax", "None") == "None"
                else 8000,
                "center": False,
            }
            self.to_mel = lambda x: mel_spectrogram(x, **mel_args)
            self.loaded = True

        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def convert(
        self,
        source_path: str,
        target_path: str,
        output_path: str,
        checkpoint_path: str | None = None,
        config_path: str | None = None,
        diffusion_steps: int = 35,
        cfg_rate: float = 0.75,
        auto_f0_adjust: bool = True,
        semi_tone_shift: float = 0.0,
    ) -> dict:
        t0 = time.time()
        self._ensure_loaded(checkpoint_path, config_path)
        sr = self.sr
        hop_length = self.hop_length
        max_context_window = (sr // hop_length) * 30
        overlap_frame_len = 16
        overlap_wave_len = overlap_frame_len * hop_length

        src_np = librosa.load(source_path, sr=sr, mono=True)[0]
        ref_np = librosa.load(target_path, sr=sr, mono=True)[0]
        # Keep ref <= 14.5s so DiT context window handles long 15.5s chunks cleanly
        ref_np = ref_np[: int(sr * 14.5)]

        source_audio = torch.tensor(src_np).unsqueeze(0).float().to(self.device)
        ref_audio = torch.tensor(ref_np).unsqueeze(0).float().to(self.device)

        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
        if converted_waves_16k.size(-1) <= 16000 * 30:
            S_alt = self.semantic_fn(converted_waves_16k)
        else:
            overlapping_time = 5
            S_alt_list = []
            buffer = None
            traversed = 0
            while traversed < converted_waves_16k.size(-1):
                if buffer is None:
                    chunk = converted_waves_16k[:, traversed : traversed + 16000 * 30]
                else:
                    chunk = torch.cat(
                        [buffer, converted_waves_16k[:, traversed : traversed + 16000 * (30 - overlapping_time)]],
                        dim=-1,
                    )
                s_chunk = self.semantic_fn(chunk)
                S_alt_list.append(s_chunk if traversed == 0 else s_chunk[:, 50 * overlapping_time :])
                buffer = chunk[:, -16000 * overlapping_time :]
                traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlapping_time
            S_alt = torch.cat(S_alt_list, dim=1)

        ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        S_ori = self.semantic_fn(ori_waves_16k)

        mel = self.to_mel(source_audio)
        mel2 = self.to_mel(ref_audio)

        target_lengths = torch.LongTensor([int(mel.size(2))]).to(self.device)
        target2_lengths = torch.LongTensor([int(mel2.size(2))]).to(self.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ori_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        F0_ori = torch.from_numpy(self.f0_fn(ori_waves_16k[0], thred=0.03)).float().to(self.device)[None]
        F0_alt = torch.from_numpy(self.f0_fn(converted_waves_16k[0], thred=0.03)).float().to(self.device)[None]

        voiced_ori = F0_ori[F0_ori > 1]
        voiced_alt = F0_alt[F0_alt > 1]
        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log = log_f0_alt.clone()
        if auto_f0_adjust and voiced_ori.numel() > 5 and voiced_alt.numel() > 5:
            med_ori = torch.median(torch.log(voiced_ori + 1e-5))
            med_alt = torch.median(torch.log(voiced_alt + 1e-5))
            shifted_log[F0_alt > 1] = log_f0_alt[F0_alt > 1] - med_alt + med_ori
        shifted_f0 = torch.exp(shifted_log)
        if abs(semi_tone_shift) > 1e-3:
            shifted_f0[F0_alt > 1] = _adjust_f0_semitones(shifted_f0[F0_alt > 1], semi_tone_shift)

        cond, _, _, _, _ = self.model.length_regulator(S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0)
        prompt_cond, _, _, _, _ = self.model.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

        max_source_window = max(128, max_context_window - mel2.size(2))
        processed = 0
        wave_chunks = []
        prev_chunk = None

        while processed < cond.size(1):
            chunk_cond = cond[:, processed : processed + max_source_window]
            is_last = processed + max_source_window >= cond.size(1)
            cat_cond = torch.cat([prompt_cond, chunk_cond], dim=1)
            autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
                vc_target = self.model.cfm.inference(
                    cat_cond,
                    torch.LongTensor([cat_cond.size(1)]).to(self.device),
                    mel2,
                    style2,
                    None,
                    int(diffusion_steps),
                    inference_cfg_rate=float(cfg_rate),
                )
                vc_target = vc_target[:, :, mel2.size(-1) :]
            vc_wave = self.vocoder_fn(vc_target.float()).squeeze()
            if self.device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if vc_wave.ndim == 1:
                vc_wave = vc_wave[None, :]

            if processed == 0:
                if is_last:
                    wave_chunks.append(vc_wave[0].cpu().numpy())
                    break
                wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
                prev_chunk = vc_wave[0, -overlap_wave_len:]
                processed += vc_target.size(2) - overlap_frame_len
            elif is_last:
                wave_chunks.append(_crossfade(prev_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
                break
            else:
                wave_chunks.append(
                    _crossfade(prev_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                )
                prev_chunk = vc_wave[0, -overlap_wave_len:]
                processed += vc_target.size(2) - overlap_frame_len

        full_wave = np.concatenate(wave_chunks).astype(np.float32)
        sf.write(output_path, full_wave, sr, subtype="FLOAT")
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "ok": True,
            "output_path": output_path,
            "sr": sr,
            "seconds": round(len(full_wave) / float(sr), 2),
            "elapsed": round(time.time() - t0, 2),
        }


def main():
    proto_out = sys.stdout
    sys.stdout = sys.stderr

    def _emit(obj: dict):
        proto_out.write("__VC_JSON__:" + json.dumps(obj) + "\n")
        proto_out.flush()

    worker = SeedVC44kWorker()
    _emit({"ready": True, "device": str(worker.device)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "ping":
                res = {"ok": True, "loaded": worker.loaded, "active_ckpt": worker.active_ckpt}
            elif cmd == "unload":
                worker.unload()
                res = {"ok": True}
            elif cmd == "convert":
                res = worker.convert(
                    source_path=req["source_path"],
                    target_path=req["target_path"],
                    output_path=req["output_path"],
                    checkpoint_path=req.get("checkpoint_path"),
                    config_path=req.get("config_path"),
                    diffusion_steps=int(req.get("diffusion_steps", 35)),
                    cfg_rate=float(req.get("cfg_rate", 0.75)),
                    auto_f0_adjust=bool(req.get("auto_f0_adjust", True)),
                    semi_tone_shift=float(req.get("semi_tone_shift", 0.0)),
                )
            elif cmd == "quit":
                worker.unload()
                _emit({"ok": True})
                break
            else:
                res = {"ok": False, "error": f"Unknown cmd: {cmd}"}
        except Exception as exc:
            res = {"ok": False, "error": str(exc)}
        _emit(res)


if __name__ == "__main__":
    main()
