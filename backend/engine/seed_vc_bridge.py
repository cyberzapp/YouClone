"""Bridge between YouClone's Python 3.12 MLX server and the isolated Seed-VC 44.1 kHz
PyTorch MPS worker (`.voxm5/seed-vc/.venv/bin/python`)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
SEED_VC_ROOT = WORKSPACE_ROOT / ".voxm5" / "seed-vc"
SEED_VC_PYTHON = (
    SEED_VC_ROOT / ".venv" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else SEED_VC_ROOT / ".venv" / "bin" / "python"
)
RUNNER_SCRIPT = Path(__file__).resolve().parent / "seed_vc_runner.py"


class SeedVCBridge:
    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def is_available(self) -> bool:
        return SEED_VC_PYTHON.is_file() and (SEED_VC_ROOT / "inference.py").is_file()

    def _read_json_line(self) -> dict:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("Seed-VC worker closed unexpectedly.")
            if line.startswith("__VC_JSON__:"):
                return json.loads(line[len("__VC_JSON__:") :])

    def _ensure_started(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.is_available():
            raise RuntimeError("Seed-VC 44.1 kHz runtime is not installed in .voxm5/seed-vc.")
        env = {
            **os.environ,
            "HF_HUB_CACHE": str(SEED_VC_ROOT / "checkpoints" / "hf_cache"),
            "PYTHONUNBUFFERED": "1",
        }
        self._proc = subprocess.Popen(
            [str(SEED_VC_PYTHON), str(RUNNER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(SEED_VC_ROOT),
            env=env,
        )
        info = self._read_json_line()
        if not info.get("ready"):
            raise RuntimeError(f"Seed-VC worker did not report ready: {info}")

    def _call(self, payload: dict) -> dict:
        with self._lock:
            self._ensure_started()
            assert self._proc is not None and self._proc.stdin is not None
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
                res = self._read_json_line()
            except Exception:
                self.shutdown_unlocked()
                raise
            if not res.get("ok"):
                raise RuntimeError(res.get("error") or "Seed-VC conversion error.")
            return res

    def convert(
        self,
        source_path: Path | str,
        target_path: Path | str,
        output_path: Path | str,
        *,
        checkpoint_path: Path | str | None = None,
        config_path: Path | str | None = None,
        diffusion_steps: int = 35,
        cfg_rate: float = 0.75,
        auto_f0_adjust: bool = True,
        semi_tone_shift: float = 0.0,
    ) -> dict:
        return self._call(
            {
                "cmd": "convert",
                "source_path": str(source_path),
                "target_path": str(target_path),
                "output_path": str(output_path),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
                "config_path": str(config_path) if config_path else None,
                "diffusion_steps": int(diffusion_steps),
                "cfg_rate": float(cfg_rate),
                "auto_f0_adjust": bool(auto_f0_adjust),
                "semi_tone_shift": float(semi_tone_shift),
            }
        )

    def unload(self):
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                try:
                    assert self._proc.stdin is not None
                    self._proc.stdin.write(json.dumps({"cmd": "unload"}) + "\n")
                    self._proc.stdin.flush()
                    self._read_json_line()
                except Exception:
                    self.shutdown_unlocked()

    def shutdown_unlocked(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def shutdown(self):
        with self._lock:
            self.shutdown_unlocked()


def find_base_seedvc_44k_ckpt() -> tuple[Path, Path]:
    cfg = SEED_VC_ROOT / "configs" / "presets" / "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"
    for p in (SEED_VC_ROOT / "checkpoints").rglob("DiT_seed_v2_uvit_whisper_base_f0_44k*.pth"):
        if p.is_file():
            return p, cfg
    raise FileNotFoundError("Base Seed-VC 44.1 kHz checkpoint not found in .voxm5/seed-vc/checkpoints.")


def run_seedvc_finetune(
    dataset_dir: Path,
    run_name: str,
    *,
    max_steps: int = 500,
    batch_size: int = 2,
    save_every: int = 250,
    pretrained_ckpt: Path | None = None,
    progress_cb: Callable[[int, int, float], None] | None = None,
) -> tuple[Path, Path]:
    """Run Seed-VC 44.1 kHz fine-tuning on MPS and return (ft_model.pth, config.yml)."""
    base_ckpt, base_cfg = find_base_seedvc_44k_ckpt()
    ckpt_arg = pretrained_ckpt if (pretrained_ckpt and pretrained_ckpt.is_file()) else base_ckpt

    env = {
        **os.environ,
        "HF_HUB_CACHE": str(SEED_VC_ROOT / "checkpoints" / "hf_cache"),
        "PYTHONUNBUFFERED": "1",
    }
    cmd = [
        str(SEED_VC_PYTHON),
        "train.py",
        "--config",
        str(base_cfg),
        "--pretrained-ckpt",
        str(ckpt_arg),
        "--dataset-dir",
        str(dataset_dir),
        "--run-name",
        run_name,
        "--batch-size",
        str(batch_size),
        "--max-steps",
        str(max_steps),
        "--max-epochs",
        "25",
        "--save-every",
        str(save_every),
        "--num-workers",
        "0",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(SEED_VC_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    step_re = re.compile(r"epoch\s+(\d+),\s+step\s+(\d+),\s+loss:\s+([0-9.]+)")
    last_lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.strip()
        if line:
            last_lines.append(line)
            if len(last_lines) > 20:
                last_lines.pop(0)
        m = step_re.search(line)
        if m and progress_cb:
            progress_cb(int(m.group(2)), max_steps, float(m.group(3)))

    code = proc.wait()
    if code != 0:
        tail = "\n".join(last_lines[-6:])
        raise RuntimeError(f"Seed-VC fine-tuning exited with code {code}: {tail}")

    run_dir = SEED_VC_ROOT / "runs" / run_name
    # Remove intermediate optimizer checkpoints (~2GB each) to keep disk space lean
    for ckpt in run_dir.glob("DiT_epoch_*.pth"):
        ckpt.unlink(missing_ok=True)
    ft_model = run_dir / "ft_model.pth"
    ft_cfg = run_dir / base_cfg.name
    if not ft_model.is_file():
        raise FileNotFoundError(f"Expected fine-tuned checkpoint missing: {ft_model}")
    return ft_model, ft_cfg
