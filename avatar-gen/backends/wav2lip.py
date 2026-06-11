"""Wav2Lip 백엔드 (외부 레포 셸아웃).

Wav2Lip 본체(레포 + 체크포인트)는 무겁고 라이선스/의존성이 별도라 여기 포함하지 않는다.
대신 외부 설치 경로를 config/env 로 받아 그 `inference.py` 를 호출한다.

준비(요약):
  git clone https://github.com/Rudrabha/Wav2Lip  (또는 유지보수 포크)
  체크포인트(wav2lip_gan.pth) 다운로드 → WAV2LIP_DIR, WAV2LIP_CKPT 지정
  Mac(MPS): inference 의 device 선택부를 mps 로 패치(레포가 cuda 가정인 경우)
  NVIDIA: 그대로 cuda 사용
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from .base import LipSyncBackend


class Wav2LipBackend(LipSyncBackend):
    name = "wav2lip"

    def __init__(
        self,
        repo_dir: str = "",
        checkpoint: str = "",
        device: str = "auto",
        base_dir: Optional[Union[str, Path]] = None,
        pads: Optional[list[int]] = None,
        resize_factor: Optional[int] = None,
        nosmooth: bool = False,
        fps: Optional[float] = None,
        gpu_id: Optional[int] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        repo_value = repo_dir or os.environ.get("WAV2LIP_DIR", "")
        checkpoint_value = checkpoint or os.environ.get("WAV2LIP_CKPT", "")
        self.repo_configured = bool(repo_value)
        self.checkpoint_configured = bool(checkpoint_value)
        self.repo_dir = self._resolve(repo_value)
        self.checkpoint = self._resolve(checkpoint_value)
        self.device = device
        self.pads = tuple(pads) if pads else None
        self.resize_factor = resize_factor
        self.nosmooth = nosmooth
        self.fps = fps
        self.gpu_id = gpu_id

    def _resolve(self, value: str) -> Path:
        if not value:
            return Path()
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.base_dir / path).resolve()

    def available(self) -> bool:
        return not self.diagnostics()

    def diagnostics(self) -> list[str]:
        issues = []
        if not self.repo_configured:
            issues.append("Wav2Lip repo is not configured (set wav2lip_dir or WAV2LIP_DIR)")
        elif not self.repo_dir.exists():
            issues.append(f"Wav2Lip repo not found: {self.repo_dir}")
        elif not (self.repo_dir / "inference.py").exists():
            issues.append(f"inference.py not found under Wav2Lip repo: {self.repo_dir}")

        if not self.checkpoint_configured:
            issues.append("Wav2Lip checkpoint is not configured (set wav2lip_ckpt or WAV2LIP_CKPT)")
        elif not self.checkpoint.exists():
            issues.append(f"Wav2Lip checkpoint not found: {self.checkpoint}")

        face_detector = self.repo_dir / "face_detection" / "detection" / "sfd" / "s3fd.pth"
        if self.repo_configured and self.repo_dir.exists() and not face_detector.exists():
            issues.append(f"face detector weight not found: {face_detector}")
        if shutil.which("ffmpeg") is None:
            issues.append("ffmpeg not found on PATH")
        return issues

    def generate(self, face_path: str, audio_path: str, out_path: str) -> Path:
        if not self.available():
            detail = "\n".join(f"- {issue}" for issue in self.diagnostics())
            raise RuntimeError(
                "Wav2Lip backend is not ready:\n"
                f"{detail}\n"
                "Set wav2lip_dir/wav2lip_ckpt in config.yaml or WAV2LIP_DIR/WAV2LIP_CKPT."
            )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(self.repo_dir / "inference.py"),
            "--checkpoint_path", str(self.checkpoint),
            "--face", face_path,
            "--audio", audio_path,
            "--outfile", out_path,
        ]
        if self.pads:
            cmd.extend(["--pads", *[str(v) for v in self.pads]])
        if self.resize_factor:
            cmd.extend(["--resize_factor", str(self.resize_factor)])
        if self.nosmooth:
            cmd.append("--nosmooth")
        if self.fps:
            cmd.extend(["--fps", str(self.fps)])

        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # 원본 Wav2Lip은 device 인자를 받지 않고 torch.cuda.is_available()를 따른다.
        subprocess.run(cmd, check=True, cwd=str(self.repo_dir), env=env)
        return Path(out_path)
