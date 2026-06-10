"""Wav2Lip 백엔드 (외부 레포 셸아웃 스켈레톤).

Wav2Lip 본체(레포 + 체크포인트)는 무겁고 라이선스/의존성이 별도라 여기 포함하지 않는다.
대신 외부 설치 경로를 config/env 로 받아 그 `inference.py` 를 호출한다. 환경이 준비되면
config 의 backend 를 "wav2lip" 으로 바꾸기만 하면 동작한다.

준비(요약):
  git clone https://github.com/Rudrabha/Wav2Lip  (또는 유지보수 포크)
  체크포인트(wav2lip_gan.pth) 다운로드 → WAV2LIP_DIR, WAV2LIP_CKPT 지정
  Mac(MPS): inference 의 device 선택부를 mps 로 패치(레포가 cuda 가정인 경우)
  NVIDIA: 그대로 cuda 사용
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .base import LipSyncBackend


class Wav2LipBackend(LipSyncBackend):
    name = "wav2lip"

    def __init__(self, repo_dir: str = "", checkpoint: str = "", device: str = "auto"):
        self.repo_dir = Path(repo_dir or os.environ.get("WAV2LIP_DIR", ""))
        self.checkpoint = Path(checkpoint or os.environ.get("WAV2LIP_CKPT", ""))
        self.device = device

    def available(self) -> bool:
        return (self.repo_dir.exists()
                and (self.repo_dir / "inference.py").exists()
                and self.checkpoint.exists())

    def generate(self, face_path: str, audio_path: str, out_path: str) -> Path:
        if not self.available():
            raise RuntimeError(
                "Wav2Lip 미설정. WAV2LIP_DIR(레포)·WAV2LIP_CKPT(체크포인트)를 지정하세요. "
                "README 의 준비 절차 참고. (그 전에는 backend=static 으로 파이프라인 검증)"
            )
        cmd = [
            sys.executable, str(self.repo_dir / "inference.py"),
            "--checkpoint_path", str(self.checkpoint),
            "--face", face_path,
            "--audio", audio_path,
            "--outfile", out_path,
        ]
        # 레포가 device 인자를 받는 포크면 전달(원본은 cuda 가정 → MPS는 레포 패치 필요)
        subprocess.run(cmd, check=True, cwd=str(self.repo_dir))
        return Path(out_path)
