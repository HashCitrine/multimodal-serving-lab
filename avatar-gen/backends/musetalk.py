"""MuseTalk 백엔드 (외부 레포 셸아웃, 실시간 립싱크).

MuseTalk 은 잠재공간 인페인팅 기반 실시간 고품질 립싱크 모델이다(V100 기준 30fps+, 다국어).
본체(레포 + 체크포인트)는 무겁고 GPU 의존성이 별도라 여기 포함하지 않는다(=Wav2Lip 백엔드와
동일 원칙). 외부 설치 경로를 config/env 로 받아 그 inference 를 호출한다.

준비(요약):
  git clone https://github.com/TMElyralab/MuseTalk
  체크포인트(models/musetalk, sd-vae, whisper 등) 다운로드 → MUSETALK_DIR / MUSETALK_CKPT 지정
  GPU 전용에 가깝다(실시간은 CUDA 권장). Mac(MPS)에서는 diagnostics 로 안내한다.

가중치/레포가 없으면 diagnostics() 로 안내하고 generate() 는 명확히 실패한다.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from .base import LipSyncBackend


class MuseTalkBackend(LipSyncBackend):
    name = "musetalk"

    def __init__(
        self,
        repo_dir: str = "",
        checkpoint: str = "",
        device: str = "auto",
        base_dir: Optional[Union[str, Path]] = None,
        fps: Optional[float] = None,
        bbox_shift: Optional[int] = None,
        gpu_id: Optional[int] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
        repo_value = repo_dir or os.environ.get("MUSETALK_DIR", "")
        checkpoint_value = checkpoint or os.environ.get("MUSETALK_CKPT", "")
        self.repo_configured = bool(repo_value)
        self.checkpoint_configured = bool(checkpoint_value)
        self.repo_dir = self._resolve(repo_value)
        self.checkpoint = self._resolve(checkpoint_value)
        self.device = device
        self.fps = fps
        self.bbox_shift = bbox_shift
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
            issues.append("MuseTalk repo is not configured (set musetalk_dir or MUSETALK_DIR)")
        elif not self.repo_dir.exists():
            issues.append(f"MuseTalk repo not found: {self.repo_dir}")
        elif not (self.repo_dir / "inference.py").exists() and not (self.repo_dir / "scripts" / "inference.py").exists():
            issues.append(f"inference.py not found under MuseTalk repo: {self.repo_dir}")

        if not self.checkpoint_configured:
            issues.append("MuseTalk checkpoint dir is not configured (set musetalk_ckpt or MUSETALK_CKPT)")
        elif not self.checkpoint.exists():
            issues.append(f"MuseTalk checkpoint not found: {self.checkpoint}")

        if shutil.which("ffmpeg") is None:
            issues.append("ffmpeg not found on PATH")
        return issues

    def _inference_script(self) -> Path:
        nested = self.repo_dir / "scripts" / "inference.py"
        return nested if nested.exists() else (self.repo_dir / "inference.py")

    def generate(self, face_path: str, audio_path: str, out_path: str) -> Path:
        if not self.available():
            detail = "\n".join(f"- {issue}" for issue in self.diagnostics())
            raise RuntimeError(
                "MuseTalk backend is not ready:\n"
                f"{detail}\n"
                "Set musetalk_dir/musetalk_ckpt in config.yaml or MUSETALK_DIR/MUSETALK_CKPT."
            )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(self._inference_script()),
            "--video_path", face_path,      # MuseTalk 은 정지 이미지/영상 모두 입력 가능
            "--audio_path", audio_path,
            "--result_dir", str(out.parent),
            "--output_name", out.stem,
        ]
        if self.bbox_shift is not None:
            cmd.extend(["--bbox_shift", str(self.bbox_shift)])
        if self.fps:
            cmd.extend(["--fps", str(self.fps)])

        env = os.environ.copy()
        if self.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        subprocess.run(cmd, check=True, cwd=str(self.repo_dir), env=env)
        return Path(out_path)
