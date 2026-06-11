"""Static 백엔드 — 립싱크 모델 없이 '정지 얼굴 + 오디오'를 mp4로 합성.

입모양은 움직이지 않는다(립싱크 아님). 목적:
  1) 무거운 모델 설치 전에도 **파이프라인 배선(text→LLM→TTS→영상)을 end-to-end로 검증**.
  2) 실제 립싱크 백엔드의 폴백/베이스라인.
ffmpeg 만 있으면 어디서나(Mac/NVIDIA) 동작한다.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .base import LipSyncBackend


class StaticBackend(LipSyncBackend):
    name = "static"

    def __init__(self, ffmpeg: str = "ffmpeg"):
        self.ffmpeg = shutil.which(ffmpeg) or ffmpeg

    def available(self) -> bool:
        return shutil.which(self.ffmpeg) is not None or Path(self.ffmpeg).exists()

    def diagnostics(self) -> list[str]:
        if self.available():
            return []
        return [f"ffmpeg not found: {self.ffmpeg}"]

    def generate(self, face_path: str, audio_path: str, out_path: str) -> Path:
        # 정지 이미지를 오디오 길이만큼 반복해 영상화
        cmd = [
            self.ffmpeg, "-y",
            "-loop", "1", "-i", face_path,
            "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest", "-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2",
            out_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(out_path)
