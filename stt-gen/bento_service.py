#!/usr/bin/env python3
"""faster-whisper STT 를 BentoML 서비스로 패키징.

0→1 서빙: 모델 → 프레임워크(BentoML) → REST API(파일 업로드)·헬스·동시성·Docker.
모델/양자화는 환경변수로 제어해 같은 서비스로 양자화 트레이드오프를 재현한다.

실행:
    pip install -r requirements.txt
    STT_MODEL=base.en STT_COMPUTE=int8 bentoml serve bento_service:WhisperSTT  # :3000
    curl -s -X POST localhost:3000/transcribe -F 'audio=@clip.wav'
"""
from __future__ import annotations

import os
import time
import wave
from pathlib import Path

import bentoml

MODEL = os.environ.get("STT_MODEL", "base.en")
COMPUTE = os.environ.get("STT_COMPUTE", "int8")
DEVICE = os.environ.get("STT_DEVICE", "auto")   # auto|cpu|cuda (NVIDIA면 auto가 cuda)
WORKERS = int(os.environ.get("BENTO_WORKERS", "1"))


@bentoml.service(
    name="whisper-stt",
    workers=WORKERS,
    traffic={"timeout": 120, "concurrency": 32},
)
class WhisperSTT:
    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        self.model = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
        print(f"[whisper-stt] {MODEL} device={DEVICE} compute_type={COMPUTE}")

    def _transcribe(self, path: str):
        segs, info = self.model.transcribe(path, beam_size=1)
        return "".join(s.text for s in segs).strip()

    @bentoml.api
    def transcribe(self, audio: Path) -> dict:
        """wav 업로드 → 텍스트 + RTF."""
        with wave.open(str(audio), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        t0 = time.perf_counter()
        text = self._transcribe(str(audio))
        dt = time.perf_counter() - t0
        return {
            "text": text,
            "audio_seconds": round(dur, 3),
            "transcribe_seconds": round(dt, 4),
            "rtf": round(dt / dur, 4) if dur else None,
            "model": MODEL,
            "compute_type": COMPUTE,
        }
