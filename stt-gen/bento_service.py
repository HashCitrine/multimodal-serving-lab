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
# 기본 전사 언어. ""/"auto"=자동감지. 요청별로 transcribe(language=...) 로 덮어쓸 수 있다.
# (.en 모델은 영어 전용 — 한국어 등은 다국어 모델 STT_MODEL=small 등 필요)
LANGUAGE = os.environ.get("STT_LANGUAGE", "auto")
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
        print(f"[whisper-stt] {MODEL} device={DEVICE} compute_type={COMPUTE} language={LANGUAGE}")

    def _transcribe(self, path: str, language: str = ""):
        lang = None if (not language or language == "auto") else language
        segs, info = self.model.transcribe(
            path,
            beam_size=1,
            language=lang,
            task="transcribe",
            condition_on_previous_text=False,
        )
        return {
            "text": "".join(s.text for s in segs).strip(),
            "detected_language": getattr(info, "language", None) or lang or "auto",
            "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        }

    @bentoml.api
    def transcribe(self, audio: Path, language: str = "") -> dict:
        """wav 업로드 → 텍스트 + RTF. language="" 면 서비스 기본값(STT_LANGUAGE) 사용."""
        lang = language or LANGUAGE
        with wave.open(str(audio), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        t0 = time.perf_counter()
        result = self._transcribe(str(audio), lang)
        dt = time.perf_counter() - t0
        return {
            "text": result["text"],
            "audio_seconds": round(dur, 3),
            "transcribe_seconds": round(dt, 4),
            "rtf": round(dt / dur, 4) if dur else None,
            "model": MODEL,
            "compute_type": COMPUTE,
            "language": lang,
            "detected_language": result["detected_language"],
            "language_probability": result["language_probability"],
        }
