"""faster-whisper STT 어댑터 — 직접 구현 baseline 서버로 같은 모델을 서빙.

입력은 전사할 wav 파일 경로(`{"input": "/path/clip.wav"}`)이며, 메트릭(dict)을 반환한다.
BentoML 서비스(stt-gen/bento_service.py)와 동일 모델을 같은 벤치로 비교하기 위함.
"""
from __future__ import annotations

import time
import wave
from typing import Any, List

from .base import ModelAdapter


class WhisperSTTAdapter(ModelAdapter):
    name = "whisper-stt"

    def __init__(self, model: str = "base.en", compute_type: str = "int8",
                 device: str = "auto", beam_size: int = 1):
        self.model_name = model
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_name, device=self.device,
                                   compute_type=self.compute_type)

    def infer(self, batch: List[Any]) -> List[Any]:
        out = []
        for item in batch:
            path = item.get("input", "") if isinstance(item, dict) else str(item)
            with wave.open(str(path), "rb") as w:
                dur = w.getnframes() / w.getframerate()
            t0 = time.perf_counter()
            segs, _ = self._model.transcribe(str(path), beam_size=self.beam_size)
            text = "".join(s.text for s in segs).strip()
            dt = time.perf_counter() - t0
            out.append({
                "chars": len(text),
                "audio_seconds": round(dur, 3),
                "transcribe_seconds": round(dt, 4),
                "rtf": round(dt / dur, 4) if dur else None,
            })
        return out
