"""Piper TTS 어댑터 — 직접 만든 baseline 서버로 같은 모델을 서빙하기 위한 래퍼.

목적: BentoML(tts-gen/bento_service.py)과 **동일한 Piper 모델**을 직접 구현 서버에도
올려, 같은 벤치 하니스로 두 서빙 방식을 나란히 비교하기 위함.
배칭/네트워크 오버헤드를 보기 위해 오디오 바이트 대신 메트릭(dict)만 반환한다.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, List

from .base import ModelAdapter

# tts-gen/models 에 받아둔 보이스를 재사용
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "tts-gen" / "models"


class PiperTTSAdapter(ModelAdapter):
    name = "piper-tts"

    def __init__(self, voice: str = "en_US-lessac-medium",
                 model_dir: str = "", length_scale: float = 1.0,
                 use_cuda: bool = False):
        self.voice_name = voice
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self.length_scale = length_scale
        self.use_cuda = use_cuda    # NVIDIA: True + onnxruntime-gpu 설치
        self._voice = None
        self._sc = None

    def load(self) -> None:
        from piper import PiperVoice, SynthesisConfig

        model_path = self.model_dir / f"{self.voice_name}.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"보이스 모델 없음: {model_path}. tts-gen 에서 먼저 다운로드하세요."
            )
        self._voice = PiperVoice.load(str(model_path), use_cuda=self.use_cuda)
        self._sc = SynthesisConfig(length_scale=self.length_scale)
        self.sample_rate = self._voice.config.sample_rate

    def warmup(self) -> None:
        if self._voice is not None:
            list(self._voice.synthesize("warm up", self._sc))

    def infer(self, batch: List[Any]) -> List[Any]:
        import numpy as np

        out = []
        for item in batch:
            text = item.get("input", "") if isinstance(item, dict) else str(item)
            t0 = time.perf_counter()
            chunks = list(self._voice.synthesize(str(text), self._sc))
            synth_s = time.perf_counter() - t0
            audio = np.concatenate([c.audio_int16_array for c in chunks])
            audio_s = len(audio) / self.sample_rate
            out.append({
                "audio_seconds": round(audio_s, 3),
                "synth_seconds": round(synth_s, 4),
                "rtf": round(synth_s / audio_s, 4) if audio_s else None,
            })
        return out
