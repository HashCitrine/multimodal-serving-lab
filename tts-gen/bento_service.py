#!/usr/bin/env python3
"""Piper TTS 를 BentoML 서비스로 패키징.

0→1 서빙: 모델을 프레임워크(BentoML)로 감싸 REST API·헬스체크·동시성/타임아웃 관리·
Docker 빌드(`bentoml build` → `containerize`)까지 한 번에 얻는다. 직접 만든
`serve/` 베이스라인(동일 모델)을 같은 벤치로 비교해 프레임워크의 가치를 가늠한다.

실행:
    pip install -r requirements.txt
    python synthesize.py --download            # 보이스 모델 1회 내려받기
    bentoml serve bento_service:PiperTTS       # http://127.0.0.1:3000
    # 합성 요청:
    curl -s -X POST http://127.0.0.1:3000/synthesize \
         -H 'Content-Type: application/json' \
         -d '{"text":"hello from bentoml"}' -o out.wav
"""
from __future__ import annotations

import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Annotated

import numpy as np
import bentoml
import onnxruntime as ort

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_VOICE = os.environ.get("PIPER_VOICE", "en_US-lessac-medium")

# --- 처리량 튜닝 레버 (환경변수) ---
# ORT_INTRA_OP: 인스턴스당 onnxruntime intra-op 스레드 수 (0=기본=전체 코어)
# BENTO_WORKERS: 서비스 프로세스(replica) 수
_INTRA = int(os.environ.get("ORT_INTRA_OP", "0"))
_WORKERS = int(os.environ.get("BENTO_WORKERS", "1"))

if _INTRA > 0:
    # Piper가 InferenceSession(sess_options=SessionOptions()) 를 기본값으로 만들므로,
    # 세션 생성 시점에 intra-op 스레드 수를 주입(인스턴스당 코어 점유 제한).
    _orig_session = ort.InferenceSession

    def _session_with_threads(*args, **kwargs):
        so = kwargs.get("sess_options") or ort.SessionOptions()
        so.intra_op_num_threads = _INTRA
        so.inter_op_num_threads = 1
        kwargs["sess_options"] = so
        return _orig_session(*args, **kwargs)

    ort.InferenceSession = _session_with_threads


def _voice_path() -> Path:
    p = MODEL_DIR / f"{DEFAULT_VOICE}.onnx"
    if not p.exists():
        raise FileNotFoundError(
            f"보이스 모델이 없습니다: {p}\n"
            f"먼저 `python synthesize.py --download` 로 내려받으세요."
        )
    return p


@bentoml.service(
    name="piper-tts",
    workers=_WORKERS,                       # 프로세스 복제(replica) 수
    traffic={"timeout": 120, "concurrency": 64},
)
class PiperTTS:
    """Piper(ONNX) 음성 합성 서비스."""

    def __init__(self) -> None:
        from piper import PiperVoice

        # PIPER_CUDA=1 이면 onnxruntime CUDA EP 사용(NVIDIA). 단 onnxruntime-gpu 설치 필요.
        use_cuda = os.environ.get("PIPER_CUDA", "0") == "1"
        self.voice = PiperVoice.load(str(_voice_path()), use_cuda=use_cuda)
        self.sample_rate = self.voice.config.sample_rate
        # 워밍업(첫 요청 지연 흡수)
        from piper import SynthesisConfig
        list(self.voice.synthesize("warm up", SynthesisConfig()))
        print(f"[piper-tts] loaded {DEFAULT_VOICE} (sr={self.sample_rate}, cuda={use_cuda})")

    def _synthesize_int16(self, text: str, length_scale: float) -> np.ndarray:
        from piper import SynthesisConfig

        cfg = SynthesisConfig(length_scale=length_scale)
        chunks = list(self.voice.synthesize(text, cfg))
        return np.concatenate([c.audio_int16_array for c in chunks])

    @bentoml.api
    def synthesize(self, text: str, length_scale: float = 1.0) -> Annotated[Path, bentoml.validators.ContentType("audio/wav")]:
        """텍스트 → WAV 파일."""
        audio = self._synthesize_int16(text, length_scale)
        tmp = Path(tempfile.gettempdir()) / f"piper_{int(time.time()*1000)}.wav"
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(audio.tobytes())
        return tmp

    @bentoml.api
    def synthesize_meta(self, text: str, length_scale: float = 1.0) -> dict:
        """벤치/검증용: 오디오는 버리고 RTF 등 메트릭만 반환."""
        t0 = time.perf_counter()
        audio = self._synthesize_int16(text, length_scale)
        synth_s = time.perf_counter() - t0
        audio_s = len(audio) / self.sample_rate
        return {
            "audio_seconds": round(audio_s, 3),
            "synth_seconds": round(synth_s, 4),
            "rtf": round(synth_s / audio_s, 4) if audio_s else None,
            "realtime_x": round(audio_s / synth_s, 1) if synth_s else None,
            "chars": len(text),
            "voice": DEFAULT_VOICE,
        }
