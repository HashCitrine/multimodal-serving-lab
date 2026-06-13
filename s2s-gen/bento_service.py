#!/usr/bin/env python3
"""Sesame CSM(표현형 TTS) 를 BentoML 서비스로 패키징.

0→1 서빙: STT(whisper-stt)·TTS(piper-tts)와 동일하게 모델을 프레임워크(BentoML)로 감싸
REST API·헬스체크·동시성/타임아웃 관리를 한 번에 얻는다. 핵심 동기는 **저지연**이다 —
CSM-1B 는 로드가 무거워(수~수십 초) 매 요청마다 로드하면 실시간 대화가 불가능하다.
서비스로 띄워 모델을 **1회 로드·상주**시키면 이후 요청은 합성 비용만 든다.

CSM 본체(레포 + 가중치)는 외부 clone(`CSM_DIR`)·HF 게이트라 저장소에 포함하지 않는다.
로드/합성 로직은 `csm_runtime` 공통 모듈을 backends/csm.py 와 공유한다.

실행(저지연 상주):
    CSM_DIR=../_external/csm \
    uv run --extra csm bentoml serve bento_service:CSMTTS --port 3003
    # 합성 요청:
    curl -s -X POST http://127.0.0.1:3003/synthesize \
         -H 'Content-Type: application/json' \
         -d '{"text":"hello from csm"}' -o out.wav
"""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Annotated

import bentoml

import csm_runtime

SCRIPT_DIR = Path(__file__).resolve().parent

# --- 서비스 설정 레버(환경변수) — stt/tts 서비스 컨벤션 ---
# CSM_DIR    : SesameAILabs/csm clone 경로(필수). 없으면 config.yaml 의 csm_dir.
# CSM_DEVICE : auto|cpu|cuda|mps (auto→cuda→mps→cpu)
# CSM_SPEAKER/CSM_MAX_AUDIO_MS : 합성 기본값
# BENTO_WORKERS : 서비스 프로세스(replica) 수 (CSM 은 무거워 기본 1 권장)
_WORKERS = int(os.environ.get("BENTO_WORKERS", "1"))


def _config_csm_dir() -> str:
    """config.yaml 의 csm_dir(상대경로면 s2s-gen 기준 절대화) — 환경변수 미지정 시 폴백."""
    try:
        import yaml
        cfg = yaml.safe_load(open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8")) or {}
    except Exception:
        return ""
    raw = cfg.get("csm_dir", "") or ""
    if not raw:
        return ""
    p = Path(raw).expanduser()
    return str(p if p.is_absolute() else (SCRIPT_DIR / p).resolve())


def _resolve_csm_dir() -> str:
    d = os.environ.get("CSM_DIR", "") or _config_csm_dir()
    if not d:
        raise RuntimeError(
            "CSM_DIR 미설정 — git clone SesameAILabs/csm 후 CSM_DIR(또는 config.yaml csm_dir)을 지정하세요."
        )
    if not (Path(d).expanduser() / "generator.py").exists():
        raise RuntimeError(f"CSM repo 에 generator.py 가 없습니다: {d}")
    return d


@bentoml.service(
    name="csm-tts",
    workers=_WORKERS,
    traffic={"timeout": 600, "concurrency": 4},   # CSM 합성은 무거움 → 낮은 동시성·긴 타임아웃
)
class CSMTTS:
    """Sesame CSM 표현형 음성 합성 서비스(모델 1회 로드·상주)."""

    def __init__(self) -> None:
        csm_dir = _resolve_csm_dir()
        self.device = csm_runtime.resolve_device(os.environ.get("CSM_DEVICE", "auto"))
        self.speaker = int(os.environ.get("CSM_SPEAKER", "0"))
        self.max_audio_ms = float(os.environ.get("CSM_MAX_AUDIO_MS", "20000"))
        t0 = time.perf_counter()
        self.gen = csm_runtime.load_generator(csm_dir, self.device)
        self.sample_rate = int(self.gen.sample_rate)
        # 워밍업(첫 요청의 컴파일/캐시 지연 흡수) — 짧은 발화 1회.
        tmp = Path(tempfile.gettempdir()) / "csm_warmup.wav"
        csm_runtime.synthesize_pcm16(self.gen, "warm up", self.speaker, 2000, tmp)
        print(f"[csm-tts] loaded device={self.device} sr={self.sample_rate} "
              f"({time.perf_counter()-t0:.1f}s)")

    @bentoml.api
    def synthesize(
        self, text: str, speaker: int = -1, max_audio_ms: float = 0.0
    ) -> Annotated[Path, bentoml.validators.ContentType("audio/wav")]:
        """텍스트 → WAV(PCM16) 파일."""
        spk = self.speaker if speaker < 0 else int(speaker)
        ms = self.max_audio_ms if max_audio_ms <= 0 else float(max_audio_ms)
        tmp = Path(tempfile.gettempdir()) / f"csm_{int(time.time()*1000)}.wav"
        csm_runtime.synthesize_pcm16(self.gen, text, spk, ms, tmp)
        return tmp

    @bentoml.api
    def synthesize_meta(
        self, text: str, speaker: int = -1, max_audio_ms: float = 0.0
    ) -> dict:
        """벤치/검증용: 합성하되 RTF 등 메트릭만 반환(오디오는 임시파일로 버림)."""
        spk = self.speaker if speaker < 0 else int(speaker)
        ms = self.max_audio_ms if max_audio_ms <= 0 else float(max_audio_ms)
        tmp = Path(tempfile.gettempdir()) / f"csm_meta_{int(time.time()*1000)}.wav"
        t0 = time.perf_counter()
        audio_s = csm_runtime.synthesize_pcm16(self.gen, text, spk, ms, tmp)
        synth_s = time.perf_counter() - t0
        try:
            tmp.unlink()
        except OSError:
            pass
        return {
            "audio_seconds": round(audio_s, 3),
            "synth_seconds": round(synth_s, 4),
            "rtf": round(synth_s / audio_s, 4) if audio_s else None,
            "realtime_x": round(audio_s / synth_s, 2) if synth_s else None,
            "chars": len(text),
            "device": self.device,
            "speaker": spk,
        }
