"""S2S 백엔드 레지스트리. config 의 backend 이름으로 선택.

  cascade : STT→LLM→TTS 기준선(앞선 voice-agent 자산 재사용). 무거운 모델 없이 항상 동작.
  csm     : Sesame CSM — STT→LLM→CSM(표현형 TTS) 캐스케이드 변형. cascade 상속, MPS 로컬 실행.
  melo    : MeloTTS — STT→LLM→MeloTTS 다국어 TTS 캐스케이드 변형.
  moshi   : Kyutai Moshi — 풀듀플렉스라 파일→파일 불가. 라이브 전용(lab-ui 런처). 여기선 안내 스텁.
"""
from __future__ import annotations

from typing import Any, Dict

from .base import S2SBackend


def build_backend(cfg: Dict[str, Any]) -> S2SBackend:
    name = cfg.get("backend", "cascade")
    if name == "cascade":
        from .cascade import CascadeBackend
        return CascadeBackend(cfg)
    if name == "moshi":
        from .moshi import MoshiBackend  # 라이브 전용 스텁(파일→파일 불가)
        return MoshiBackend(cfg)
    if name == "csm":
        from .csm import CSMBackend
        return CSMBackend(cfg)  # cascade 상속 — stt/llm 서브딕트 + csm_dir/device/speaker/max_audio_ms
    if name == "melo":
        from .melo import MeloBackend
        return MeloBackend(cfg)
    raise ValueError(f"unknown s2s backend '{name}' (cascade|csm|melo|moshi)")
