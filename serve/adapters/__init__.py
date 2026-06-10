"""어댑터 레지스트리.

이름 → 어댑터 생성 함수 매핑. 새 모달리티(tts/stt/llm/avatar)를 추가할 때
여기 등록만 하면 서버 config의 adapter.name 으로 선택할 수 있다.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from .base import ModelAdapter
from .echo import EchoAdapter


def _build_echo(cfg: Dict[str, Any]) -> ModelAdapter:
    return EchoAdapter(
        latency_ms=float(cfg.get("latency_ms", 50.0)),
        per_item_ms=float(cfg.get("per_item_ms", 0.0)),
    )


def _build_piper_tts(cfg: Dict[str, Any]) -> ModelAdapter:
    from .piper_tts import PiperTTSAdapter

    return PiperTTSAdapter(
        voice=cfg.get("voice", "en_US-lessac-medium"),
        model_dir=cfg.get("model_dir", ""),
        length_scale=float(cfg.get("length_scale", 1.0)),
        use_cuda=bool(cfg.get("use_cuda", False)),
    )


def _build_whisper_stt(cfg: Dict[str, Any]) -> ModelAdapter:
    from .whisper_stt import WhisperSTTAdapter

    return WhisperSTTAdapter(
        model=cfg.get("model", "base.en"),
        compute_type=cfg.get("compute_type", "int8"),
        device=cfg.get("device", "auto"),
        beam_size=int(cfg.get("beam_size", 1)),
    )


# name -> factory(config_dict) -> ModelAdapter
REGISTRY: Dict[str, Callable[[Dict[str, Any]], ModelAdapter]] = {
    "echo": _build_echo,
    "piper-tts": _build_piper_tts,
    "whisper-stt": _build_whisper_stt,
    # 이후 단계에서 등록 예정:
    # "llm": _build_llm,
    # "avatar": _build_avatar,
}


def build_adapter(cfg: Dict[str, Any]) -> ModelAdapter:
    name = cfg.get("name", "echo")
    if name not in REGISTRY:
        raise ValueError(f"unknown adapter '{name}'. available: {list(REGISTRY)}")
    return REGISTRY[name](cfg)
