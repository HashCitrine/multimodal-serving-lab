"""립싱크 백엔드 레지스트리. config 의 backend 이름으로 선택."""
from __future__ import annotations

from typing import Any, Dict

from .base import LipSyncBackend


def build_backend(cfg: Dict[str, Any]) -> LipSyncBackend:
    name = cfg.get("backend", "static")
    if name == "static":
        from .static import StaticBackend
        return StaticBackend(ffmpeg=cfg.get("ffmpeg", "ffmpeg"))
    if name == "wav2lip":
        from .wav2lip import Wav2LipBackend
        return Wav2LipBackend(
            repo_dir=cfg.get("wav2lip_dir", ""),
            checkpoint=cfg.get("wav2lip_ckpt", ""),
            device=cfg.get("device", "auto"),
        )
    raise ValueError(f"unknown lipsync backend '{name}' (static|wav2lip)")
