"""Kyutai Moshi 백엔드 — full-duplex 라이브 전용 (파일→파일 아님).

Moshi(full-duplex)는 사용자/시스템 오디오를 동시 스트림으로 모델링하는 실시간 대화 모델이라
'입력 wav 한 개 → 응답 wav 한 개'식 단일턴 오프라인 API가 없다. 로컬 실행은 **라이브 웹 데모**다:

  uv run --python 3.12 --with moshi_mlx python -m moshi_mlx.local_web -q 4   # http://localhost:8998

따라서 이 backend 는 파일→파일 계약(`generate(in_wav,out_wav)`)을 충족할 수 없고, 호출 시 라이브
런처로 안내하는 스텁이다. 실제 로컬 실행은 lab-ui 의 "Moshi (full-duplex 라이브)" 카드(기동/정지/링크)
또는 위 명령으로 한다.
"""
from __future__ import annotations

from .base import S2SBackend

_LIVE_HINT = (
    "Moshi는 full-duplex 실시간 모델이라 파일→파일로 실행하지 않습니다. "
    "lab-ui 'Moshi (full-duplex 라이브)' 카드로 기동하거나, "
    "`uv run --python 3.12 --with moshi_mlx python -m moshi_mlx.local_web -q 4` 후 http://localhost:8998 에서 대화하세요."
)


class MoshiBackend(S2SBackend):
    name = "moshi"

    def __init__(self, cfg=None):
        self.cfg = cfg or {}

    def available(self) -> bool:
        return False  # 파일→파일 계약으로는 실행 불가(라이브 전용)

    def diagnostics(self) -> list[str]:
        return [_LIVE_HINT]

    def generate(self, in_wav: str, out_wav: str) -> dict:
        raise RuntimeError(_LIVE_HINT)
