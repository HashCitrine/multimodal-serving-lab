"""더미 Echo 어댑터.

무거운 모델 없이 서빙 레이어(큐·동적 배칭·메트릭·벤치)를 end-to-end로
검증하기 위한 어댑터. 지정한 지연(latency_ms)만큼 블로킹한 뒤 입력을 그대로
(대문자로 변환해) 돌려준다. 실제 모델 추론의 '블로킹 + 배치' 특성을 흉내 낸다.
"""
from __future__ import annotations

import time
from typing import Any, List

from .base import ModelAdapter


class EchoAdapter(ModelAdapter):
    name = "echo"

    def __init__(self, latency_ms: float = 50.0, per_item_ms: float = 0.0):
        # latency_ms: 배치 1회당 고정 지연(모델 forward 비용 흉내)
        # per_item_ms: 배치 항목당 추가 지연(배치가 클수록 느려지는 모델 흉내)
        self.latency_ms = latency_ms
        self.per_item_ms = per_item_ms

    def infer(self, batch: List[Any]) -> List[Any]:
        time.sleep((self.latency_ms + self.per_item_ms * len(batch)) / 1000.0)
        out = []
        for x in batch:
            text = x.get("input", "") if isinstance(x, dict) else str(x)
            out.append({"echo": str(text).upper(), "len": len(str(text))})
        return out
