"""모델 어댑터 인터페이스.

서빙 레이어는 어떤 모달리티(TTS/STT/LLM/avatar)든 이 인터페이스 뒤에 끼워
동일한 방식으로 로드·추론·교체한다. 추론은 '배치 네이티브'로 정의한다
(스케줄러가 동적 마이크로배칭으로 여러 요청을 모아 한 번에 넘긴다).
"""
from __future__ import annotations

import abc
from typing import Any, List


class ModelAdapter(abc.ABC):
    #: 사람이 읽는 어댑터 이름 (헬스체크/로그에 노출)
    name: str = "base"

    def load(self) -> None:
        """무거운 초기화(모델 가중치 로드 등). 서버 기동 시 1회 호출."""
        pass

    def warmup(self) -> None:
        """첫 요청 지연(컴파일/캐시)을 흡수하기 위한 워밍업. 선택."""
        pass

    @abc.abstractmethod
    def infer(self, batch: List[Any]) -> List[Any]:
        """입력 리스트를 받아 같은 길이의 출력 리스트를 반환한다.

        배치를 지원하지 않는 모델이면 내부에서 for 루프로 처리해도 된다.
        이 함수는 블로킹(동기)이어도 되며, 스케줄러가 별도 스레드에서 호출한다.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """그레이스풀 셧다운 시 자원 정리. 선택."""
        pass
