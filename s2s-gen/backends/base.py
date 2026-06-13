"""S2S(speech-to-speech) 백엔드 인터페이스.

음성-대-음성은 '입력 음성 → (네이티브 음성 모델) → 출력 음성'을 한 모델/한 단계로 처리하는
방식이다. 캐스케이드(STT→LLM→TTS)와 대비되며, 단계 경계가 없어 지연이 낮은 대신 단계별
관찰가능성(추적 로그)이 약하다.

모델 본체(Moshi/CSM 등)는 가중치·의존성이 무거우므로 백엔드로 분리해 교체 가능하게 둔다.
환경(Mac/MPS·NVIDIA)에 맞는 백엔드를 config 에서 고른다. cascade 백엔드는 무거운 모델 없이도
항상 동작하는 기준선이다.
"""
from __future__ import annotations

import abc
from pathlib import Path


class S2SBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(self, in_wav: str, out_wav: str) -> dict:
        """입력 음성(wav) → 응답 음성(wav). 단계별/종단 지연을 담은 metrics dict 반환.

        반환 dict 권장 키:
          - ttfa_s : time-to-first-audio (첫 응답 오디오까지)
          - e2e_s  : 종단 지연(입력 종료→응답 음성 생성)
          - out_s  : 생성된 응답 음성 길이(초)
          - rtf    : e2e_s / out_s (작을수록 빠름)
          - text   : (가능하면) 인식/응답 텍스트. S2S라 없을 수도 있다.
        """
        raise NotImplementedError

    def available(self) -> bool:
        """이 백엔드를 현재 환경에서 실행할 수 있는지(의존성/가중치 점검)."""
        return True

    def diagnostics(self) -> list[str]:
        """사용 불가할 때 사람이 바로 고칠 수 있는 점검 메시지."""
        return []
