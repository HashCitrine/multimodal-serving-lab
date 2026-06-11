"""립싱크 백엔드 인터페이스.

아바타(토킹헤드) 생성은 '얼굴 이미지 + 음성 → 입모양이 맞는 영상' 단계다. 실제 모델
(Wav2Lip / SadTalker / MuseTalk 등)은 의존성·체크포인트가 무거우므로 백엔드로 분리해
교체 가능하게 둔다. 환경(Mac/NVIDIA)에 맞는 백엔드를 config 에서 고른다.
"""
from __future__ import annotations

import abc
from pathlib import Path


class LipSyncBackend(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def generate(self, face_path: str, audio_path: str, out_path: str) -> Path:
        """얼굴 이미지(또는 영상) + 오디오 → 립싱크 mp4 경로 반환."""
        raise NotImplementedError

    def available(self) -> bool:
        """이 백엔드를 현재 환경에서 실행할 수 있는지(의존성/체크포인트 점검)."""
        return True

    def diagnostics(self) -> list[str]:
        """사용 불가할 때 사람이 바로 고칠 수 있는 점검 메시지."""
        return []
