"""Sesame CSM 백엔드 — STT→LLM→CSM(표현형 TTS) 캐스케이드 변형.

CSM(`sesame/csm-1b`)은 단독 speech-to-speech가 아니라 conversational **text-to-speech**다
(`generate(text, speaker, context) -> waveform`, Apple Silicon MPS 지원). 따라서 cascade 의
STT+LLM 은 그대로 쓰고 **TTS 단계만 CSM 으로 교체**한다 — Piper(cascade) 대비 운율·표현력 비교가 목적.

**저지연 서빙**: CSM 은 로드가 무거워 매 호출마다 로드하면 실시간성이 없다. 그래서 `_synthesize` 는
service-aware 다 — `csm_service_url`(BentoML csm-tts 상주 서비스)이 떠 있으면 HTTP 로 합성을
위임(모델 재로드 0), 없으면 인프로세스로 1회 로드·캐시해 단독 동작한다. 로드/합성 로직은
`csm_runtime` 공통 모듈로 bento_service.py 와 공유한다(README).
"""
from __future__ import annotations

import importlib.util
import json
import os
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any, Dict

from .cascade import CascadeBackend

import csm_runtime


class CSMBackend(CascadeBackend):
    name = "csm"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)  # STT/LLM 설정(stt/llm 서브딕트) 재사용
        self.csm_dir_value = cfg.get("csm_dir", "") or os.environ.get("CSM_DIR", "")
        self.device_pref = cfg.get("device", "auto")
        self.speaker = int(cfg.get("csm_speaker", cfg.get("speaker", 0)))
        self.max_audio_ms = float(cfg.get("csm_max_audio_ms", cfg.get("max_audio_ms", 20000)))
        self.service_url = (cfg.get("csm_service_url", "") or os.environ.get("CSM_SERVICE_URL", "")).rstrip("/")
        self._gen = None

    def _csm_dir(self) -> Path:
        if not self.csm_dir_value:
            return Path()
        p = Path(self.csm_dir_value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    # cascade 의 TTS 점검을 CSM 점검으로 교체(Piper 보이스는 불필요).
    def _tts_diagnostics(self) -> list[str]:
        issues = []
        # 서비스가 떠 있으면 인프로세스 torch/CSM repo/HF 토큰이 없어도 합성 가능 → 통과.
        if self.service_url and self._service_reachable():
            return issues
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchaudio") is None:
            issues.append("torch/torchaudio 미설치 — `uv sync --extra csm` (또는 csm-tts 서비스 기동)")
        d = self._csm_dir()
        if not self.csm_dir_value:
            issues.append("CSM repo가 설정되지 않음 (set csm_dir or CSM_DIR; git clone SesameAILabs/csm)")
        elif not (d / "generator.py").exists():
            issues.append(f"generator.py not found under CSM repo: {d}")
        if not csm_runtime.hf_token_present():
            issues.append("HF 토큰 없음 — `huggingface-cli login` + 게이트 동의(sesame/csm-1b, meta-llama/Llama-3.2-1B)")
        return issues

    # --- 상주 서비스(csm-tts) 경유 --------------------------------------------
    def _service_reachable(self) -> bool:
        if not self.service_url:
            return False
        try:
            with urllib.request.urlopen(f"{self.service_url}/readyz", timeout=0.8) as r:
                return r.status < 500
        except Exception:
            return False

    def _synthesize_via_service(self, text: str, out: Path) -> float:
        body = json.dumps({"text": text or "ok", "speaker": self.speaker,
                           "max_audio_ms": self.max_audio_ms}).encode("utf-8")
        req = urllib.request.Request(f"{self.service_url}/synthesize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out.write_bytes(r.read())
        with wave.open(str(out), "rb") as w:
            return w.getnframes() / float(w.getframerate())

    # --- 인프로세스 폴백(1회 로드·캐시) ---------------------------------------
    def _get_gen(self):
        if self._gen is None:
            device = csm_runtime.resolve_device(self.device_pref)
            self._gen = csm_runtime.load_generator(str(self._csm_dir()), device)
        return self._gen

    # cascade 의 _synthesize(Piper)를 CSM 으로 교체. 반환: 응답 음성 길이(초).
    def _synthesize(self, text: str, out: Path) -> float:
        if self.service_url and self._service_reachable():
            return self._synthesize_via_service(text, out)
        gen = self._get_gen()
        return csm_runtime.synthesize_pcm16(gen, text, self.speaker, self.max_audio_ms, out)
