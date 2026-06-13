"""Cascade 백엔드 — STT→LLM→TTS 기준선.

앞선 voice-agent 자산(faster-whisper + OpenAI 호환 LLM + Piper)을 그대로 묶어, S2S 모델과
'같은 인터페이스(in_wav→out_wav+metrics)'로 비교할 수 있게 만든 기준선이다. 무거운 음성
네이티브 모델 가중치 없이 어디서나(Mac/CPU) 동작한다. 단계 경계가 살아 있어 관찰가능성이
높은 대신, 종단 지연은 S2S보다 크다.

cascade 는 종단 지연(e2e_s)을 STT+LLM+TTS 합으로 보고, TTFA(첫 응답 오디오)는 TTS가 끝나야
첫 오디오가 나오므로 e2e_s 와 동일하게 둔다(스트리밍 미적용 기준선).
"""
from __future__ import annotations

import shutil
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .base import S2SBackend


class CascadeBackend(S2SBackend):
    name = "cascade"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.base_dir = Path(__file__).resolve().parents[1]
        self._stt = None
        self._voice = None
        self._client = None

    # --- 점검 -------------------------------------------------------------
    def available(self) -> bool:
        return not self.diagnostics()

    def diagnostics(self) -> list[str]:
        issues = []
        if shutil.which("ffmpeg") is None:
            # faster-whisper 가 일부 입력 디코딩에 ffmpeg 를 쓴다(STT 공통).
            issues.append("ffmpeg not found on PATH")
        issues.extend(self._tts_diagnostics())  # TTS 단계 점검은 하위 클래스가 교체(csv: CSM)
        return issues

    def _tts_diagnostics(self) -> list[str]:
        """TTS 단계 준비 점검. cascade 는 Piper 보이스, csm 은 torch/CSM_DIR/HF 토큰으로 override."""
        voice = self._voice_path()
        if not voice.exists():
            return [f"Piper voice not found: {voice} (tts-gen 보이스 준비 필요)"]
        return []

    # --- 리소스 -----------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        p = Path(rel).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def _voice_path(self) -> Path:
        tts = self.cfg.get("tts", {})
        return self._resolve(tts.get("voice_dir", "../tts-gen/models")) / f"{tts.get('voice', 'en_US-lessac-medium')}.onnx"

    def _get_voice(self):
        if self._voice is None:
            from piper import PiperVoice
            self._voice = PiperVoice.load(str(self._voice_path()))
        return self._voice

    def _get_stt(self):
        if self._stt is None:
            from faster_whisper import WhisperModel
            stt = self.cfg.get("stt", {})
            self._stt = WhisperModel(stt.get("model", "base.en"),
                                     device=stt.get("device", "auto"),
                                     compute_type=stt.get("compute_type", "int8"))
        return self._stt

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            prov = self.cfg.get("llm", {})
            self._client = OpenAI(base_url=prov.get("base_url", "http://localhost:11434/v1"),
                                  api_key=prov.get("api_key", "EMPTY"))
        return self._client

    def _synthesize(self, text: str, out: Path) -> float:
        """응답 텍스트 → 음성 wav. 반환: 응답 음성 길이(초). 하위 클래스(csm)가 override."""
        return self._tts_to_wav(text, out)

    def _tts_to_wav(self, text: str, out: Path) -> float:
        from piper import SynthesisConfig
        voice = self._get_voice()
        audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text or "ok", SynthesisConfig())])
        sr = voice.config.sample_rate
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
        return len(audio) / sr

    # --- 실행 -------------------------------------------------------------
    def generate(self, in_wav: str, out_wav: str) -> dict:
        if not self.available():
            detail = "\n".join(f"- {i}" for i in self.diagnostics())
            raise RuntimeError(f"cascade backend not ready:\n{detail}")
        prov = self.cfg.get("llm", {})
        sys_prompt = prov.get("system", "You are a concise, friendly tutor. Answer in 1-2 short sentences.")

        # 1) STT
        stt = self._get_stt()
        t0 = time.perf_counter()
        segs, _ = stt.transcribe(in_wav, beam_size=1)
        question = "".join(s.text for s in segs).strip()
        stt_s = time.perf_counter() - t0

        # 2) LLM (스트리밍, TTFT 측정)
        client = self._get_client()
        t0 = time.perf_counter(); ttft = None; answer = ""
        stream = client.chat.completions.create(
            model=prov.get("model", "llama3.2:1b-instruct-q4_K_M"),
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": question}],
            stream=True, temperature=0.5, max_tokens=prov.get("max_tokens", 80))
        for ch in stream:
            d = ch.choices[0].delta.content
            if d:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                answer += d
        llm_s = time.perf_counter() - t0

        # 3) TTS (cascade=Piper, csm=CSM — _synthesize 가 교체점)
        t0 = time.perf_counter()
        out_s = self._synthesize(answer.strip(), Path(out_wav))
        tts_s = time.perf_counter() - t0

        e2e_s = stt_s + llm_s + tts_s
        return {
            "backend": self.name,
            "text": answer.strip(),
            "question": question,
            "stt_s": stt_s,
            "llm_ttft_s": ttft or 0.0,
            "llm_s": llm_s,
            "tts_s": tts_s,
            "ttfa_s": e2e_s,           # 비스트리밍 기준선: 첫 오디오는 TTS 끝나야 나온다
            "e2e_s": e2e_s,
            "out_s": out_s,
            "rtf": (e2e_s / out_s) if out_s else 0.0,
        }
