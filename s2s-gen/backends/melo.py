"""MeloTTS 백엔드 — STT→LLM→MeloTTS 다국어 캐스케이드 변형."""
from __future__ import annotations

import importlib.util
import json
import os
import urllib.request
from urllib.error import HTTPError
import wave
from pathlib import Path
from typing import Any, Dict

from .cascade import CascadeBackend
from melo_runtime import prepare_unidic_lite


class MeloBackend(CascadeBackend):
    name = "melo"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)
        self.device_pref = cfg.get("melo_device", cfg.get("device", "auto"))
        self.language = cfg.get("melo_default_language", "EN")
        self.speaker = cfg.get("melo_speaker", "")
        self.speed = float(cfg.get("melo_speed", 1.0))
        self.service_url = (cfg.get("melo_service_url", "") or os.environ.get("MELO_SERVICE_URL", "")).rstrip("/")
        self._models: dict[str, Any] = {}

    def _tts_diagnostics(self) -> list[str]:
        if self.service_url and self._service_reachable():
            return []
        if importlib.util.find_spec("melo") is None:
            return ["MeloTTS 미설치 — lab-ui melo-tts 서비스를 기동하거나 `uv run --with 'melotts @ git+https://github.com/myshell-ai/MeloTTS.git' ...` 로 실행"]
        return []

    def _service_reachable(self) -> bool:
        if not self.service_url:
            return False
        try:
            with urllib.request.urlopen(f"{self.service_url}/readyz", timeout=0.8) as r:
                return r.status < 500
        except Exception:
            return False

    def _melo_language(self) -> str:
        code = (self.cfg.get("language") or "auto").strip()
        if code != "auto":
            entry = (self.cfg.get("languages", {}) or {}).get(code, {}) or {}
            return entry.get("melo_lang") or self.language
        return self.language

    def _synthesize_via_service(self, text: str, out: Path, language: str) -> float:
        body = json.dumps({
            "text": text or "ok",
            "language": language,
            "speaker": self.speaker,
            "speed": self.speed,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.service_url}/synthesize", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out.write_bytes(r.read())
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"melo-tts service HTTP {e.code}: {detail}") from e
        with wave.open(str(out), "rb") as w:
            return w.getnframes() / float(w.getframerate())

    def _get_model(self, language: str):
        if language not in self._models:
            prepare_unidic_lite()
            from melo.api import TTS

            self._models[language] = TTS(language=language, device=self.device_pref)
        return self._models[language]

    @staticmethod
    def _speaker_id(model, language: str, speaker: str):
        speakers = model.hps.data.spk2id
        keys = list(speakers.keys())

        def pick(key: str):
            return speakers[key] if key in keys else None

        if speaker:
            if str(speaker).isdigit():
                return int(speaker)
            return pick(speaker) if pick(speaker) is not None else speakers[keys[0]]
        if language == "EN":
            return pick("EN-Default") if pick("EN-Default") is not None else speakers[keys[0]]
        return pick(language) if pick(language) is not None else speakers[keys[0]]

    def _synthesize(self, text: str, out: Path) -> float:
        language = self._melo_language()
        if self.service_url and self._service_reachable():
            try:
                return self._synthesize_via_service(text, out, language)
            except Exception as e:
                print(f"[melo] service failed; falling back to in-process synthesis: {e}")

        model = self._get_model(language)
        speaker_id = self._speaker_id(model, language, str(self.speaker or ""))
        model.tts_to_file(text or "ok", speaker_id, str(out), speed=self.speed, quiet=True)
        with wave.open(str(out), "rb") as w:
            return w.getnframes() / float(w.getframerate())
