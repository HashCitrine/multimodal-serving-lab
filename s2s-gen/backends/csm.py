"""Sesame CSM 백엔드 — STT→LLM→CSM(표현형 TTS) 캐스케이드 변형.

CSM(`sesame/csm-1b`)은 단독 speech-to-speech가 아니라 conversational **text-to-speech**다
(`load_csm_1b(device)` → `generate(text, speaker, context) -> waveform`, Apple Silicon MPS 지원).
따라서 cascade 의 STT+LLM 은 그대로 쓰고 **TTS 단계만 CSM 으로 교체**한다 — Piper(cascade) 대비
운율·표현력 비교가 목적이다.

CSM 본체(레포 + 가중치)는 무겁고 HF 게이트(라이선스 동의 필요)라 저장소에 포함하지 않는다.
외부 clone 경로(`CSM_DIR`)와 HF 로그인이 필요하다(README). 가중치는 첫 실행 시 HF 에서 받는다.
준비 안 됐으면 diagnostics() 로 안내하고 generate() 는 명확히 실패한다.

준비(요약):
  git clone https://github.com/SesameAILabs/csm   → CSM_DIR (또는 config csm_dir)
  huggingface-cli login + 게이트 동의: sesame/csm-1b, meta-llama/Llama-3.2-1B
  uv sync --extra csm  (torch/torchaudio) + uv pip install -r $CSM_DIR/requirements.txt
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict

from .cascade import CascadeBackend

# CSM 워터마커(silentcipher)의 torch.istft 가 쓰는 unfold_backward 는 MPS 미구현 →
# 해당 op만 CPU 폴백. torch import 이전(이 모듈 로드 시점)에 설정해야 가장 안전하다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = Path.home()
    return (home / ".cache" / "huggingface" / "token").exists() or (home / ".huggingface" / "token").exists()


class CSMBackend(CascadeBackend):
    name = "csm"

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)  # STT/LLM 설정(stt/llm 서브딕트) 재사용
        self.csm_dir_value = cfg.get("csm_dir", "") or os.environ.get("CSM_DIR", "")
        self.device_pref = cfg.get("device", "auto")
        self.speaker = int(cfg.get("csm_speaker", cfg.get("speaker", 0)))
        self.max_audio_ms = float(cfg.get("csm_max_audio_ms", cfg.get("max_audio_ms", 20000)))
        self._gen = None

    def _resolve_device(self) -> str:
        """'auto' → torch 가용 장치(cuda→mps→cpu)로 해석. load_csm_1b 는 구체 장치 문자열을 요구한다."""
        if self.device_pref != "auto":
            return self.device_pref
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _csm_dir(self) -> Path:
        if not self.csm_dir_value:
            return Path()
        p = Path(self.csm_dir_value).expanduser()
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    # cascade 의 TTS 점검을 CSM 점검으로 교체(Piper 보이스는 불필요).
    def _tts_diagnostics(self) -> list[str]:
        issues = []
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("torchaudio") is None:
            issues.append("torch/torchaudio 미설치 — `uv sync --extra csm`")
        d = self._csm_dir()
        if not self.csm_dir_value:
            issues.append("CSM repo가 설정되지 않음 (set csm_dir or CSM_DIR; git clone SesameAILabs/csm)")
        elif not (d / "generator.py").exists():
            issues.append(f"generator.py not found under CSM repo: {d}")
        if not _hf_token_present():
            issues.append("HF 토큰 없음 — `huggingface-cli login` + 게이트 동의(sesame/csm-1b, meta-llama/Llama-3.2-1B)")
        return issues

    @staticmethod
    def _ensure_bnb_stub() -> None:
        """moshi 0.2.2 의 quantize.linear()는 비양자화 경로에서도 `import bitsandbytes`를
        함수 최상단에서 무조건 실행한다. bitsandbytes 는 Apple Silicon 휠이 마땅치 않은데,
        fp(bf16) 가중치에선 실제로 bnb 가 호출되지 않으므로(=is_quantized False) import 만
        통과시키면 된다. 가짜 모듈을 주입하되 실제 양자화 경로가 쓰이면 명확히 실패시킨다.

        주의: catch-all __getattr__ 를 쓰면 torch 의 모듈 introspection(getmodule이 모든
        sys.modules 의 __file__ 접근)이 깨진다. 따라서 진짜 문자열 __file__ 을 주고, moshi 가
        실제로 참조하는 이름(MatmulLtState, matmul)만 명시적으로 정의한다."""
        if importlib.util.find_spec("bitsandbytes") is not None:
            return
        import types
        stub = types.ModuleType("bitsandbytes")
        stub.__file__ = "<bitsandbytes-stub>"  # introspection(getsourcefile) 호환용 문자열
        stub.__spec__ = None
        def _need_bnb(*_a, **_k):
            raise RuntimeError("bitsandbytes가 필요합니다(양자화 CSM 가중치). fp 가중치에선 호출되지 않아야 합니다.")
        stub.MatmulLtState = _need_bnb  # type: ignore[attr-defined]
        stub.matmul = _need_bnb         # type: ignore[attr-defined]
        sys.modules["bitsandbytes"] = stub

    def _get_gen(self):
        if self._gen is None:
            # CSM 워터마커(silentcipher)의 torch.istft 는 MPS 미구현 op(unfold_backward)를 쓴다.
            # 해당 op만 CPU로 폴백시켜 MPS 생성을 유지한다(env는 op 디스패치 시점에 읽힌다).
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            d = self._csm_dir()
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))  # CSM repo(generator.py, models.py 등)를 import 경로에
            self._ensure_bnb_stub()
            device = self._resolve_device()
            try:
                self._gen = self._load_generator(device)
            except Exception as e:
                if "Gated" in type(e).__name__ or "401" in str(e) or "403" in str(e):
                    raise RuntimeError(
                        "sesame/csm-1b 접근 권한이 없습니다(HF 게이트). "
                        "https://huggingface.co/sesame/csm-1b 에서 'Agree and access repository'를 누른 뒤, "
                        "huggingface-cli login 한 계정과 일치하는지 확인하세요. "
                        "meta-llama/Llama-3.2-1B 게이트도 동의해야 합니다."
                    ) from e
                raise
        return self._gen

    def _load_generator(self, device: str):
        try:
            from generator import load_csm_1b  # type: ignore  (CSM repo 제공)
            return load_csm_1b(device=device)
        except TypeError:
            # 현재 sesame/csm-1b 는 config.json 이 transformers 형식이라, 구버전 csm 의
            # PyTorchModelHubMixin.from_pretrained 가 ModelArgs config 주입에 실패한다
            # (Model.__init__() missing 'config'). ckpt.pt + 표준 ModelArgs 로 직접 구성해
            # 우회한다(원래 csm 로딩과 동치). Generator 가 mimi/토크나이저/watermarker 를 채운다.
            import torch  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from models import Model, ModelArgs  # type: ignore
            from generator import Generator  # type: ignore
            args = ModelArgs(backbone_flavor="llama-1B", decoder_flavor="llama-100M",
                             text_vocab_size=128256, audio_vocab_size=2051, audio_num_codebooks=32)
            model = Model(args).to(device=device, dtype=torch.bfloat16)
            state = torch.load(hf_hub_download("sesame/csm-1b", "ckpt.pt"), map_location=device)
            model.load_state_dict(state)
            return Generator(model)

    # cascade 의 _synthesize(Piper)를 CSM 으로 교체. 반환: 응답 음성 길이(초).
    def _synthesize(self, text: str, out: Path) -> float:
        import torch  # type: ignore
        import torchaudio  # type: ignore
        gen = self._get_gen()
        wav = gen.generate(
            text=text or "ok",
            speaker=self.speaker,
            context=[],
            max_audio_length_ms=self.max_audio_ms,
        )
        sr = int(gen.sample_rate)
        # CSM 출력은 float32 텐서 → 저장소 공통 규약(PCM16 WAV)으로 맞춰 브라우저·wave 호환 보장.
        pcm = (wav.detach().cpu().clamp(-1.0, 1.0) * 32767.0).to(torch.int16).unsqueeze(0)
        torchaudio.save(str(out), pcm, sr, encoding="PCM_S", bits_per_sample=16)
        return float(wav.shape[-1]) / float(sr)
