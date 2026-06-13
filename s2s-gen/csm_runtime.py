"""CSM(Sesame csm-1b) 로드·합성 공통 런타임.

CSM 모델을 한 번 로드해 재사용하는 코드를 한 곳에 모은다. 두 소비처가 공유한다:
  - `bento_service.py` (CSMTTS BentoML 서비스) — 모델 1회 로드 후 상주
  - `backends/csm.py` (CLI 백엔드, 서비스 미기동 시 인프로세스 폴백)

CSM 본체(레포 + 가중치)는 무겁고 HF 게이트(라이선스 동의)라 저장소에 포함하지 않는다.
외부 clone 경로(`CSM_DIR`/config `csm_dir`)와 HF 로그인이 필요하다.

준비(요약):
  git clone https://github.com/SesameAILabs/csm   → CSM_DIR
  huggingface-cli login + 게이트 동의: sesame/csm-1b, meta-llama/Llama-3.2-1B
  uv sync --extra csm
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# CSM 워터마커(silentcipher)의 torch.istft 가 쓰는 unfold_backward 는 MPS 미구현 →
# 해당 op만 CPU 폴백. torch import 이전(이 모듈 로드 시점)에 설정해야 가장 안전하다.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def hf_token_present() -> bool:
    """HF 토큰(게이트 가중치 다운로드용)이 있는지 — 환경변수 또는 로컬 캐시."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    home = Path.home()
    return (home / ".cache" / "huggingface" / "token").exists() or (home / ".huggingface" / "token").exists()


def resolve_device(pref: str = "auto") -> str:
    """'auto' → torch 가용 장치(cuda→mps→cpu)로 해석. load_csm_1b 는 구체 장치 문자열을 요구한다."""
    if pref and pref != "auto":
        return pref
    import torch  # type: ignore
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_bnb_stub() -> None:
    """moshi 0.2.2 의 quantize.linear()는 비양자화 경로에서도 `import bitsandbytes`를
    함수 최상단에서 무조건 실행한다. bitsandbytes 는 Apple Silicon 휠이 마땅치 않은데,
    fp(bf16) 가중치에선 실제로 bnb 가 호출되지 않으므로(=is_quantized False) import 만
    통과시키면 된다. 가짜 모듈을 주입하되 실제 양자화 경로가 쓰이면 명확히 실패시킨다.

    주의: catch-all __getattr__ 를 쓰면 torch 의 모듈 introspection(getmodule이 모든
    sys.modules 의 __file__ 접근)이 깨진다. 따라서 진짜 문자열 __file__ 을 주고, moshi 가
    실제로 참조하는 이름(MatmulLtState, matmul)만 명시적으로 정의한다.

    또한 transformers 는 import 시 `importlib.util.find_spec("bitsandbytes")` 를 호출하는데,
    sys.modules 에 모듈이 있고 __spec__ 이 None 이면 `ValueError: __spec__ is None` 을 던진다
    (BentoML 서비스처럼 STT/LLM 보다 먼저 stub 이 주입되는 경로에서 발생). 따라서 진짜
    ModuleSpec(loader=None)을 준다 — find_spec 은 통과하고, transformers 의 metadata.version 조회는
    실제 dist 가 없어 PackageNotFoundError → '미설치'로 처리되어 우리가 원하는 동작이 된다."""
    import importlib.machinery
    import importlib.util
    if importlib.util.find_spec("bitsandbytes") is not None:
        return
    stub = types.ModuleType("bitsandbytes")
    stub.__file__ = "<bitsandbytes-stub>"  # introspection(getsourcefile) 호환용 문자열
    stub.__spec__ = importlib.machinery.ModuleSpec("bitsandbytes", loader=None)

    def _need_bnb(*_a, **_k):
        raise RuntimeError("bitsandbytes가 필요합니다(양자화 CSM 가중치). fp 가중치에선 호출되지 않아야 합니다.")

    stub.MatmulLtState = _need_bnb  # type: ignore[attr-defined]
    stub.matmul = _need_bnb         # type: ignore[attr-defined]
    sys.modules["bitsandbytes"] = stub


def load_generator(csm_dir: str, device: str):
    """CSM Generator 를 로드한다(호출자가 1회 호출 후 캐시). device 는 구체 문자열.

    실패 시 게이트 권한 문제(401/403/Gated)는 친화적 한국어 메시지로 다시 던진다.
    """
    ensure_bnb_stub()
    d = Path(csm_dir).expanduser()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))  # CSM repo(generator.py, models.py 등)를 import 경로에
    try:
        return _load_generator_impl(device)
    except Exception as e:  # noqa: BLE001
        if "Gated" in type(e).__name__ or "401" in str(e) or "403" in str(e):
            raise RuntimeError(
                "sesame/csm-1b 접근 권한이 없습니다(HF 게이트). "
                "https://huggingface.co/sesame/csm-1b 에서 'Agree and access repository'를 누른 뒤, "
                "huggingface-cli login 한 계정과 일치하는지 확인하세요. "
                "meta-llama/Llama-3.2-1B 게이트도 동의해야 합니다."
            ) from e
        raise


def _load_generator_impl(device: str):
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


def synthesize_pcm16(gen, text: str, speaker: int, max_audio_ms: float, out: Path) -> float:
    """CSM 으로 text→음성 합성 후 PCM16 WAV 로 저장. 반환: 응답 음성 길이(초).

    CSM 출력은 float32 텐서 → 저장소 공통 규약(PCM16 WAV)으로 맞춰 브라우저·wave 호환 보장.
    """
    import torch  # type: ignore
    import torchaudio  # type: ignore
    wav = gen.generate(
        text=text or "ok",
        speaker=int(speaker),
        context=[],
        max_audio_length_ms=float(max_audio_ms),
    )
    sr = int(gen.sample_rate)
    pcm = (wav.detach().cpu().clamp(-1.0, 1.0) * 32767.0).to(torch.int16).unsqueeze(0)
    torchaudio.save(str(out), pcm, sr, encoding="PCM_S", bits_per_sample=16)
    return float(wav.shape[-1]) / float(sr)
