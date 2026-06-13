#!/usr/bin/env python3
"""Live Voice — 브라우저 마이크 기반 턴제 음성 대화(STT→LLM→TTS).

lab-ui의 다른 카드는 CLI를 subprocess로 실행하는 얇은 오케스트레이션이지만, Live Voice는
낮은 지연을 위해 모델(faster-whisper·Piper)을 이 프로세스에 캐싱하고 한 요청 안에서
STT→LLM→TTS를 직접 실행한다. LLM은 Ollama 네이티브 /api/chat로 호출해 reasoning(think)을
끌 수 있다.

이 모듈은 비즈니스 로직과 라우트를 모두 담고 있고, app.py는 `router`만 마운트한다
(app.py는 `python app.py`로 실행돼 __main__이 되므로 이 모듈은 app.py를 import하지 않는다 —
필요한 경로/사전점검 헬퍼는 여기서 독립적으로 정의한다).

설계 메모:
- 멀티턴 메모리: 서버가 대화 history를 보관하고 매 턴 system + history 전체를 LLM에 전달.
- 동시성: 로컬 단일 사용자 가정. _LIVE_LOCK으로 턴을 직렬화(모델/ history 충돌 방지).
- 응답 음성은 voice-agent/outputs/live_<turn_id>.wav에 저장 →
  app.py의 /api/artifacts/voice-agent/<file> 라우트로 서빙(path traversal 방어 재사용).
- 무거운 import(numpy/faster_whisper/piper/openai)는 함수 안에서 lazy 처리.
"""
from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

# --- 경로/상수 (app.py와 독립 계산: live_voice.py는 lab-ui 디렉터리에 위치) ---
LAB_UI = Path(__file__).resolve().parent
ROOT = LAB_UI.parent  # 저장소 루트 (multimodal-serving-lab)
UPLOADS = Path(tempfile.gettempdir()) / "multimodal-serving-lab" / "uploads"
VOICE_CFG_PATH = ROOT / "voice-agent" / "config.yaml"
LIVE_OUT = ROOT / "voice-agent" / "outputs"  # /api/artifacts/voice-agent/<file> 로 서빙

# STT 선택지 — stt-gen 카드 allowlist와 동일하게 유지.
STT_MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]
COMPUTE_TYPES = ["int8", "float32", "float16", "int8_float16"]
DEVICES = ["auto", "cpu", "cuda"]

# 지표 표시 정의(키 → 라벨/단위/설명/방향). app.py의 voice-agent 지표와 동일한 형태로 반환.
_METRICS: dict[str, tuple[str, str, str, str]] = {
    "e2e_ms": ("E2E", "ms", "사용자 발화 종료부터 응답 음성 생성까지의 전체 지연입니다.", "낮을수록 좋습니다."),
    "stt_ms": ("STT", "ms", "음성 입력을 텍스트로 바꾸는 단계 시간입니다.", "낮을수록 빠릅니다."),
    "llm_ttft_ms": ("LLM TTFT", "ms", "LLM 첫 토큰 대기 시간입니다.", "낮을수록 대화 반응성이 좋습니다."),
    "llm_total_ms": ("LLM total", "ms", "LLM 응답 생성 전체 시간입니다.", "낮을수록 빠릅니다."),
    "tts_ms": ("TTS", "ms", "응답 텍스트를 음성으로 합성하는 시간입니다.", "낮을수록 빠릅니다."),
    "bottleneck": ("병목 단계", "", "한 턴에서 가장 오래 걸린 단계입니다.", "최적화 우선순위를 보여줍니다."),
    "bottleneck_pct": ("병목 비중", "%", "전체 시간 중 병목 단계가 차지한 비율입니다.", "높을수록 해당 단계 영향이 큽니다."),
    "ttfa_ms": ("TTFA", "ms", "입력 종료부터 첫 응답 오디오까지의 시간입니다.", "낮을수록 대화 반응성이 좋습니다."),
    "rtf": ("RTF", "", "응답 처리 시간 / 응답 음성 길이입니다.", "1보다 작으면 실시간보다 빠릅니다."),
}

# S2S 모드 백엔드 — 파일→파일(턴제)로 동작하는 것만. moshi(full-duplex)는 라이브 전용이라 제외.
S2S_BACKENDS = ["csm"]
PIPELINE_MODES = ["cascade", "s2s"]

# --- 프로세스 내 상태(모델 캐시 + 대화 메모리) ---
_STT_CACHE: dict[tuple, Any] = {}  # (model, compute_type, device) -> WhisperModel
_TTS_CACHE: dict[str, Any] = {}    # voice name -> PiperVoice
_MODEL_LOCK = threading.Lock()     # 모델 로드 직렬화

_LIVE_HISTORY: list[dict] = []     # [{role, content}, ...] 멀티턴 메모리(로컬 단일 세션)
_LIVE_TURN = 0                     # 누적 턴 카운터(turn_id 생성용)
_LIVE_LOCK = threading.Lock()      # 턴 직렬화 + history 보호
_LLM_WARMED: set[str] = set()      # 모델별 1회 warmup 여부(콜드 로드 시 빈 응답 방지)


def _label(raw: dict) -> dict:
    """raw 지표({key: value})를 UI 표시용 형태({key: {label,value,unit,help,direction}})로."""
    out = {}
    for k, v in raw.items():
        label, unit, help_, direction = _METRICS.get(k, (k, "", "", ""))
        out[k] = {"label": label, "value": v, "unit": unit, "help": help_, "direction": direction}
    return out


# --- 사전 점검(app.preflight의 동명 헬퍼와 동작 동일 — 모듈 독립성을 위해 별도 정의) ---
def _ollama_up(base: str = "http://localhost:11434") -> bool:
    try:
        return httpx.get(f"{base}/api/tags", timeout=0.6).status_code == 200
    except Exception:
        return False


def _voice_present(voice_dir: Path, voice: str) -> bool:
    # piper voice는 <name>.onnx 형태로 캐시된다.
    return voice_dir.exists() and any(voice_dir.glob(f"{voice}*.onnx"))


def _ollama_models(base: str = "http://localhost:11434") -> list[str]:
    """Ollama에 설치된 모델명 목록(정렬). 미기동/실패 시 빈 리스트."""
    try:
        r = httpx.get(f"{base}/api/tags", timeout=1.5)
        return sorted(m["name"] for m in r.json().get("models", []))
    except Exception:
        return []


def _live_cfg() -> dict:
    return yaml.safe_load(open(VOICE_CFG_PATH, "r", encoding="utf-8"))


def _live_voice_dir(tts_cfg: dict) -> Path:
    # config의 voice_dir은 voice-agent 디렉터리 기준 상대경로("../tts-gen/models").
    return (VOICE_CFG_PATH.parent / tts_cfg.get("voice_dir", "../tts-gen/models")).resolve()


# --- 모델 로드/캐시 ---
def _get_stt(model: str, compute_type: str, device: str):
    key = (model, compute_type, device)
    with _MODEL_LOCK:
        m = _STT_CACHE.get(key)
        if m is None:
            import numpy as np
            from faster_whisper import WhisperModel
            m = WhisperModel(model, device=device, compute_type=compute_type)
            # warmup: 첫 전사는 느리므로 1초 무음으로 한 번 돌려 캐시를 데운다(콜드스타트 제거).
            list(m.transcribe(np.zeros(16000, dtype=np.float32), beam_size=1)[0])
            _STT_CACHE[key] = m
        return m


def _get_voice(voice_dir: Path, voice: str):
    with _MODEL_LOCK:
        v = _TTS_CACHE.get(voice)
        if v is None:
            from piper import PiperVoice
            v = PiperVoice.load(str(voice_dir / f"{voice}.onnx"), use_cuda=False)
            _TTS_CACHE[voice] = v
        return v


def _tts_to_wav(voice, text: str, out: Path) -> float:
    """voice-agent/agent.py:tts_to_wav 로직 재사용 — mono/16-bit/voice sample_rate WAV 기록."""
    import numpy as np
    from piper import SynthesisConfig
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    sr = voice.config.sample_rate
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return len(audio) / sr


def _llm_chat_stream(llm_cfg: dict, messages: list[dict]):
    """LLM 스트리밍 → (answer, ttft_s, total_s).

    기본은 Ollama 네이티브 /api/chat — think 플래그로 reasoning을 끌 수 있어 저지연(thinking
    모델에서 핵심). base_url이 Ollama가 아니면(OpenAI 호환 endpoint) /v1로 폴백한다(think 제어 불가).
    """
    import json
    base = llm_cfg.get("base_url", "http://localhost:11434/v1")
    model = llm_cfg["model"]
    think = bool(llm_cfg.get("think", False))
    max_tokens = int(llm_cfg.get("max_tokens", 256))
    native = (base[:-3] if base.rstrip("/").endswith("/v1") else base).rstrip("/")
    body = {"model": model, "messages": messages, "stream": True, "think": think,
            "options": {"num_predict": max_tokens, "temperature": 0.5}}
    t0 = time.perf_counter()
    ttft = None
    answer = ""
    try:
        with httpx.stream("POST", native + "/api/chat", json=body, timeout=120.0) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                c = json.loads(line).get("message", {}).get("content", "")
                if c:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    answer += c
        return answer.strip(), ttft or 0.0, time.perf_counter() - t0
    except Exception:
        # 비-Ollama OpenAI 호환 endpoint 폴백 (예: vLLM) — thinking 제어는 불가.
        from openai import OpenAI
        client = OpenAI(base_url=base, api_key=llm_cfg.get("api_key", "EMPTY"))
        t0 = time.perf_counter()
        ttft = None
        answer = ""
        stream = client.chat.completions.create(
            model=model, messages=messages, stream=True, temperature=0.5, max_tokens=max_tokens)
        for ch in stream:
            d = ch.choices[0].delta.content
            if d:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                answer += d
        return answer.strip(), ttft or 0.0, time.perf_counter() - t0


# --- 모델 서비스(BentoML) 클라이언트 — app.py SERVICES 포트와 일치 유지 ---
# 저지연 서빙: 모델을 상주 서비스로 띄워두고 HTTP 로 호출한다. csm-tts 는 특히 로드가 무거워
# (매 턴 재로드 불가) 상주 서비스가 핵심. 경량 모델(STT/Piper)은 서비스화 시 HTTP 오버헤드로
# 인프로세스 대비 지연이 늘 수 있어 toggle(serving_mode)로 둘을 실측 비교한다.
SVC_STT_URL = "http://127.0.0.1:3001"   # whisper-stt
SVC_TTS_URL = "http://127.0.0.1:3002"   # piper-tts
SVC_CSM_URL = "http://127.0.0.1:3003"   # csm-tts (표현형 TTS)
SERVING_MODES = ["in_process", "served"]


def _svc_up(url: str) -> bool:
    """BentoML /readyz — 모델 로딩이 끝나 요청 가능한가(CSM 은 로드가 길다)."""
    try:
        return httpx.get(f"{url}/readyz", timeout=0.8).status_code < 500
    except Exception:
        return False


def _stt_inproc(in_path: Path, model: str, compute: str, device: str) -> str:
    stt = _get_stt(model, compute, device)
    segs, _ = stt.transcribe(str(in_path), beam_size=1)
    return "".join(s.text for s in segs).strip()


def _stt_via_svc(in_path: Path) -> str:
    """whisper-stt 서비스로 전사(멀티파트 wav 업로드)."""
    with open(in_path, "rb") as f:
        r = httpx.post(f"{SVC_STT_URL}/transcribe",
                       files={"audio": (in_path.name, f, "audio/wav")}, timeout=120.0)
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def _tts_via_svc(url: str, text: str, out: Path) -> float:
    """piper-tts/csm-tts 서비스로 합성(JSON text→wav 바이트). 반환: 응답 음성 길이(초)."""
    r = httpx.post(f"{url}/synthesize", json={"text": text or "..."}, timeout=600.0)
    r.raise_for_status()
    out.write_bytes(r.content)
    with wave.open(str(out), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def _run_turn(in_path: Path, stt_fn, llm_cfg: dict, tts_fn, tts_label: str = "TTS"):
    """한 대화 턴: STT → LLM(멀티턴) → TTS. _LIVE_LOCK 보유 상태에서만 호출.

    STT/TTS 는 콜러블로 주입한다 — 인프로세스(faster-whisper/Piper) 또는 BentoML 서비스 호출을
    같은 파이프라인·메트릭·history 로 처리하기 위함. csm 경로는 tts_fn 만 csm-tts 서비스로 바뀐다.
      stt_fn(in_path) -> transcript(str)
      tts_fn(text, out_wav) -> 응답 음성 길이(초)
    """
    global _LIVE_TURN
    budget: dict[str, float] = {}

    # 입력 음성 길이
    try:
        with wave.open(str(in_path), "rb") as w:
            in_dur = w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        in_dur = 0.0

    # 1) STT (인프로세스 또는 whisper-stt 서비스)
    t0 = time.perf_counter()
    transcript = stt_fn(in_path)
    budget["stt_s"] = time.perf_counter() - t0

    if not transcript:
        return {
            "turn_id": None, "transcript": "", "answer": None, "audio_url": None,
            "input_audio_s": round(in_dur, 2), "output_audio_s": None,
            "note": "전사 결과가 비어 있습니다. 더 또렷하게 다시 말해 주세요.",
            "metrics": _label({"stt_ms": round(budget["stt_s"] * 1000)}),
        }

    # 2) LLM (멀티턴 — 서버 history에 누적 후 system + history 전체 전달)
    sys_prompt = llm_cfg.get("system", "You are a concise, friendly senior software developer. Answer software and programming questions in 1-2 short sentences.")
    # warmup: 모델 콜드 로드(첫 턴) 비용을 흡수해 이후 턴의 정상상태 지연을 측정한다.
    model_name = llm_cfg["model"]
    if model_name not in _LLM_WARMED:
        try:
            _llm_chat_stream(llm_cfg, [{"role": "user", "content": "hi"}])
        except Exception:
            pass
        _LLM_WARMED.add(model_name)
    _LIVE_HISTORY.append({"role": "user", "content": transcript})
    messages = [{"role": "system", "content": sys_prompt}, *_LIVE_HISTORY]
    try:
        answer, ttft, llm_total = _llm_chat_stream(llm_cfg, messages)
    except Exception:
        _LIVE_HISTORY.pop()  # user 메시지 롤백 — history 오염 방지
        raise
    budget["llm_ttft_s"] = ttft
    budget["llm_total_s"] = llm_total
    if not answer:
        _LIVE_HISTORY.pop()  # 유효한 assistant 턴이 없으므로 user 메시지도 롤백
        return {
            "turn_id": None, "transcript": transcript, "answer": "", "audio_url": None,
            "input_audio_s": round(in_dur, 2), "output_audio_s": None,
            "note": "LLM이 빈 응답을 반환했습니다. 다시 시도해 주세요.",
            "metrics": _label({
                "stt_ms": round(budget["stt_s"] * 1000),
                "llm_total_ms": round(budget["llm_total_s"] * 1000),
            }),
        }
    _LIVE_HISTORY.append({"role": "assistant", "content": answer})

    # 3) TTS (인프로세스 Piper, piper-tts 서비스, 또는 csm-tts 서비스 — tts_fn 으로 주입)
    _LIVE_TURN += 1
    turn_id = _LIVE_TURN
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    out_wav = LIVE_OUT / f"live_{turn_id}.wav"
    t0 = time.perf_counter()
    out_dur = tts_fn(answer or "...", out_wav)
    budget["tts_s"] = time.perf_counter() - t0

    total = budget["stt_s"] + budget["llm_total_s"] + budget["tts_s"]
    stage = max([("STT", budget["stt_s"]), ("LLM", budget["llm_total_s"]), (tts_label, budget["tts_s"])],
                key=lambda x: x[1])
    raw = {
        "e2e_ms": round(total * 1000),
        "stt_ms": round(budget["stt_s"] * 1000),
        "llm_ttft_ms": round(budget["llm_ttft_s"] * 1000),
        "llm_total_ms": round(budget["llm_total_s"] * 1000),
        "tts_ms": round(budget["tts_s"] * 1000),
        "rtf": round(total / out_dur, 3) if out_dur else 0,
        "bottleneck": stage[0],
        "bottleneck_pct": round(stage[1] / total * 100) if total else 0,
    }
    return {
        "turn_id": turn_id,
        "transcript": transcript,
        "answer": answer,
        "audio_url": f"/api/artifacts/voice-agent/live_{turn_id}.wav",
        "input_audio_s": round(in_dur, 2),
        "output_audio_s": round(out_dur, 2),
        "metrics": _label(raw),
    }


# ---------------------------------------------------------------------------
# 라우트 — app.py에서 app.include_router(live_voice.router)로 마운트.
# ---------------------------------------------------------------------------
router = APIRouter()


@router.get("/api/voice/live/options")
def options():
    """Live Voice 카드 컨트롤 정의(설명 포함) + 현재 세션 정보.

    각 control은 프런트의 paramInput()이 그대로 렌더할 수 있는 형태이고,
    name은 /api/voice/live 의 Form 필드명과 일치한다(model/compute_type/device/voice/llm_model).
    """
    cfg = {}
    try:
        cfg = _live_cfg()
    except Exception:
        pass
    stt_cfg = cfg.get("stt", {})
    tts_cfg = cfg.get("tts", {})
    llm_cfg = cfg.get("llm", {})
    vdir = _live_voice_dir(tts_cfg)
    voices = sorted(p.stem for p in vdir.glob("*.onnx")) if vdir.exists() else []

    llm_default = llm_cfg.get("model", "")
    llm_models = _ollama_models()
    if llm_default and llm_default not in llm_models:  # 설치 목록을 못 받아도 현재 모델은 노출
        llm_models = [llm_default, *llm_models]

    def ctl(name, choices, default, help_, impact):
        return {"name": name, "type": "choice", "choices": choices,
                "default": default, "help": help_, "impact": impact}

    controls = [
        ctl("pipeline_mode", PIPELINE_MODES, "cascade",
            "음성 응답 파이프라인 방식입니다.",
            "cascade는 STT→LLM→TTS(Piper). s2s는 표현형 TTS(아래 backend)로 응답합니다. 어느 쪽이든 LLM은 Ollama입니다."),
        ctl("serving_mode", SERVING_MODES, "in_process",
            "모델을 어디서 실행할지 — 인프로세스 vs 상주 BentoML 서비스.",
            "in_process는 STT/Piper를 이 프로세스에 캐싱(경량 모델엔 최속). served는 whisper-stt/piper-tts 서비스로 HTTP 호출(오버헤드로 더 느릴 수 있음 — 실측 비교용). s2s(csm)는 TTS가 항상 csm-tts 서비스이며, '모델 서비스' 카드에서 먼저 기동해야 합니다."),
        ctl("s2s_backend", S2S_BACKENDS, "csm",
            "s2s 모드에서 쓸 표현형 TTS 모델입니다.",
            "csm(Sesame)은 로드가 무거워 csm-tts 상주 서비스로 동작합니다. '모델 서비스' 카드에서 csm-tts를 기동하세요(외부 가중치 필요). cascade 모드에서는 무시됩니다. moshi(풀듀플렉스)는 'Moshi 라이브' 카드를 쓰세요."),
        ctl("model", STT_MODELS, stt_cfg.get("model", "base.en"),
            "faster-whisper STT 모델 크기입니다.",
            "큰 모델은 정확도가 좋아질 수 있지만 다운로드, 메모리, 전사 시간이 늘어납니다."),
        ctl("compute_type", COMPUTE_TYPES, stt_cfg.get("compute_type", "int8"),
            "STT 모델 계산 정밀도/양자화 방식입니다.",
            "int8은 CPU에서 가볍고, float16/int8_float16은 CUDA 환경에 적합합니다."),
        ctl("device", DEVICES, stt_cfg.get("device", "auto"),
            "STT 실행 장치입니다.",
            "auto는 가능한 장치를 고르고, cpu/cuda는 명시적으로 고정합니다."),
        ctl("voice", voices or [tts_cfg.get("voice", "en_US-lessac-medium")],
            tts_cfg.get("voice", "en_US-lessac-medium"),
            "응답 음성을 합성할 Piper 보이스입니다.",
            "보이스마다 말투·발음·샘플레이트가 다릅니다. TTS 카드에서 받은 보이스만 보입니다."),
        ctl("llm_model", llm_models or [llm_default], llm_default,
            "응답을 생성하는 LLM입니다(Ollama 설치 모델).",
            "큰 모델은 품질이 좋아질 수 있지만 응답이 느려집니다. reasoning 모델은 thinking 때문에 더 느릴 수 있습니다."),
        {  # 자유 텍스트 — 프런트는 type:"str"을 textarea로 렌더(전체 폭).
            "name": "system", "type": "str",
            "default": llm_cfg.get("system", ""),
            "help": "LLM의 역할·말투를 정하는 시스템 프롬프트입니다.",
            "impact": "비우면 config 기본값을 사용합니다. 음성 응답이라 1-2문장으로 짧게 답하도록 두는 것이 좋습니다.",
        },
    ]
    return {"controls": controls, "turns": len(_LIVE_HISTORY) // 2}


@router.post("/api/voice/live/reset")
def reset():
    with _LIVE_LOCK:
        _LIVE_HISTORY.clear()
    return {"status": "cleared"}


@router.post("/api/voice/live")
def turn(
    audio: UploadFile = File(...),
    model: Optional[str] = Form(None),
    compute_type: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    llm_model: Optional[str] = Form(None),
    system: Optional[str] = Form(None),
    pipeline_mode: Optional[str] = Form(None),
    s2s_backend: Optional[str] = Form(None),
    serving_mode: Optional[str] = Form(None),
):
    # 동기 def → FastAPI가 스레드풀에서 실행(블로킹 파이프라인이 이벤트 루프를 막지 않음).
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix != ".wav":
        return JSONResponse({"error": "Live Voice 입력은 wav 파일만 지원합니다."}, status_code=400)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    in_path = UPLOADS / f"live_{uuid.uuid4().hex}.wav"
    try:
        with open(in_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        return JSONResponse({"error": f"업로드 저장 실패: {e}"}, status_code=400)
    finally:
        audio.file.close()

    try:
        cfg = _live_cfg()
    except Exception as e:
        return JSONResponse({"error": f"voice-agent/config.yaml 로드 실패: {e}"}, status_code=500)
    stt_cfg, llm_cfg, tts_cfg = cfg.get("stt", {}), cfg.get("llm", {}), cfg.get("tts", {})

    mode = pipeline_mode or "cascade"        # cascade | s2s
    serving = serving_mode or "in_process"   # in_process | served
    if llm_model:  # UI에서 고른 LLM 모델로 그 턴만 override(base_url/think는 config 유지)
        llm_cfg = {**llm_cfg, "model": llm_model}
    if system and system.strip():  # UI에서 편집한 시스템 프롬프트로 override(비우면 config 기본값)
        llm_cfg = {**llm_cfg, "system": system}
    stt_model = model or stt_cfg.get("model", "base.en")
    stt_compute = compute_type or stt_cfg.get("compute_type", "int8")
    stt_device = device or stt_cfg.get("device", "auto")
    voice_name = voice or tts_cfg.get("voice", "en_US-lessac-medium")
    voice_dir = _live_voice_dir(tts_cfg)

    # 공통 사전 점검 — LLM(Ollama)
    if not _ollama_up():
        return JSONResponse(
            {"error": f"Ollama 미기동 — `ollama serve` 후 모델을 pull 하세요 (예: ollama pull {llm_cfg.get('model', '')})."},
            status_code=503)

    # STT 경로 선택(인프로세스 캐시 vs whisper-stt 서비스)
    if serving == "served":
        if not _svc_up(SVC_STT_URL):
            return JSONResponse(
                {"error": "whisper-stt 서비스 미기동 — '모델 서비스' 카드에서 whisper-stt 를 기동하거나 serving_mode 를 in_process 로 두세요."},
                status_code=503)
        stt_fn = _stt_via_svc
    else:
        stt_fn = lambda p: _stt_inproc(p, stt_model, stt_compute, stt_device)  # noqa: E731

    # TTS 경로 선택
    if mode == "s2s":
        # csm 은 로드가 무거워 항상 csm-tts 상주 서비스로 합성(매 턴 재로드 회피 = 저지연의 핵심).
        if not _svc_up(SVC_CSM_URL):
            return JSONResponse(
                {"error": "csm-tts 서비스 미기동 — '모델 서비스' 카드에서 csm-tts 를 기동하세요(외부 CSM 가중치 필요, 첫 기동 시 로딩에 수십 초)."},
                status_code=503)
        tts_fn = lambda text, out: _tts_via_svc(SVC_CSM_URL, text, out)  # noqa: E731
        tts_label = "CSM"
    elif serving == "served":
        if not _svc_up(SVC_TTS_URL):
            return JSONResponse(
                {"error": "piper-tts 서비스 미기동 — '모델 서비스' 카드에서 piper-tts 를 기동하거나 serving_mode 를 in_process 로 두세요."},
                status_code=503)
        tts_fn = lambda text, out: _tts_via_svc(SVC_TTS_URL, text, out)  # noqa: E731
        tts_label = "TTS"
    else:
        if not _voice_present(voice_dir, voice_name):
            return JSONResponse(
                {"error": f"Piper voice '{voice_name}' 없음 — TTS 카드에서 '보이스 다운로드'(--download)를 먼저 실행하세요."},
                status_code=503)
        tts_fn = lambda text, out: _tts_to_wav(_get_voice(voice_dir, voice_name), text, out)  # noqa: E731
        tts_label = "TTS"

    # 턴은 한 번에 하나씩(모델/대화 history 충돌 방지 — 로컬 단일 사용자 가정)
    with _LIVE_LOCK:
        try:
            return _run_turn(in_path, stt_fn, llm_cfg, tts_fn, tts_label)
        except Exception as e:
            return JSONResponse({"error": f"처리 실패: {type(e).__name__}: {e}"}, status_code=500)
