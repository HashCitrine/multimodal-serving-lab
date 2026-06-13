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
S2S_GEN_DIR = ROOT / "s2s-gen"

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


def _extract_s2s_error(out: str) -> str:
    """s2s.py 실패 출력에서 사람이 읽을 한 줄 메시지를 뽑는다(미준비 항목/RuntimeError)."""
    import re
    m = re.search(r"RuntimeError:\s*(.+)", out)
    if m:
        return m.group(1).strip()
    items = re.findall(r"^\s*-\s*(.+)$", out, re.M)  # "백엔드가 아직 준비되지 않았습니다" 불릿
    if items:
        return " / ".join(i.strip() for i in items[:4])
    tail = [l for l in out.splitlines() if l.strip()][-1:]
    return tail[0].strip() if tail else "S2S 실행 실패"


def _run_s2s_turn(in_path: Path, s2s_backend: str, device: str, tts_cfg: dict):
    """한 S2S 턴 — s2s-gen CLI 를 그 venv 에서 subprocess 로 실행.

    Live Voice 의 cascade 는 인프로세스(저지연)지만, csm 은 torch/CSM 의존성이 lab-ui venv 가
    아니라 s2s-gen venv(`--extra csm`)에 있다. 따라서 in-process import 대신
    `uv run --extra csm python s2s.py --backend <b> --audio <wav>` 로 올바른 환경에서 실행하고,
    생성된 wav(`s2s-gen/outputs/answer_<b>.wav`)를 가져와 서빙한다. csm_dir 등은 s2s-gen/config.yaml
    이 제공한다. 모델을 매 턴 로드하므로 csm 턴은 느리다(빠른 대화는 cascade 사용). _LIVE_LOCK 보유 중 호출.
    """
    global _LIVE_TURN
    import re
    import shutil
    import subprocess

    cmd = ["uv", "run", "--extra", "csm", "python", "s2s.py",
           "--backend", s2s_backend, "--audio", str(in_path)]
    try:
        r = subprocess.run(cmd, cwd=str(S2S_GEN_DIR), capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {
            "turn_id": None, "transcript": "", "answer": None, "audio_url": None,
            "input_audio_s": None, "output_audio_s": None,
            "note": f"S2S '{s2s_backend}' 처리 시간 초과(>600s). csm 은 무거우니 'S2S' 카드 또는 cascade 를 쓰세요.",
            "metrics": _label({}),
        }
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        return {
            "turn_id": None, "transcript": "", "answer": None, "audio_url": None,
            "input_audio_s": None, "output_audio_s": None,
            "note": f"S2S '{s2s_backend}' 준비/실행 실패 — {_extract_s2s_error(out)}",
            "metrics": _label({}),
        }

    def g(pat):
        m = re.search(pat, out)
        return m.group(1) if m else None

    src = S2S_GEN_DIR / "outputs" / f"answer_{s2s_backend}.wav"
    _LIVE_TURN += 1
    turn_id = _LIVE_TURN
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    audio_url = None
    if src.exists():
        shutil.copyfile(src, LIVE_OUT / f"live_{turn_id}.wav")  # 기존 artifacts 라우트로 서빙
        audio_url = f"/api/artifacts/voice-agent/live_{turn_id}.wav"

    raw = {}
    if g(r"TTFA\s*:\s*([0-9.]+)\s*ms"):
        raw["ttfa_ms"] = round(float(g(r"TTFA\s*:\s*([0-9.]+)\s*ms")))
    if g(r"E2E\s*:\s*([0-9.]+)\s*ms"):
        raw["e2e_ms"] = round(float(g(r"E2E\s*:\s*([0-9.]+)\s*ms")))
    if g(r"RTF\s*:\s*([0-9.]+)"):
        raw["rtf"] = round(float(g(r"RTF\s*:\s*([0-9.]+)")), 3)
    out_s = g(r"\[응답 음성\]\s*([0-9.]+)s")
    return {
        "turn_id": turn_id,
        "transcript": g(r'\[인식\]\s*"(.*?)"') or "(음성 입력)",
        "answer": g(r'\[응답\]\s*"(.*?)"') or f"(CSM 음성 응답 — {s2s_backend})",
        "audio_url": audio_url,
        "input_audio_s": None,
        "output_audio_s": float(out_s) if out_s else None,
        "metrics": _label(raw),
    }


def _run_turn(in_path: Path, stt_model: str, stt_compute: str, stt_device: str,
              voice_dir: Path, voice_name: str, llm_cfg: dict):
    """한 대화 턴: STT → LLM(멀티턴) → TTS. _LIVE_LOCK 보유 상태에서만 호출."""
    global _LIVE_TURN
    budget: dict[str, float] = {}

    # 입력 음성 길이
    try:
        with wave.open(str(in_path), "rb") as w:
            in_dur = w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        in_dur = 0.0

    # 1) STT
    stt = _get_stt(stt_model, stt_compute, stt_device)
    t0 = time.perf_counter()
    segs, _ = stt.transcribe(str(in_path), beam_size=1)
    transcript = "".join(s.text for s in segs).strip()
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

    # 3) TTS
    _LIVE_TURN += 1
    turn_id = _LIVE_TURN
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    out_wav = LIVE_OUT / f"live_{turn_id}.wav"
    voice_obj = _get_voice(voice_dir, voice_name)
    t0 = time.perf_counter()
    out_dur = _tts_to_wav(voice_obj, answer or "...", out_wav)
    budget["tts_s"] = time.perf_counter() - t0

    total = budget["stt_s"] + budget["llm_total_s"] + budget["tts_s"]
    stage = max([("STT", budget["stt_s"]), ("LLM", budget["llm_total_s"]), ("TTS", budget["tts_s"])],
                key=lambda x: x[1])
    raw = {
        "e2e_ms": round(total * 1000),
        "stt_ms": round(budget["stt_s"] * 1000),
        "llm_ttft_ms": round(budget["llm_ttft_s"] * 1000),
        "llm_total_ms": round(budget["llm_total_s"] * 1000),
        "tts_ms": round(budget["tts_s"] * 1000),
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
            "cascade는 STT→LLM→TTS(인프로세스, 빠름·관찰가능). s2s는 음성 네이티브 모델(아래 backend)로 단계 없는 응답을 시도합니다."),
        ctl("s2s_backend", S2S_BACKENDS, "csm",
            "s2s 모드에서 쓸 음성 모델입니다.",
            "csm(Sesame, 표현형 TTS)은 외부 가중치가 필요합니다. 미준비 시 설치 안내가 표시됩니다. cascade 모드에서는 무시됩니다. moshi(풀듀플렉스)는 'Moshi 라이브' 카드를 쓰세요."),
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

    # S2S 모드 — 음성 네이티브 backend(moshi/csm)에 위임. cascade와 달리 STT/LLM/TTS·Ollama·voice가
    # 불필요하다(backend가 음성→음성을 직접 처리). 한 번에 하나의 턴만(모델 충돌 방지).
    if (pipeline_mode or "cascade") == "s2s":
        s2s_dev = device or cfg.get("device", "auto")
        s2s_name = s2s_backend or S2S_BACKENDS[0]
        with _LIVE_LOCK:
            try:
                return _run_s2s_turn(in_path, s2s_name, s2s_dev, tts_cfg)
            except Exception as e:
                return JSONResponse({"error": f"S2S 처리 실패: {type(e).__name__}: {e}"}, status_code=500)

    if llm_model:  # UI에서 고른 LLM 모델로 그 턴만 override(base_url/think는 config 유지)
        llm_cfg = {**llm_cfg, "model": llm_model}
    if system and system.strip():  # UI에서 편집한 시스템 프롬프트로 override(비우면 config 기본값)
        llm_cfg = {**llm_cfg, "system": system}
    stt_model = model or stt_cfg.get("model", "base.en")
    stt_compute = compute_type or stt_cfg.get("compute_type", "int8")
    stt_device = device or stt_cfg.get("device", "auto")
    voice_name = voice or tts_cfg.get("voice", "en_US-lessac-medium")
    voice_dir = _live_voice_dir(tts_cfg)

    # 사전 점검 — 실패 시 화면에 명확한 한국어 안내(503)
    if not _ollama_up():
        return JSONResponse(
            {"error": f"Ollama 미기동 — `ollama serve` 후 모델을 pull 하세요 (예: ollama pull {llm_cfg.get('model', '')})."},
            status_code=503)
    if not _voice_present(voice_dir, voice_name):
        return JSONResponse(
            {"error": f"Piper voice '{voice_name}' 없음 — TTS 카드에서 '보이스 다운로드'(--download)를 먼저 실행하세요."},
            status_code=503)

    # 턴은 한 번에 하나씩(모델/대화 history 충돌 방지 — 로컬 단일 사용자 가정)
    with _LIVE_LOCK:
        try:
            return _run_turn(in_path, stt_model, stt_compute, stt_device,
                             voice_dir, voice_name, llm_cfg)
        except Exception as e:
            return JSONResponse({"error": f"처리 실패: {type(e).__name__}: {e}"}, status_code=500)
