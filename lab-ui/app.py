#!/usr/bin/env python3
"""lab-ui — 멀티모달 서빙 랩 대시보드 (FastAPI).

저장소의 8개 실험(serve/sd-gen/video-gen/tts-gen/stt-gen/llm-serve/voice-agent/
avatar-gen)을 브라우저에서 실행·확인하는 얇은 레이어. 기존 CLI는 수정하지 않고,
target별 allowlist 인자만으로 subprocess를 실행해 로그·산출물을 보여준다.

실행:
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:7860

설계 메모:
- 동시 실행은 1개로 제한(로컬 GPU/MPS/CPU 메모리 충돌 완화).
- serve는 상시 FastAPI 서버 → 별도 start/stop + HTTP 프록시로 관리.
- 무거운 모델/런타임(모델 가중치·ffmpeg·Ollama)은 자동 설치하지 않음. preflight로 안내.
- 각 target은 uv 프로젝트(pyproject.toml)다. `uv run`으로 실행하므로 첫 실행 시
  uv가 해당 디렉터리의 .venv를 자동 생성·동기화한다(사전 준비 0단계).
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

LAB_UI = Path(__file__).resolve().parent
ROOT = LAB_UI.parent  # 저장소 루트 (multimodal-serving-lab)
STATIC = LAB_UI / "static"
UPLOADS = Path(tempfile.gettempdir()) / "multimodal-serving-lab" / "uploads"
LOG_TAIL = 600  # 작업별 로그 보관 줄 수


# ---------------------------------------------------------------------------
# Target 정의: 실제 CLI/config(코드에서 확인)에서 도출한 allowlist.
# 각 param: type(str|int|float|bool|choice) + flag + (choices/min/max).
# bool은 store_true 플래그. 미지정/빈 값은 전달하지 않는다(=CLI 기본값 사용).
# ---------------------------------------------------------------------------
def P(flag, type_, **kw):
    return {"flag": flag, "type": type_, **kw}


TARGETS: dict[str, dict] = {
    "serve": {
        "label": "Serving spine",
        "dir": "serve",
        "kind": "serve",  # 상시 서버 + bench job
        "script": "bench.py",  # job 경로로는 bench만 실행
        "params": {
            "concurrency": P("--concurrency", "intlist", default="1 2 4"),
            "requests": P("--requests", "int", min=1, max=100000, default=32),
            "payload": P("--payload", "str", default="hello serving spine"),
        },
        "produces_files": False,
    },
    "sd-gen": {
        "label": "Image (Stable Diffusion)",
        "dir": "sd-gen",
        "kind": "job",
        "script": "generate.py",
        "params": {
            "prompt": P("-p", "str", default="a cinematic mountain landscape at sunset"),
            "negative_prompt": P("-n", "str"),
            "width": P("-W", "int", min=64, max=2048, default=512),
            "height": P("-H", "int", min=64, max=2048, default=512),
            "steps": P("-s", "int", min=1, max=150, default=20),
            "guidance": P("-g", "float", min=0.0, max=30.0),
            "num_images": P("-N", "int", min=1, max=8),
            "seed": P("--seed", "int", min=-1, max=2**31 - 1),
        },
        "produces_files": True,
    },
    "video-gen": {
        "label": "Video (AnimateDiff / Zeroscope)",
        "dir": "video-gen",
        "kind": "job",
        "script": "generate.py",
        "params": {
            "prompt": P("-p", "str", required=True, default="a cat walking in a garden"),
            "negative": P("-n", "str"),
            "model": P("-m", "choice", choices=["animatediff", "zeroscope"], default="animatediff"),
            "steps": P("--steps", "int", min=1, max=150, default=20),
            "guidance": P("--guidance", "float", min=0.0, max=30.0),
            "width": P("--width", "int", min=64, max=2048),
            "height": P("--height", "int", min=64, max=2048),
            "frames": P("--frames", "int", min=1, max=240, default=16),
            "fps": P("--fps", "int", min=1, max=60),
            "seed": P("--seed", "int", min=-1, max=2**31 - 1),
        },
        "produces_files": True,
    },
    "tts-gen": {
        "label": "TTS (Piper)",
        "dir": "tts-gen",
        "kind": "job",
        "script": "synthesize.py",
        "params": {
            "text": P("-t", "str", default="Hello, this is a local Piper text to speech test."),
            "voice": P("--voice", "str"),
            "length_scale": P("--length-scale", "float", min=0.1, max=5.0),
            "download": P("--download", "bool"),
            "cuda": P("--cuda", "bool"),
        },
        "produces_files": True,
    },
    "stt-gen": {
        "label": "STT (faster-whisper)",
        "dir": "stt-gen",
        "kind": "job",
        "script": "transcribe.py",
        "params": {
            "from_tts": P("--from-tts", "str", default="hello from whisper"),
            "model": P("--model", "choice",
                       choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3"],
                       default="base.en"),
            "compute_type": P("--compute-type", "choice",
                              choices=["int8", "float32", "float16", "int8_float16"],
                              default="int8"),
            "device": P("--device", "choice", choices=["auto", "cpu", "cuda"], default="auto"),
        },
        "produces_files": False,
    },
    "llm-serve": {
        "label": "LLM (Ollama / vLLM)",
        "dir": "llm-serve",
        "kind": "job",
        "script": "chat.py",
        "params": {
            "prompt": P("-p", "str", required=True, default="Explain MLOps in one sentence."),
            "model": P("--model", "str"),
            "max_tokens": P("--max-tokens", "int", min=1, max=4096, default=64),
        },
        "produces_files": False,
    },
    "voice-agent": {
        "label": "Voice agent (STT→LLM→TTS)",
        "dir": "voice-agent",
        "kind": "job",
        "script": "agent.py",
        "params": {
            "ask": P("--ask", "str", default="How do you spell necessary?"),
        },
        "produces_files": True,
    },
    "avatar-gen": {
        "label": "Avatar (talking head)",
        "dir": "avatar-gen",
        "kind": "job",
        "script": "pipeline.py",
        "params": {
            "prompt": P("--prompt", "str"),
            "text": P("--text", "str", default="Hello, I am your tutor."),
            "backend": P("--backend", "choice", choices=["static", "wav2lip"], default="static"),
            "device": P("--device", "choice", choices=["auto", "cuda", "mps", "cpu"], default="auto"),
        },
        "produces_files": True,
    },
}


def target_dir(target: str) -> Path:
    return ROOT / TARGETS[target]["dir"]


def outputs_dir(target: str) -> Path:
    # 모든 서브 프로젝트 config의 output.dir 기본값은 "./outputs" (cwd=target dir 기준).
    return target_dir(target) / "outputs"


def target_cmd(target: str) -> list[str]:
    """target을 실행하는 명령 prefix.

    각 서브 프로젝트는 uv 프로젝트(pyproject.toml + .python-version)다.
    `uv run`이 cwd(target dir)의 .venv를 자동 생성·동기화한 뒤 파이썬을 실행하므로,
    사용자는 별도 사전 준비 없이 첫 실행에서 의존성이 설치된다.
    """
    return ["uv", "run", "python"]


# ---------------------------------------------------------------------------
# 인자 검증 — allowlist 밖 파라미터는 거부, 타입/choice/범위 검사 후 단일 인자로 전달.
# ---------------------------------------------------------------------------
def build_args(target: str, payload: dict) -> list[str]:
    spec = TARGETS[target]["params"]
    unknown = set(payload) - set(spec)
    if unknown:
        raise ValueError(f"허용되지 않은 파라미터: {sorted(unknown)}")

    args: list[str] = []
    for name, p in spec.items():
        if name not in payload:
            continue
        val = payload[name]
        t = p["type"]
        if t == "bool":
            if bool(val):
                args.append(p["flag"])
            continue
        if val is None or (isinstance(val, str) and val.strip() == ""):
            continue
        if t == "str":
            args += [p["flag"], str(val)]
        elif t == "choice":
            if str(val) not in p["choices"]:
                raise ValueError(f"{name}: {val!r}는 허용값 {p['choices']} 아님")
            args += [p["flag"], str(val)]
        elif t == "int":
            iv = int(val)
            _range_check(name, iv, p)
            args += [p["flag"], str(iv)]
        elif t == "float":
            fv = float(val)
            _range_check(name, fv, p)
            args += [p["flag"], str(fv)]
        elif t == "intlist":
            items = val if isinstance(val, list) else str(val).split()
            ints = [int(x) for x in items]
            if not ints or any(i < 1 or i > 1024 for i in ints):
                raise ValueError(f"{name}: 1~1024 정수 목록이어야 함")
            args += [p["flag"], *[str(i) for i in ints]]
        else:  # pragma: no cover - 정의 오류 방지용
            raise ValueError(f"알 수 없는 타입 {t}")

    # required 검사
    for name, p in spec.items():
        if p.get("required") and not args_contains(args, p["flag"]):
            raise ValueError(f"필수 파라미터 누락: {name}")
    return args


def _range_check(name, v, p):
    if "min" in p and v < p["min"]:
        raise ValueError(f"{name}: {v} < 최소 {p['min']}")
    if "max" in p and v > p["max"]:
        raise ValueError(f"{name}: {v} > 최대 {p['max']}")


def args_contains(args: list[str], flag: str) -> bool:
    return flag in args


# ---------------------------------------------------------------------------
# Job 관리 — 동시 실행 1개(글로벌 락). 로그는 합쳐서 tail 보관(stderr 포함).
# ---------------------------------------------------------------------------
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()        # JOBS dict 보호
RUN_LOCK = threading.Lock()         # 동시 실행 1개 제한


def _new_job(target: str, cmd: list[str]) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "target": target,
        "cmd": cmd,
        "status": "queued",  # queued|running|succeeded|failed
        "log": deque(maxlen=LOG_TAIL),
        "returncode": None,
        "queued_at": time.time(),
        "started_at": None,
        "ended_at": None,
    }
    with JOBS_LOCK:
        JOBS[job["id"]] = job
    return job


def _run_job(job: dict):
    target = job["target"]
    cwd = str(target_dir(target))
    with RUN_LOCK:  # 동시 실행 1개 — 락을 못 잡으면 queued 상태로 대기
        job["status"] = "running"
        job["started_at"] = time.time()
        try:
            proc = subprocess.Popen(
                job["cmd"], cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
        except Exception as e:  # 실행 자체 실패(파일 없음 등)
            job["log"].append(f"[lab-ui] 실행 실패: {e}")
            job["status"] = "failed"
            job["returncode"] = -1
            job["ended_at"] = time.time()
            return
        assert proc.stdout is not None
        for line in proc.stdout:
            job["log"].append(line.rstrip("\n"))
        proc.wait()
        job["returncode"] = proc.returncode
        job["status"] = "succeeded" if proc.returncode == 0 else "failed"
        job["ended_at"] = time.time()


def list_artifacts(target: str) -> list[dict]:
    d = outputs_dir(target)
    if not d.exists():
        return []
    items = []
    for f in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if f.is_file():
            st = f.stat()
            items.append({
                "name": f.name,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "role": artifact_role(target, f.name),
            })
    return items[:50]


def artifact_role(target: str, name: str) -> str:
    suffix = Path(name).suffix.lower()
    if target == "voice-agent":
        if name == "answer.wav":
            return "answer_audio"
        if name == "question.wav":
            return "question_audio"
    if target == "avatar-gen":
        if suffix == ".mp4":
            return "avatar_video"
        if name == "speech.wav":
            return "speech_audio"
        if name == "placeholder_face.png":
            return "placeholder_face"
    if suffix in (".wav", ".mp3", ".ogg", ".flac"):
        return "audio"
    if suffix in (".mp4", ".mov", ".webm"):
        return "video"
    if suffix in (".png", ".jpg", ".jpeg", ".webp"):
        return "image"
    return "file"


def parse_job_metrics(target: str, log: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    for line in log:
        if "[metrics]" in line:
            metrics.update(_parse_key_values(line.split("[metrics]", 1)[1]))
        if target == "serve":
            row = _parse_serve_bench_row(line)
            if row:
                rows.append(row)
            if "[server metrics]" in line:
                metrics.update(_parse_key_values(line.split("[server metrics]", 1)[1]))
        elif target == "voice-agent":
            _parse_voice_line(line, metrics)
        elif target == "avatar-gen":
            _parse_avatar_line(line, metrics)

    return _label_metrics(target, metrics), rows


def _parse_key_values(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cleaned = text.replace("≈", "=")
    for key, value in re.findall(r"([A-Za-z0-9_./-]+)\s*=\s*([^\s,)]+)", cleaned):
        out[_metric_key(key)] = _metric_value(value)
    rt = re.search(r"\(x([0-9.]+)\s+realtime\)", text)
    if rt:
        out["realtime_x"] = float(rt.group(1))
    return out


def _metric_key(key: str) -> str:
    aliases = {
        "audio": "audio_s",
        "synth": "synth_s",
        "transcribe": "transcribe_s",
        "total": "total_s",
        "decode_tok/s": "decode_tok_s",
        "throughput_rps": "rps",
    }
    return aliases.get(key, key).replace("/", "_").lower()


def _metric_value(value: str) -> Any:
    v = value.strip().rstrip(",")
    if v == "unknown":
        return v
    multiplier = 1.0
    if v.endswith("ms"):
        multiplier = 1.0
        v = v[:-2]
    elif v.endswith("s"):
        multiplier = 1.0
        v = v[:-1]
    try:
        n = float(v)
        return int(n) if n.is_integer() else round(n * multiplier, 4)
    except ValueError:
        return value.strip()


def _parse_voice_line(line: str, metrics: dict[str, Any]) -> None:
    patterns = [
        (r"STT\s*:\s*([0-9.]+)\s*ms", "stt_ms"),
        (r"LLM TTFT\s*:\s*([0-9.]+)\s*ms", "llm_ttft_ms"),
        (r"LLM total\s*:\s*([0-9.]+)\s*ms", "llm_total_ms"),
        (r"TTS\s*:\s*([0-9.]+)\s*ms", "tts_ms"),
        (r"E2E 응답지연\s*:\s*([0-9.]+)\s*ms", "e2e_ms"),
        (r"병목 단계\s*:\s*([A-Za-z0-9_-]+)\s*\(([0-9.]+)%\)", "bottleneck"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, line)
        if not m:
            continue
        if key == "bottleneck":
            metrics["bottleneck"] = m.group(1)
            metrics["bottleneck_pct"] = float(m.group(2))
        else:
            metrics[key] = float(m.group(1))


def _parse_avatar_line(line: str, metrics: dict[str, Any]) -> None:
    m = re.search(r"\[TTS\]\s+audio=([0-9.]+)s\s+synth=([0-9.]+)s", line)
    if m:
        metrics["audio_s"] = float(m.group(1))
        metrics["tts_s"] = float(m.group(2))
    m = re.search(r"\[lipsync:([^\]]+)\]\s+([0-9.]+)s\s+RTF=([0-9.]+)", line)
    if m:
        metrics["backend"] = m.group(1)
        metrics["lipsync_s"] = float(m.group(2))
        metrics["lipsync_rtf"] = float(m.group(3))
    m = re.search(r"\[E2E\]\s+([0-9.]+)s", line)
    if m:
        metrics["e2e_s"] = float(m.group(1))


def _parse_serve_bench_row(line: str) -> Optional[dict[str, Any]]:
    parts = line.split()
    if len(parts) != 7:
        return None
    try:
        conc, reqs = int(parts[0]), int(parts[1])
        wall, rps, p50, p95, p99 = [float(x) for x in parts[2:]]
    except ValueError:
        return None
    return {
        "concurrency": conc,
        "requests": reqs,
        "wall_s": wall,
        "rps": rps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
    }


def _label_metrics(target: str, metrics: dict[str, Any]) -> dict[str, Any]:
    labels = {
        "audio_s": "오디오 길이",
        "synth_s": "합성 시간",
        "transcribe_s": "전사 시간",
        "rtf": "RTF",
        "wer": "WER",
        "realtime_x": "실시간 배수",
        "sr": "샘플레이트",
        "chars": "문자 수",
        "ttft": "TTFT",
        "tokens": "토큰 수",
        "decode_tok_s": "Decode tok/s",
        "total_s": "총 시간",
        "stt_ms": "STT",
        "llm_ttft_ms": "LLM TTFT",
        "llm_total_ms": "LLM total",
        "tts_ms": "TTS",
        "e2e_ms": "E2E",
        "bottleneck": "병목 단계",
        "bottleneck_pct": "병목 비중",
        "tts_s": "TTS 시간",
        "lipsync_s": "립싱크 시간",
        "lipsync_rtf": "립싱크 RTF",
        "e2e_s": "E2E",
        "backend": "Backend",
        "avg_batch_size": "평균 배치",
        "max_observed_batch": "최대 배치",
        "total_batches": "총 배치 수",
    }
    units = {
        "audio_s": "s", "synth_s": "s", "transcribe_s": "s", "total_s": "s",
        "ttft": "ms", "stt_ms": "ms", "llm_ttft_ms": "ms", "llm_total_ms": "ms",
        "tts_ms": "ms", "e2e_ms": "ms", "tts_s": "s", "lipsync_s": "s", "e2e_s": "s",
        "decode_tok_s": "tok/s", "realtime_x": "x", "bottleneck_pct": "%",
    }
    order = {
        "tts-gen": ["audio_s", "synth_s", "rtf", "realtime_x", "sr", "chars"],
        "stt-gen": ["audio_s", "transcribe_s", "rtf", "wer"],
        "llm-serve": ["ttft", "tokens", "decode_tok_s", "total_s"],
        "voice-agent": ["e2e_ms", "stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ms", "bottleneck", "bottleneck_pct"],
        "avatar-gen": ["backend", "audio_s", "tts_s", "lipsync_s", "lipsync_rtf", "e2e_s"],
        "serve": ["avg_batch_size", "max_observed_batch", "total_batches"],
    }.get(target, list(metrics))
    return {
        key: {
            "label": labels.get(key, key),
            "value": metrics[key],
            "unit": units.get(key, ""),
        }
        for key in order
        if key in metrics
    }


def job_view(job: dict) -> dict:
    log = list(job["log"])
    metrics, metric_rows = parse_job_metrics(job["target"], log)
    return {
        "id": job["id"],
        "target": job["target"],
        "status": job["status"],
        "returncode": job["returncode"],
        "queued_at": job["queued_at"],
        "started_at": job["started_at"],
        "ended_at": job["ended_at"],
        "log": log,
        "metrics": metrics,
        "metric_rows": metric_rows,
        "artifacts": list_artifacts(job["target"]) if TARGETS[job["target"]]["produces_files"] else [],
    }


# ---------------------------------------------------------------------------
# Preflight — "그냥 동작" vs "사전 준비 필요"를 판별해 카드에 안내.
# 무거운 import는 피하고, venv 존재 + 외부 런타임(Ollama/ffmpeg/voice cache)만 빠르게 점검.
# ---------------------------------------------------------------------------
def _ollama_up(base: str = "http://localhost:11434") -> bool:
    try:
        r = httpx.get(f"{base}/api/tags", timeout=0.6)
        return r.status_code == 200
    except Exception:
        return False


def _voice_present(voice_models_dir: Path, voice: str) -> bool:
    # piper voice는 <name>.onnx 형태로 캐시된다.
    if not voice_models_dir.exists():
        return False
    return any(voice_models_dir.glob(f"{voice}*.onnx"))


def preflight(target: str) -> dict:
    d = target_dir(target)
    checks: list[dict] = []

    # uv 설치 여부 — 모든 타깃이 `uv run`으로 실행된다.
    uv_ok = shutil.which("uv") is not None
    checks.append({
        "name": "uv",
        "ok": uv_ok,
        "hint": "" if uv_ok else "uv 미설치 — `brew install uv` (또는 https://astral.sh/uv).",
    })

    # 의존성 동기화 여부 — .venv 가 있으면 sync 완료. 없어도 uv run 첫 실행 시 자동 설치.
    synced = (d / ".venv" / "bin" / "python").exists()
    checks.append({
        "name": "deps synced",
        "ok": synced,
        "hint": "" if synced else f"cd {TARGETS[target]['dir']} && uv sync "
                                  "(또는 카드 실행 시 첫 회 자동 설치 — 무거운 타깃은 시간이 걸립니다).",
    })

    if target in ("tts-gen", "voice-agent", "avatar-gen"):
        vdir = ROOT / "tts-gen" / "models"
        v_ok = _voice_present(vdir, "en_US-lessac-medium")
        checks.append({
            "name": "piper voice",
            "ok": v_ok,
            "hint": "" if v_ok else "TTS 카드에서 '보이스 다운로드'(--download)를 먼저 실행하세요.",
        })

    if target in ("llm-serve", "voice-agent", "avatar-gen"):
        o_ok = _ollama_up()
        checks.append({
            "name": "Ollama",
            "ok": o_ok,
            "hint": "" if o_ok else "Ollama 미기동 — `ollama serve` 후 모델을 pull 하세요 "
                                    "(예: ollama pull llama3.2:1b-instruct-q4_K_M).",
        })

    if target in ("avatar-gen", "video-gen"):
        f_ok = shutil.which("ffmpeg") is not None
        checks.append({
            "name": "ffmpeg",
            "ok": f_ok,
            "hint": "" if f_ok else "ffmpeg 미설치 — `brew install ffmpeg`.",
        })

    status = "ready" if all(c["ok"] for c in checks) else "needs_setup"
    return {"target": target, "status": status, "checks": checks}


def avatar_wav2lip_preflight() -> dict:
    cfg = {}
    try:
        with open(ROOT / "avatar-gen" / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        pass
    lip = cfg.get("lipsync", {})
    base = ROOT / "avatar-gen"

    def resolve(value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (base / p).resolve()

    repo_value = lip.get("wav2lip_dir") or os.environ.get("WAV2LIP_DIR", "")
    ckpt_value = lip.get("wav2lip_ckpt") or os.environ.get("WAV2LIP_CKPT", "")
    repo = resolve(repo_value) if repo_value else Path()
    ckpt = resolve(ckpt_value) if ckpt_value else Path()

    checks = [
        {
            "name": "Wav2Lip repo",
            "ok": bool(repo_value) and (repo / "inference.py").exists(),
            "hint": "" if repo_value else "config.yaml의 wav2lip_dir 또는 WAV2LIP_DIR가 필요합니다.",
        },
        {
            "name": "Wav2Lip checkpoint",
            "ok": bool(ckpt_value) and ckpt.exists(),
            "hint": "" if ckpt_value else "config.yaml의 wav2lip_ckpt 또는 WAV2LIP_CKPT가 필요합니다.",
        },
        {
            "name": "ffmpeg",
            "ok": shutil.which("ffmpeg") is not None,
            "hint": "" if shutil.which("ffmpeg") else "ffmpeg 미설치 — `brew install ffmpeg`.",
        },
    ]
    if repo_value and not checks[0]["ok"]:
        checks[0]["hint"] = f"inference.py를 찾을 수 없습니다: {repo}"
    if ckpt_value and not checks[1]["ok"]:
        checks[1]["hint"] = f"checkpoint를 찾을 수 없습니다: {ckpt}"
    return {
        "target": "avatar-gen",
        "backend": "wav2lip",
        "status": "ready" if all(c["ok"] for c in checks) else "needs_setup",
        "checks": checks,
    }


# ---------------------------------------------------------------------------
# serve 상시 서버 관리 + HTTP 프록시
# ---------------------------------------------------------------------------
def _serve_cfg() -> dict:
    try:
        with open(ROOT / "serve" / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_SC = _serve_cfg().get("server", {})
SERVE_HOST = _SC.get("host", "127.0.0.1")
SERVE_PORT = int(_SC.get("port", 8000))
SERVE_URL = f"http://{SERVE_HOST}:{SERVE_PORT}"

SERVE: dict[str, Any] = {"proc": None, "started_at": None, "log": deque(maxlen=LOG_TAIL)}
SERVE_LOCK = threading.Lock()


def _serve_running() -> bool:
    proc = SERVE["proc"]
    return proc is not None and proc.poll() is None


def _serve_drain(proc):
    assert proc.stdout is not None
    for line in proc.stdout:
        SERVE["log"].append(line.rstrip("\n"))


def serve_start() -> dict:
    with SERVE_LOCK:
        if _serve_running():
            return {"status": "already_running", "url": SERVE_URL}
        SERVE["log"].clear()
        proc = subprocess.Popen(
            [*target_cmd("serve"), "server.py"],
            cwd=str(target_dir("serve")),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            # 새 세션(프로세스 그룹) → `uv run`이 띄운 자식 서버까지 그룹 단위로 종료 가능.
            start_new_session=True,
        )
        SERVE["proc"] = proc
        SERVE["started_at"] = time.time()
        threading.Thread(target=_serve_drain, args=(proc,), daemon=True).start()
        return {"status": "starting", "url": SERVE_URL}


def _kill_group(proc, sig) -> None:
    """proc의 프로세스 그룹 전체에 시그널. 그룹이 없으면 proc만."""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError):
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass


def serve_stop() -> dict:
    with SERVE_LOCK:
        proc = SERVE["proc"]
        if proc is None or proc.poll() is not None:
            SERVE["proc"] = None
            return {"status": "stopped"}
        # `uv run python server.py`는 uv(부모) + 실제 서버(자식) 구조라
        # 그룹 단위 종료로 자식까지 확실히 내려 8000(또는 설정 포트)을 해제한다.
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGKILL)
        SERVE["proc"] = None
        return {"status": "stopped"}


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------
app = FastAPI(title="multimodal-serving-lab · lab-ui")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {
        "root": str(ROOT),
        "targets": [
            {"id": t, "label": meta["label"], "kind": meta["kind"],
             "params": _param_view(meta["params"]),
             "produces_files": meta["produces_files"]}
            for t, meta in TARGETS.items()
        ],
    }


def _param_view(params: dict) -> list[dict]:
    out = []
    for name, p in params.items():
        out.append({
            "name": name, "type": p["type"],
            "choices": p.get("choices"),
            "required": bool(p.get("required")),
            "default": p.get("default"),
        })
    return out


@app.get("/api/preflight/{target}")
async def api_preflight(target: str):
    if target not in TARGETS:
        raise HTTPException(404, "알 수 없는 target")
    return preflight(target)


@app.get("/api/preflight/avatar-gen/wav2lip")
async def api_avatar_wav2lip_preflight():
    return avatar_wav2lip_preflight()


class JobRequest(BaseModel):
    params: dict = {}


@app.post("/api/jobs/{target}")
async def create_job(target: str, body: JobRequest):
    if target not in TARGETS:
        raise HTTPException(404, "알 수 없는 target")
    try:
        cli_args = build_args(target, body.params or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    # serve bench(bench.py)는 lab-ui가 관리하는 serve 포트를 향하도록 --url을 주입한다.
    # (config의 server.port 변경 시 bench.py 기본값 8000과 어긋나는 것을 방지)
    if target == "serve":
        cli_args = ["--url", SERVE_URL, *cli_args]
    cmd = [*target_cmd(target), TARGETS[target]["script"], *cli_args]
    job = _new_job(target, cmd)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job["id"], "cmd": cmd}


@app.post("/api/stt/audio")
async def create_stt_audio_job(
    audio: UploadFile = File(...),
    model: Optional[str] = Form(None),
    compute_type: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
):
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix != ".wav":
        raise HTTPException(400, "현재 STT 업로드는 wav 파일만 지원합니다.")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / f"stt_{uuid.uuid4().hex}.wav"
    try:
        with open(path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
    except Exception as e:
        raise HTTPException(400, f"업로드 저장 실패: {e}")
    finally:
        await audio.close()

    payload = {
        k: v for k, v in {
            "model": model,
            "compute_type": compute_type,
            "device": device,
        }.items()
        if v
    }
    try:
        cli_args = build_args("stt-gen", payload)
    except ValueError as e:
        raise HTTPException(400, str(e))
    cmd = [*target_cmd("stt-gen"), TARGETS["stt-gen"]["script"], "--audio", str(path), *cli_args]
    job = _new_job("stt-gen", cmd)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return {"job_id": job["id"], "cmd": cmd}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "알 수 없는 job")
    return job_view(job)


@app.get("/api/artifacts/{target}/{filename}")
async def get_artifact(target: str, filename: str):
    if target not in TARGETS:
        raise HTTPException(404, "알 수 없는 target")
    base = outputs_dir(target).resolve()
    candidate = (base / filename).resolve()
    # path traversal 차단: outputs 루트 밖이면 거부
    if not str(candidate).startswith(str(base) + "/") and candidate != base:
        raise HTTPException(403, "허용되지 않은 경로")
    if not candidate.is_file():
        raise HTTPException(404, "파일 없음")
    return FileResponse(str(candidate))


# --- serve 전용 ---
@app.post("/api/serve/start")
async def api_serve_start():
    return serve_start()


@app.post("/api/serve/stop")
async def api_serve_stop():
    return serve_stop()


@app.get("/api/serve/status")
async def api_serve_status():
    return {
        "running": _serve_running(),
        "url": SERVE_URL,
        "started_at": SERVE["started_at"],
        "log": list(SERVE["log"]),
    }


@app.get("/api/serve/proxy/{kind}")
async def api_serve_proxy_get(kind: str):
    if kind not in ("health", "metrics"):
        raise HTTPException(404, "health|metrics 만 지원")
    if not _serve_running():
        return JSONResponse({"error": "serve 서버가 기동되지 않았습니다. 먼저 start 하세요."},
                            status_code=409)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{SERVE_URL}/{kind}", timeout=5.0)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"프록시 실패: {e}"}, status_code=502)


class InferBody(BaseModel):
    input: Any


@app.post("/api/serve/proxy/infer")
async def api_serve_proxy_infer(body: InferBody):
    if not _serve_running():
        return JSONResponse({"error": "serve 서버가 기동되지 않았습니다. 먼저 start 하세요."},
                            status_code=409)
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{SERVE_URL}/infer", json={"input": body.input}, timeout=30.0)
        return JSONResponse(r.json(), status_code=r.status_code)
    except Exception as e:
        return JSONResponse({"error": f"프록시 실패: {e}"}, status_code=502)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")


if __name__ == "__main__":
    main()
