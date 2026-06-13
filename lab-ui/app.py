#!/usr/bin/env python3
"""lab-ui — 멀티모달 서빙 랩 대시보드 (FastAPI).

저장소의 9개 실험(serve/sd-gen/video-gen/tts-gen/stt-gen/llm-serve/voice-agent/
avatar-gen/s2s-gen)을 브라우저에서 실행·확인하는 얇은 레이어. 기존 CLI는 수정하지 않고,
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
import json
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

import live_voice

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
                       choices=["tiny", "base", "small", "medium",
                                "tiny.en", "base.en", "small.en", "medium.en", "large-v3"],
                       default="base.en"),
            "language": P("--language", "choice",
                          choices=["auto", "ko", "en", "ja", "zh"], default="auto"),
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
            "language": P("--language", "choice",
                          choices=["auto", "ko", "en", "ja", "zh"], default="auto"),
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
            "backend": P("--backend", "choice", choices=["static", "wav2lip", "musetalk"], default="static"),
            "device": P("--device", "choice", choices=["auto", "cuda", "mps", "cpu"], default="auto"),
        },
        "produces_files": True,
    },
    "s2s-gen": {
        "label": "S2S (음성 네이티브 vs 캐스케이드)",
        "dir": "s2s-gen",
        "kind": "job",
        "script": "s2s.py",
        "params": {
            "backend": P("--backend", "choice", choices=["cascade", "csm", "melo"], default="cascade"),
            "ask": P("--ask", "str", default="How do you spell necessary?"),
            "language": P("--language", "choice", choices=["auto", "ko", "en", "ja", "zh"], default="auto"),
            "device": P("--device", "choice", choices=["auto", "cuda", "mps", "cpu"], default="auto"),
        },
        "produces_files": True,
    },
}


PARAM_HELP: dict[str, dict[str, dict[str, str]]] = {
    "serve": {
        "concurrency": {
            "help": "벤치마크에서 동시에 보낼 요청 수 목록입니다.",
            "impact": "값을 키우면 동적 배칭과 큐 대기, 처리량 변화가 더 잘 드러납니다.",
            "placeholder": "예: 1 2 4 8",
        },
        "requests": {
            "help": "각 동시성 단계에서 보낼 총 요청 수입니다.",
            "impact": "값이 클수록 결과가 안정적이지만 벤치마크 시간이 늘어납니다.",
        },
        "payload": {
            "help": "/infer에 보낼 입력 문자열입니다.",
            "impact": "echo 어댑터에서는 출력 내용만 달라지고 지연 특성은 거의 변하지 않습니다.",
        },
    },
    "sd-gen": {
        "prompt": {
            "help": "생성하려는 이미지 내용을 자연어로 설명합니다.",
            "impact": "구체적으로 쓸수록 구도, 소재, 스타일이 프롬프트에 더 강하게 맞춰집니다.",
        },
        "negative_prompt": {
            "help": "피하고 싶은 요소를 적습니다.",
            "impact": "불필요한 품질 저하, 워터마크, 흐림 같은 특징을 줄이는 데 씁니다.",
            "empty": "기본 negative prompt를 사용합니다.",
            "fallback": "blurry, low quality, bad anatomy, ugly, watermark",
            "placeholder": "예: blurry, low quality, watermark",
        },
        "width": {
            "help": "이미지 너비(px)입니다.",
            "impact": "커질수록 디테일과 메모리 사용량, 생성 시간이 늘어납니다.",
        },
        "height": {
            "help": "이미지 높이(px)입니다.",
            "impact": "커질수록 디테일과 메모리 사용량, 생성 시간이 늘어납니다.",
        },
        "steps": {
            "help": "디퓨전 추론 반복 횟수입니다.",
            "impact": "보통 늘리면 품질이 안정되지만 생성 시간이 거의 비례해 증가합니다.",
        },
        "guidance": {
            "help": "CFG scale입니다. 프롬프트를 얼마나 강하게 따를지 정합니다.",
            "impact": "너무 낮으면 프롬프트 반영이 약하고, 너무 높으면 과장되거나 깨질 수 있습니다.",
            "empty": "기본 guidance scale을 사용합니다.",
            "fallback": "7.5",
        },
        "num_images": {
            "help": "한 번에 생성할 이미지 개수입니다.",
            "impact": "개수를 늘리면 비교 후보가 늘지만 실행 시간과 메모리 사용량도 늘어납니다.",
            "empty": "기본 이미지 개수를 사용합니다.",
            "fallback": "1",
        },
        "seed": {
            "help": "난수 시드입니다. 같은 설정과 시드를 쓰면 결과 재현에 도움이 됩니다.",
            "impact": "-1은 매번 랜덤이며, 정수를 넣으면 같은 구도를 다시 확인하기 쉽습니다.",
            "empty": "기본 시드 설정을 사용합니다.",
            "fallback": "-1",
            "placeholder": "예: -1 또는 1234",
        },
    },
    "video-gen": {
        "prompt": {
            "help": "생성할 영상 장면과 움직임을 설명합니다.",
            "impact": "움직임, 피사체, 카메라 표현을 명시하면 결과 방향이 더 분명해집니다.",
        },
        "negative": {
            "help": "영상에서 피하고 싶은 요소를 적습니다.",
            "impact": "흐림, 낮은 품질, 왜곡 같은 특징을 줄이는 데 씁니다.",
            "empty": "기본 negative prompt를 사용합니다.",
            "fallback": "bad quality, worst quality, low resolution, blurry",
            "placeholder": "예: blurry, bad quality, flicker",
        },
        "model": {
            "help": "사용할 비디오 생성 백엔드입니다.",
            "impact": "모델마다 기본 해상도, 프레임 수, 품질과 속도 특성이 다릅니다.",
        },
        "steps": {
            "help": "비디오 생성 추론 반복 횟수입니다.",
            "impact": "늘리면 안정성이 좋아질 수 있지만 생성 시간이 증가합니다.",
        },
        "guidance": {
            "help": "프롬프트 반영 강도입니다.",
            "impact": "높을수록 프롬프트를 강하게 따르지만 과도하면 품질이 불안정할 수 있습니다.",
            "empty": "선택한 비디오 모델의 기본 guidance 값을 사용합니다.",
            "fallback": "animatediff=7.5, zeroscope=9.0",
        },
        "width": {
            "help": "영상 프레임 너비(px)입니다.",
            "impact": "커질수록 선명도와 비용이 함께 늘어납니다.",
            "empty": "선택한 비디오 모델의 기본 너비를 사용합니다.",
            "fallback": "animatediff=512, zeroscope=576",
        },
        "height": {
            "help": "영상 프레임 높이(px)입니다.",
            "impact": "커질수록 선명도와 비용이 함께 늘어납니다.",
            "empty": "선택한 비디오 모델의 기본 높이를 사용합니다.",
            "fallback": "animatediff=512, zeroscope=320",
        },
        "frames": {
            "help": "생성할 프레임 수입니다.",
            "impact": "값이 클수록 영상이 길어지고 생성 시간이 늘어납니다.",
        },
        "fps": {
            "help": "저장할 mp4의 초당 프레임 수입니다.",
            "impact": "같은 프레임 수에서 fps가 높으면 재생 시간이 짧아지고 더 빠르게 보입니다.",
            "empty": "선택한 비디오 모델의 기본 fps를 사용합니다.",
            "fallback": "8",
        },
        "seed": {
            "help": "난수 시드입니다.",
            "impact": "-1은 매번 랜덤이며, 정수를 넣으면 유사한 결과 재현에 도움이 됩니다.",
            "fallback": "-1",
            "placeholder": "예: -1 또는 1234",
        },
    },
    "tts-gen": {
        "text": {
            "help": "Piper가 음성으로 읽을 텍스트입니다.",
            "impact": "문장이 길수록 오디오 길이와 합성 시간이 늘어납니다.",
        },
        "voice": {
            "help": "사용할 Piper 보이스 이름입니다.",
            "impact": "목소리와 언어가 바뀌며, 해당 .onnx 보이스 모델이 models에 있어야 합니다. 한국어는 ko_KR-kss-medium(커뮤니티 보이스)을 download로 먼저 받으세요.",
            "empty": "기본 Piper 보이스를 사용합니다.",
            "fallback": "en_US-lessac-medium",
            "placeholder": "예: en_US-lessac-medium 또는 ko_KR-kss-medium",
        },
        "length_scale": {
            "help": "발화 속도 계수입니다.",
            "impact": "1보다 크면 느리게, 1보다 작으면 빠르게 말합니다.",
            "empty": "기본 발화 속도를 사용합니다.",
            "fallback": "1.0",
        },
        "download": {
            "help": "체크하면 보이스 모델을 다운로드하고 합성은 하지 않습니다.",
            "impact": "TTS/voice-agent/avatar 실행 전에 필요한 보이스 파일을 준비합니다.",
        },
        "cuda": {
            "help": "체크하면 onnxruntime CUDA 실행을 요청합니다.",
            "impact": "NVIDIA GPU와 onnxruntime-gpu가 준비된 환경에서 합성 속도가 달라질 수 있습니다.",
        },
    },
    "stt-gen": {
        "from_tts": {
            "help": "TTS로 먼저 합성한 뒤 다시 전사할 기준 텍스트입니다.",
            "impact": "왕복 검증에서는 이 텍스트와 전사 결과를 비교해 WER를 계산합니다.",
        },
        "model": {
            "help": "faster-whisper 모델 크기입니다. .en 접미사는 영어 전용입니다.",
            "impact": "큰 모델은 정확도가 좋아질 수 있지만 다운로드, 메모리, 전사 시간이 늘어납니다. 한국어 등 비영어는 .en이 아닌 다국어 모델(tiny/base/small/medium 또는 large-v3)이 필요합니다.",
        },
        "language": {
            "help": "전사 언어입니다.",
            "impact": "auto는 자동 감지, ko/ja/zh 등은 해당 언어로 강제합니다. .en 모델로는 한국어 등이 인식되지 않으니 다국어 모델과 함께 쓰세요.",
        },
        "compute_type": {
            "help": "모델 계산 정밀도/양자화 방식입니다.",
            "impact": "int8은 CPU에서 가볍고, float16/int8_float16은 CUDA 환경에 적합합니다.",
        },
        "device": {
            "help": "STT 실행 장치입니다.",
            "impact": "auto는 가능한 장치를 고르고, cpu/cuda는 명시적으로 고정합니다.",
        },
    },
    "llm-serve": {
        "prompt": {
            "help": "LLM에 보낼 사용자 질문 또는 지시문입니다.",
            "impact": "응답 내용과 토큰 수, 지연 시간이 프롬프트에 따라 달라집니다.",
        },
        "model": {
            "help": "OpenAI 호환 서버에 요청할 모델 이름입니다.",
            "impact": "다른 모델을 지정하면 품질, 속도, 메모리 사용량이 달라집니다.",
            "empty": "기본 LLM 모델을 사용합니다.",
            "fallback": "llama3.2:1b-instruct-q4_K_M",
            "placeholder": "예: llama3.2:1b-instruct-q4_K_M",
        },
        "max_tokens": {
            "help": "생성할 최대 토큰 수입니다.",
            "impact": "값이 클수록 긴 답변이 가능하지만 총 지연 시간이 늘 수 있습니다.",
        },
    },
    "voice-agent": {
        "ask": {
            "help": "음성 에이전트에게 물어볼 질문 텍스트입니다.",
            "impact": "이 텍스트를 입력 음성으로 합성한 뒤 STT, LLM, TTS 전체 지연을 측정합니다. 한국어로 물으려면 language를 ko로 두고 한국어 보이스가 준비돼 있어야 합니다.",
        },
        "language": {
            "help": "대화 언어입니다 — STT 인식 언어·LLM 응답 언어·기본 보이스를 함께 결정합니다.",
            "impact": "auto는 자동 감지 후 같은 언어로 응답합니다. ko 등은 config의 languages 기본값(다국어 STT 모델·보이스·프롬프트)을 사용합니다.",
        },
    },
    "avatar-gen": {
        "prompt": {
            "help": "LLM이 말할 문장을 만들도록 줄 프롬프트입니다.",
            "impact": "입력하면 LLM 생성 단계를 거친 뒤 그 응답을 음성/영상으로 만듭니다.",
            "empty": "아래 text 값을 그대로 말하게 합니다.",
            "placeholder": "예: Greet a new learner in one sentence.",
        },
        "text": {
            "help": "아바타가 그대로 말할 텍스트입니다.",
            "impact": "LLM을 건너뛰고 바로 TTS와 영상 합성에 사용합니다.",
        },
        "backend": {
            "help": "토킹헤드 생성 방식입니다.",
            "impact": "static은 정지 영상 검증용, wav2lip/musetalk은 외부 모델로 실제 립싱크를 시도합니다(musetalk은 GPU 권장).",
        },
        "device": {
            "help": "립싱크 백엔드 실행 장치입니다.",
            "impact": "auto는 cuda, mps, cpu 순으로 가능한 장치를 고릅니다.",
        },
    },
    "s2s-gen": {
        "backend": {
            "help": "음성 응답 파이프라인 방식입니다.",
            "impact": "cascade는 STT→LLM→TTS 기준선, csm은 표현형 TTS, melo는 한국어 등 다국어 TTS로 응답합니다.",
        },
        "ask": {
            "help": "에이전트에게 물어볼 질문 텍스트입니다.",
            "impact": "이 텍스트를 입력 음성으로 합성한 뒤 backend의 TTFA/E2E/RTF를 측정합니다.",
        },
        "language": {
            "help": "대화 언어입니다.",
            "impact": "ko/en/ja/zh를 고르면 STT 모델·전사 언어·LLM 응답 언어·기본 질문 보이스·MeloTTS 언어가 함께 바뀝니다.",
        },
        "device": {
            "help": "음성 네이티브 백엔드 실행 장치입니다.",
            "impact": "auto는 cuda, mps, cpu 순으로 고릅니다. csm/melo의 로컬 실행 장치를 덮어씁니다.",
        },
    },
}


DEFAULT_EMPTY_HELP = "기본 설정값을 사용합니다."


METRIC_HELP: dict[str, dict[str, str]] = {
    "audio_s": {"help": "생성 또는 입력된 오디오의 길이입니다.", "direction": "작업 규모를 해석하는 기준입니다."},
    "synth_s": {"help": "TTS 음성 합성에 걸린 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "transcribe_s": {"help": "STT 전사에 걸린 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "rtf": {"help": "처리 시간 / 오디오 길이입니다.", "direction": "1보다 작으면 실시간보다 빠릅니다."},
    "wer": {"help": "기준 텍스트와 전사 결과의 단어 오류율입니다.", "direction": "낮을수록 정확합니다."},
    "realtime_x": {"help": "실시간 대비 처리 배수입니다.", "direction": "높을수록 빠릅니다."},
    "sr": {"help": "오디오 샘플레이트입니다.", "direction": "오디오 포맷 확인용입니다."},
    "chars": {"help": "입력 텍스트 문자 수입니다.", "direction": "합성 부하를 해석하는 기준입니다."},
    "ttft": {"help": "요청 후 첫 토큰이 나오기까지의 시간입니다.", "direction": "낮을수록 반응이 빠릅니다."},
    "tokens": {"help": "스트리밍 중 생성된 토큰 수의 근사값입니다.", "direction": "응답 길이 해석용입니다."},
    "decode_tok_s": {"help": "첫 토큰 이후 초당 생성한 토큰 수입니다.", "direction": "높을수록 생성 처리량이 좋습니다."},
    "total_s": {"help": "LLM 요청 전체 완료 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "stt_ms": {"help": "음성 입력을 텍스트로 바꾸는 단계 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "llm_ttft_ms": {"help": "LLM 첫 토큰 대기 시간입니다.", "direction": "낮을수록 대화 반응성이 좋습니다."},
    "llm_total_ms": {"help": "LLM 응답 생성 전체 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "tts_ms": {"help": "응답 텍스트를 음성으로 합성하는 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "e2e_ms": {"help": "사용자 발화 종료부터 응답 음성 생성까지의 전체 지연입니다.", "direction": "낮을수록 좋습니다."},
    "ttfa_ms": {"help": "입력 종료부터 첫 응답 오디오가 나오기까지의 시간입니다.", "direction": "낮을수록 대화 반응성이 좋습니다."},
    "bottleneck": {"help": "한 턴에서 가장 오래 걸린 단계입니다.", "direction": "최적화 우선순위를 보여줍니다."},
    "bottleneck_pct": {"help": "전체 시간 중 병목 단계가 차지한 비율입니다.", "direction": "높을수록 해당 단계 영향이 큽니다."},
    "tts_s": {"help": "아바타 파이프라인의 음성 합성 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "lipsync_s": {"help": "음성과 얼굴을 합쳐 영상으로 만드는 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "lipsync_rtf": {"help": "립싱크 시간 / 오디오 길이입니다.", "direction": "1보다 작으면 실시간보다 빠릅니다."},
    "e2e_s": {"help": "아바타 파이프라인 전체 완료 시간입니다.", "direction": "낮을수록 빠릅니다."},
    "backend": {"help": "사용한 실행 백엔드입니다.", "direction": "결과 비교 기준입니다."},
    "avg_batch_size": {"help": "서버가 실제로 묶어 처리한 평균 배치 크기입니다.", "direction": "동적 배칭 효과 확인용입니다."},
    "max_observed_batch": {"help": "관측된 최대 배치 크기입니다.", "direction": "설정한 최대 배치에 얼마나 접근했는지 보여줍니다."},
    "total_batches": {"help": "서버가 처리한 총 배치 수입니다.", "direction": "벤치마크 규모 확인용입니다."},
}


def target_dir(target: str) -> Path:
    return ROOT / TARGETS[target]["dir"]


def outputs_dir(target: str) -> Path:
    # 모든 서브 프로젝트 config의 output.dir 기본값은 "./outputs" (cwd=target dir 기준).
    return target_dir(target) / "outputs"


MELO_WITH = [
    "melotts @ git+https://github.com/myshell-ai/MeloTTS.git",
    "soundfile>=0.12",
    "unidic-lite>=1.0.8",
    "python-mecab-ko>=1.3.7",
]


def target_cmd(target: str, extra: Optional[str] = None, with_packages: Optional[list[str]] = None) -> list[str]:
    """target을 실행하는 명령 prefix.

    각 서브 프로젝트는 uv 프로젝트(pyproject.toml + .python-version)다.
    `uv run`이 cwd(target dir)의 .venv를 자동 생성·동기화한 뒤 파이썬을 실행하므로,
    사용자는 별도 사전 준비 없이 첫 실행에서 의존성이 설치된다.

    주의: `uv run` 은 매 실행마다 venv 를 lockfile(+지정 extra)에 맞춰 재동기화한다. 따라서
    optional extra 가 필요한 백엔드(예: s2s-gen csm = torch)는 반드시 같은 `--extra` 로 실행해야
    한다. extra 없이 실행하면 그 extra 패키지가 매번 제거된다(=torch 미설치 오류의 원인).
    """
    with_args: list[str] = []
    for pkg in with_packages or []:
        with_args.extend(["--with", pkg])
    return ["uv", "run", *(["--extra", extra] if extra else []), *with_args, "python"]


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
            env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            proc = subprocess.Popen(
                job["cmd"], cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=env,
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
        elif target == "s2s-gen":
            _parse_s2s_line(line, metrics)

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


def _parse_s2s_line(line: str, metrics: dict[str, Any]) -> None:
    m = re.search(r"\[backend\]\s+([A-Za-z0-9_-]+)", line)
    if m:
        metrics["backend"] = m.group(1)
    patterns = [
        (r"STT\s*:\s*([0-9.]+)\s*ms", "stt_ms"),
        (r"LLM TTFT\s*:\s*([0-9.]+)\s*ms", "llm_ttft_ms"),
        (r"LLM total\s*:\s*([0-9.]+)\s*ms", "llm_total_ms"),
        (r"TTS\s*:\s*([0-9.]+)\s*ms", "tts_ms"),
        (r"TTFA\s*:\s*([0-9.]+)\s*ms", "ttfa_ms"),
        (r"E2E\s*:\s*([0-9.]+)\s*ms", "e2e_ms"),
        (r"RTF\s*:\s*([0-9.]+)", "rtf"),
    ]
    for pattern, key in patterns:
        m = re.search(pattern, line)
        if m:
            metrics[key] = float(m.group(1))


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
        "ttfa_ms": "TTFA",
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
        "tts_ms": "ms", "e2e_ms": "ms", "ttfa_ms": "ms", "tts_s": "s", "lipsync_s": "s", "e2e_s": "s",
        "decode_tok_s": "tok/s", "realtime_x": "x", "bottleneck_pct": "%",
    }
    order = {
        "tts-gen": ["audio_s", "synth_s", "rtf", "realtime_x", "sr", "chars"],
        "stt-gen": ["audio_s", "transcribe_s", "rtf", "wer"],
        "llm-serve": ["ttft", "tokens", "decode_tok_s", "total_s"],
        "voice-agent": ["e2e_ms", "stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ms", "bottleneck", "bottleneck_pct"],
        "avatar-gen": ["backend", "audio_s", "tts_s", "lipsync_s", "lipsync_rtf", "e2e_s"],
        "s2s-gen": ["backend", "ttfa_ms", "e2e_ms", "rtf", "stt_ms", "llm_ttft_ms", "llm_total_ms", "tts_ms"],
        "serve": ["avg_batch_size", "max_observed_batch", "total_batches"],
    }.get(target, list(metrics))
    return {
        key: {
            "label": labels.get(key, key),
            "value": metrics[key],
            "unit": units.get(key, ""),
            "help": METRIC_HELP.get(key, {}).get("help", ""),
            "direction": METRIC_HELP.get(key, {}).get("direction", ""),
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
    # PiperVoice.load("<name>.onnx")는 같은 경로의 <name>.onnx.json도 필요하다.
    cfg = voice_models_dir / f"{voice}.onnx.json"
    if not (voice_models_dir / f"{voice}.onnx").exists() or not cfg.exists():
        return False
    try:
        phoneme_type = (json.loads(cfg.read_text(encoding="utf-8")).get("phoneme_type") or "espeak").lower()
    except Exception:
        return False
    return phoneme_type in ("espeak", "text")


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

    if target in ("tts-gen", "voice-agent", "avatar-gen", "s2s-gen"):
        vdir = ROOT / "tts-gen" / "models"
        v_ok = _voice_present(vdir, "en_US-lessac-medium")
        checks.append({
            "name": "piper voice",
            "ok": v_ok,
            "hint": "" if v_ok else "TTS 카드에서 '보이스 다운로드'(--download)를 먼저 실행하세요.",
        })

    if target in ("voice-agent", "s2s-gen"):
        vdir = ROOT / "tts-gen" / "models"
        ko_ok = _voice_present(vdir, "ko_KR-kss-medium")
        checks.append({
            "name": "korean piper voice",
            "ok": ko_ok,
            "hint": "" if ko_ok else "한국어 --ask/cascade 입력 합성용 보이스가 없습니다: "
                                    "cd tts-gen && uv run python synthesize.py --download --voice ko_KR-kss-medium",
        })

    if target in ("llm-serve", "voice-agent", "avatar-gen", "s2s-gen"):
        o_ok = _ollama_up()
        checks.append({
            "name": "Ollama",
            "ok": o_ok,
            "hint": "" if o_ok else "Ollama 미기동 — `ollama serve` 후 모델을 pull 하세요 "
                                    "(예: ollama pull gemma4:12b-mlx).",
        })

    if target in ("avatar-gen", "video-gen", "s2s-gen"):
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
# Moshi (full-duplex 라이브) — moshi_mlx.local_web 웹 서버를 관리(serve 패턴 재사용).
# Moshi는 파일→파일이 없어 s2s-gen backend로 못 돌리고, 자체 웹 UI(:8998)에서 실시간 대화한다.
# 별도 3.12 환경이 필요하므로 `uv run --python 3.12 --with moshi_mlx`로 임시 환경에서 기동한다.
# ---------------------------------------------------------------------------
MOSHI_PORT = 8998
MOSHI_URL = f"http://localhost:{MOSHI_PORT}"
MOSHI: dict[str, Any] = {"proc": None, "started_at": None, "log": deque(maxlen=LOG_TAIL)}
MOSHI_LOCK = threading.Lock()


def _moshi_running() -> bool:
    proc = MOSHI["proc"]
    return proc is not None and proc.poll() is None


def _moshi_web_up() -> bool:
    try:
        return httpx.get(MOSHI_URL, timeout=0.6).status_code < 500
    except Exception:
        return False


def _moshi_drain(proc):
    assert proc.stdout is not None
    for line in proc.stdout:
        MOSHI["log"].append(line.rstrip("\n"))


def moshi_start(quant: int = 4) -> dict:
    with MOSHI_LOCK:
        if _moshi_running():
            return {"status": "already_running", "url": MOSHI_URL}
        q = 8 if int(quant) == 8 else 4
        MOSHI["log"].clear()
        # uv가 Python 3.12를 자동 확보하고 moshi_mlx를 임시 설치해 웹 서버를 띄운다.
        # 첫 기동 시 가중치(kyutai/moshika-mlx-q{q})를 HF에서 받는다(비게이트).
        proc = subprocess.Popen(
            ["uv", "run", "--python", "3.12", "--with", "moshi_mlx",
             "python", "-m", "moshi_mlx.local_web", "-q", str(q)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            start_new_session=True,  # 그룹 단위 종료(uv 부모 + 서버 자식)
        )
        MOSHI["proc"] = proc
        MOSHI["started_at"] = time.time()
        threading.Thread(target=_moshi_drain, args=(proc,), daemon=True).start()
        return {"status": "starting", "url": MOSHI_URL, "quant": q}


def moshi_stop() -> dict:
    with MOSHI_LOCK:
        proc = MOSHI["proc"]
        if proc is None or proc.poll() is not None:
            MOSHI["proc"] = None
            return {"status": "stopped"}
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGKILL)
        MOSHI["proc"] = None
        return {"status": "stopped"}


# ---------------------------------------------------------------------------
# 모델 서비스(BentoML) 매니저 — 모델을 1회 로드·상주시키는 서비스를 기동/정지/상태.
# stt/tts/csm 을 같은 패턴으로 띄워 "인프로세스 vs 서빙" 지연을 실측한다. csm-tts 는 특히
# 로드가 무거워(매 턴 재로드 불가) 상주 서비스가 저지연의 핵심이다. serve/moshi 런처와 동형.
# (bentoml serve 는 uv 부모 + 서버 자식 구조 → start_new_session + killpg 로 그룹 종료.)
# ---------------------------------------------------------------------------
SERVICES: dict[str, dict] = {
    "whisper-stt": {"dir": "stt-gen", "target": "bento_service:WhisperSTT", "port": 3001, "extra": None},
    "piper-tts":   {"dir": "tts-gen", "target": "bento_service:PiperTTS",   "port": 3002, "extra": None},
    "csm-tts":     {"dir": "s2s-gen", "target": "bento_service:CSMTTS",     "port": 3003, "extra": "csm"},
    "melo-tts":    {"dir": "s2s-gen", "target": "bento_service:MeloTTS",    "port": 3004, "extra": None,
                    "with": ["bentoml>=1.4", *MELO_WITH]},
}
SVC: dict[str, dict] = {
    name: {"proc": None, "started_at": None, "log": deque(maxlen=LOG_TAIL)} for name in SERVICES
}
SVC_LOCK = threading.Lock()
_ERROR_RE = re.compile(r"(traceback|error|exception|runtimeerror|failed|no such file)", re.I)


def _svc_url(name: str) -> str:
    return f"http://127.0.0.1:{SERVICES[name]['port']}"


def _svc_running(name: str) -> bool:
    proc = SVC[name]["proc"]
    return proc is not None and proc.poll() is None


def _svc_up(name: str) -> bool:
    """BentoML /readyz — 모델 로딩이 끝나 요청을 받을 수 있나(CSM 은 로드가 길다)."""
    try:
        return httpx.get(f"{_svc_url(name)}/readyz", timeout=0.6).status_code < 500
    except Exception:
        return False


def _svc_drain(name: str, proc):
    assert proc.stdout is not None
    for line in proc.stdout:
        SVC[name]["log"].append(line.rstrip("\n"))


def svc_start(name: str) -> dict:
    spec = SERVICES[name]
    with SVC_LOCK:
        if _svc_running(name):
            return {"status": "already_running", "url": _svc_url(name)}
        SVC[name]["log"].clear()
        extra = spec["extra"]
        with_args: list[str] = []
        for pkg in spec.get("with", []):
            with_args.extend(["--with", pkg])
        # `uv run [--extra X] [--with pkg] bentoml serve ...` — 모델 1회 로드.
        cmd = ["uv", "run", *(["--extra", extra] if extra else []), *with_args,
               "bentoml", "serve", spec["target"], "--port", str(spec["port"])]
        # lab-ui 가 (루트 .venv 에서) 실행 중이면 VIRTUAL_ENV 가 상속돼 bentoml 워커가 그 venv 로 뜬다.
        # → s2s-gen 의 csm extra(torch 2.4 등)가 아닌 루트 venv 가 잡혀 로드 실패. VIRTUAL_ENV 를 제거해
        #   uv 가 각 프로젝트(cwd)의 .venv 를 쓰도록 강제한다.
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT / spec["dir"]),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True, env=env,
        )
        SVC[name]["proc"] = proc
        SVC[name]["started_at"] = time.time()
        threading.Thread(target=_svc_drain, args=(name, proc), daemon=True).start()
        return {"status": "starting", "url": _svc_url(name)}


def svc_stop(name: str) -> dict:
    with SVC_LOCK:
        proc = SVC[name]["proc"]
        if proc is None or proc.poll() is not None:
            SVC[name]["proc"] = None
            return {"status": "stopped"}
        _kill_group(proc, signal.SIGTERM)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            _kill_group(proc, signal.SIGKILL)
        SVC[name]["proc"] = None
        return {"status": "stopped"}


def svc_status(name: str) -> dict:
    log = list(SVC[name]["log"])
    last_error = next((line for line in reversed(log) if _ERROR_RE.search(line)), "")
    return {
        "name": name,
        "running": _svc_running(name),   # uv/서버 프로세스 살아있나
        "up": _svc_up(name),             # /readyz 응답하나(모델 로드 끝나 요청 가능)
        "url": _svc_url(name),
        "port": SERVICES[name]["port"],
        "extra": SERVICES[name]["extra"],
        "with": SERVICES[name].get("with", []),
        "started_at": SVC[name]["started_at"],
        "last_error": last_error,
        "log": log,
    }


# ---------------------------------------------------------------------------
# FastAPI 앱
# ---------------------------------------------------------------------------
app = FastAPI(title="multimodal-serving-lab · lab-ui")
app.include_router(live_voice.router)  # Live Voice 기능은 live_voice.py에 분리


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
async def health():
    return {
        "root": str(ROOT),
        "targets": [
            {"id": t, "label": meta["label"], "kind": meta["kind"],
             "params": _param_view(t, meta["params"]),
             "produces_files": meta["produces_files"]}
            for t, meta in TARGETS.items()
        ],
    }


def _param_view(target: str, params: dict) -> list[dict]:
    out = []
    for name, p in params.items():
        view = {
            "name": name, "type": p["type"],
            "choices": p.get("choices"),
            "required": bool(p.get("required")),
            "default": p.get("default"),
            "min": p.get("min"),
            "max": p.get("max"),
        }
        meta = PARAM_HELP.get(target, {}).get(name, {})
        empty_help = meta.get("empty", "" if p.get("required") else DEFAULT_EMPTY_HELP)
        if p["type"] == "bool" and "empty" not in meta:
            empty_help = "체크하지 않으면 이 플래그를 전달하지 않습니다."
        view.update({
            "help": meta.get("help", ""),
            "impact": meta.get("impact", ""),
            "empty": empty_help,
            "fallback": meta.get("fallback", ""),
            "placeholder": meta.get("placeholder", ""),
        })
        out.append(view)
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
    # s2s-gen csm/melo 백엔드는 별도 extra 가 필요 → `uv run --extra ...` 로 실행해야
    # uv 가 매 실행 시 그 패키지를 제거하지 않는다(cascade 는 가볍게 plain uv run).
    extra = None
    with_packages = None
    if target == "s2s-gen" and (body.params or {}).get("backend") in {"csm", "melo"}:
        backend = (body.params or {}).get("backend")
        if backend == "csm":
            extra = "csm"
        elif backend == "melo":
            with_packages = MELO_WITH
    cmd = [*target_cmd(target, extra=extra, with_packages=with_packages), TARGETS[target]["script"], *cli_args]
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


class MoshiStart(BaseModel):
    quant: int = 4


@app.post("/api/moshi/start")
async def api_moshi_start(body: MoshiStart = MoshiStart()):
    return moshi_start(body.quant)


@app.post("/api/moshi/stop")
async def api_moshi_stop():
    return moshi_stop()


@app.get("/api/moshi/status")
async def api_moshi_status():
    return {
        "running": _moshi_running(),     # uv/서버 프로세스 살아있나
        "web_up": _moshi_web_up(),       # :8998 응답하나(가중치 로딩 끝나 대화 가능)
        "url": MOSHI_URL,
        "started_at": MOSHI["started_at"],
        "log": list(MOSHI["log"]),
    }


# --- 모델 서비스(BentoML: whisper-stt / piper-tts / csm-tts) ---
@app.get("/api/svc")
async def api_svc_list():
    return {"services": [svc_status(name) for name in SERVICES]}


def _svc_or_404(name: str):
    if name not in SERVICES:
        raise HTTPException(404, f"알 수 없는 서비스: {name} (가능: {', '.join(SERVICES)})")


@app.post("/api/svc/{name}/start")
async def api_svc_start(name: str):
    _svc_or_404(name)
    return svc_start(name)


@app.post("/api/svc/{name}/stop")
async def api_svc_stop(name: str):
    _svc_or_404(name)
    return svc_stop(name)


@app.get("/api/svc/{name}/status")
async def api_svc_status(name: str):
    _svc_or_404(name)
    return svc_status(name)


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
