#!/usr/bin/env python3
"""S2S 지연 벤치 — 같은 질문 세트로 backend 의 TTFA·E2E·RTF 중앙값을 측정.

모델은 1회 로드(정상상태 가정) 후 N개 질문을 돌려, 캐스케이드와 음성 네이티브 모델의
'대화형 응답 지연'을 같은 축으로 비교한다.

사용:
    python bench_s2s.py --backend cascade
    python bench_s2s.py --backend csm
    python bench_s2s.py --backend melo --language ko
"""
from __future__ import annotations

import argparse
import statistics
import wave
from pathlib import Path

import numpy as np
import yaml

from backends import build_backend

SCRIPT_DIR = Path(__file__).parent

QUESTIONS = [
    "What is the plural of child?",
    "How do you spell the word necessary?",
    "Give me a synonym for happy.",
    "What is the past tense of run?",
    "Is it correct to say I have went?",
]


def resolve(rel):
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def synth(cfg, voice, sc, text, path):
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, sc)])
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(voice.config.sample_rate); w.writeframes(audio.tobytes())


def main():
    ap = argparse.ArgumentParser(description="S2S 지연 벤치")
    ap.add_argument("--backend", choices=["cascade", "csm", "melo"])
    ap.add_argument("--language", help="대화 언어(auto|ko|en|ja|zh)")
    ap.add_argument("--device")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(SCRIPT_DIR / args.config, "r", encoding="utf-8"))
    if args.backend:
        cfg["backend"] = args.backend
    if args.language:
        cfg["language"] = args.language
    if args.device:
        cfg["device"] = args.device

    backend = build_backend(cfg)
    if not backend.available():
        print(f"[{backend.name}] 백엔드가 아직 준비되지 않았습니다:")
        for issue in backend.diagnostics():
            print(f"  - {issue}")
        raise SystemExit(1)

    # 질문 음성 미리 합성 (Piper)
    from piper import PiperVoice, SynthesisConfig
    tts = cfg.get("tts", {})
    lang = cfg.get("language", "auto")
    entry = (cfg.get("languages", {}) or {}).get(lang, {}) if lang != "auto" else {}
    voice_name = entry.get("voice") or tts.get("voice", "en_US-lessac-medium")
    voice = PiperVoice.load(str(resolve(tts.get("voice_dir", "../tts-gen/models")) / f"{voice_name}.onnx"))
    sc = SynthesisConfig()
    out_dir = (SCRIPT_DIR / "outputs"); out_dir.mkdir(exist_ok=True)

    qwavs = []
    for i, q in enumerate(QUESTIONS):
        p = str(out_dir / f"bench_q{i}.wav")
        synth(cfg, voice, sc, q, p)
        qwavs.append(p)

    # 워밍업
    backend.generate(qwavs[0], str(out_dir / "bench_warm.wav"))

    ttfa, e2e, rtf = [], [], []
    for i, qw in enumerate(qwavs):
        m = backend.generate(qw, str(out_dir / f"bench_a{i}.wav"))
        ttfa.append(m.get("ttfa_s", 0) * 1000)
        e2e.append(m.get("e2e_s", 0) * 1000)
        rtf.append(m.get("rtf", 0))

    med = statistics.median
    print(f"\nbackend={backend.name}  questions={len(QUESTIONS)}  (정상상태, 모델 1회 로드)\n")
    print(f"  TTFA median : {med(ttfa):6.0f} ms")
    print(f"  E2E  median : {med(e2e):6.0f} ms")
    print(f"  RTF  median : {med(rtf):6.3f}")


if __name__ == "__main__":
    main()
