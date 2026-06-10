#!/usr/bin/env python3
"""TTS RTF(real-time factor) 벤치마크.

길이가 다른 여러 문장을 합성하며 RTF·합성시간·오디오길이를 측정해 표로 출력한다.
RTF = 합성시간 / 생성오디오길이 (낮을수록 좋음, <1 이면 실시간보다 빠름).

사용:
    python bench_rtf.py                 # config.yaml 보이스
    python bench_rtf.py --runs 5
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent

SENTENCES = [
    "Hello.",
    "This is a short sentence for benchmarking.",
    "Text to speech latency depends on input length, model size, and hardware backend.",
    ("Real time factor measures how long synthesis takes relative to the audio it "
     "produces; a value well below one means the model is faster than real time, which "
     "is what we want for responsive serving of interactive voice features."),
]


def main():
    ap = argparse.ArgumentParser(description="Piper TTS RTF 벤치")
    ap.add_argument("--runs", type=int, default=3, help="문장별 반복 측정 수")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    cfg_path = SCRIPT_DIR / args.config
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    from piper import PiperVoice, SynthesisConfig

    name = cfg["voice"]["name"]
    model_dir = Path(cfg["voice"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = SCRIPT_DIR / model_dir
    model_path = model_dir / f"{name}.onnx"
    voice = PiperVoice.load(str(model_path), use_cuda=args.cuda)
    sr = voice.config.sample_rate
    sc = SynthesisConfig(length_scale=cfg["synthesis"].get("length_scale", 1.0))

    # 워밍업
    list(voice.synthesize("warm up", sc))

    print(f"voice={name}  sr={sr}  runs={args.runs}  cuda={args.cuda}\n")
    print(f"{'chars':>6} {'audio_s':>8} {'synth_s':>8} {'RTF':>7} {'xRT':>7}")
    all_rtf = []
    for text in SENTENCES:
        synth_times = []
        audio_s = 0.0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            chunks = list(voice.synthesize(text, sc))
            synth_times.append(time.perf_counter() - t0)
            audio = np.concatenate([c.audio_int16_array for c in chunks])
            audio_s = len(audio) / sr
        synth_s = statistics.median(synth_times)
        rtf = synth_s / audio_s if audio_s else float("nan")
        all_rtf.append(rtf)
        print(f"{len(text):>6} {audio_s:>8.2f} {synth_s:>8.4f} {rtf:>7.4f} {1/rtf:>6.1f}x")

    print(f"\nmean RTF={statistics.mean(all_rtf):.4f}  "
          f"median RTF={statistics.median(all_rtf):.4f}  "
          f"(<1 = faster than real time)")


if __name__ == "__main__":
    main()
