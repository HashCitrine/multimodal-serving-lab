#!/usr/bin/env python3
"""스트리밍 TTS TTFB — 청크(문장) 단위로 합성하며 '첫 오디오까지(TTFB)'를 측정.

대화형 음성에서 체감 지연을 줄이는 핵심: 전체 합성을 기다리지 말고 **첫 문장 오디오가 나오는
즉시 재생**한다. Piper 의 synthesize 는 문장 단위로 AudioChunk 를 yield 하므로, 첫 청크까지의
시간(TTFB)이 전체 합성시간보다 훨씬 짧다 → 그 차이를 실측한다.

사용: python stream_ttfb.py
"""
from __future__ import annotations

import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

TEXT = ("Hello, and welcome to your English lesson. "
        "Today we will practice greetings and introductions. "
        "Let us begin with a simple conversation. "
        "Please repeat after me when you are ready.")


def main():
    cfg = yaml.safe_load(open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8"))
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(SCRIPT_DIR / cfg["voice"]["model_dir"] / f"{cfg['voice']['name']}.onnx"))
    sr = voice.config.sample_rate
    sc = SynthesisConfig()
    list(voice.synthesize("warm up", sc))  # 워밍업

    t0 = time.perf_counter()
    ttfb = None
    chunk_times = []
    total_samples = 0
    first_chunk_samples = 0
    for i, chunk in enumerate(voice.synthesize(TEXT, sc)):
        now = time.perf_counter() - t0
        n = len(chunk.audio_int16_array)
        total_samples += n
        if ttfb is None:
            ttfb = now
            first_chunk_samples = n
        chunk_times.append((i, now, n / sr))
    total_synth = time.perf_counter() - t0
    total_audio = total_samples / sr

    print(f"text chars={len(TEXT)}  chunks={len(chunk_times)}  sr={sr}\n")
    for i, t, dur in chunk_times:
        print(f"  chunk {i}: ready @ {t*1000:6.0f} ms   (audio {dur:.2f}s)")
    print(f"\n  TTFB(첫 오디오)     : {ttfb*1000:6.0f} ms")
    print(f"  전체 합성시간       : {total_synth*1000:6.0f} ms")
    print(f"  전체 오디오 길이    : {total_audio:5.2f} s")
    print(f"  → 비스트리밍이면 {total_synth*1000:.0f}ms 대기, 스트리밍이면 {ttfb*1000:.0f}ms 후 재생 시작 "
          f"({(1-ttfb/total_synth)*100:.0f}% 체감지연 단축)")


if __name__ == "__main__":
    main()
