#!/usr/bin/env python3
"""STT 양자화 벤치 — 모델 크기 × compute_type(int8/float32) 의 RTF·WER·메모리 트레이드오프.

핵심 질문: "int8 양자화는 언제 이득인가?" Piper(TTS)로 만든 클립을 faster-whisper로
전사하며, 각 (model, compute_type) 조합의 전사 속도(RTF)·정확도(WER)·메모리(peak RSS)를 잰다.
메모리는 조합별로 **새 서브프로세스**에서 재서 격리한다.

사용:
    python bench_stt.py --models base.en small.en --compute-types int8 float32
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Adaptive learning systems can combine voice and motion models.",
    "Real time factor measures synthesis speed relative to audio length.",
    "Quantization trades a little accuracy for lower memory and faster inference.",
]


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def _wer(ref, hyp):
    r, h = _norm(ref), _norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def make_clips(voice_dir: Path) -> list:
    from piper import PiperVoice, SynthesisConfig

    voice = PiperVoice.load(str(voice_dir / "en_US-lessac-medium.onnx"))
    sr = voice.config.sample_rate
    out = []
    tmpd = Path(tempfile.mkdtemp(prefix="stt_clips_"))
    for i, s in enumerate(SENTENCES):
        audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(s, SynthesisConfig())])
        path = tmpd / f"clip_{i}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio.tobytes())
        out.append({"text": s, "path": str(path), "dur": len(audio) / sr})
    return out


def worker(model_name: str, compute_type: str, clips_json: str, device: str = "auto") -> None:
    """서브프로세스: 한 조합을 측정해 JSON 출력(부모가 메모리 격리를 위해 호출)."""
    clips = json.loads(clips_json)
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    list(model.transcribe(clips[0]["path"], beam_size=1)[0])  # warmup
    tot_t = tot_a = 0.0
    wers = []
    for c in clips:
        t0 = time.perf_counter()
        segs, _ = model.transcribe(c["path"], beam_size=1)
        text = "".join(s.text for s in segs)
        tot_t += time.perf_counter() - t0
        tot_a += c["dur"]
        wers.append(_wer(c["text"], text))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if rss > 1e7 else rss / 1024  # macOS=bytes, linux=KB
    print(json.dumps({
        "rtf": tot_t / tot_a,
        "wer": sum(wers) / len(wers),
        "rss_mb": rss_mb,
    }))


def main():
    ap = argparse.ArgumentParser(description="STT 양자화 벤치")
    ap.add_argument("--models", nargs="+", default=["base.en", "small.en"])
    ap.add_argument("--compute-types", nargs="+", dest="compute_types",
                    default=["float32", "int8"])
    ap.add_argument("--voice-dir", default="../tts-gen/models")
    ap.add_argument("--device", default="auto",
                    help="auto|cpu|cuda (NVIDIA면 auto가 cuda 선택; GPU에선 float16/int8_float16 추천)")
    ap.add_argument("--_worker", nargs=4, help=argparse.SUPPRESS)  # 내부용: model ct clips device
    args = ap.parse_args()

    if args._worker:
        worker(args._worker[0], args._worker[1], args._worker[2], args._worker[3])
        return

    voice_dir = Path(args.voice_dir)
    if not voice_dir.is_absolute():
        voice_dir = SCRIPT_DIR / voice_dir
    clips = make_clips(voice_dir)
    clips_json = json.dumps(clips)
    total_audio = sum(c["dur"] for c in clips)
    print(f"clips={len(clips)} total_audio={total_audio:.1f}s  device={args.device}\n")
    print(f"{'model':>10} {'compute':>9} {'RTF':>8} {'WER':>7} {'mem(MB)':>9}")
    for m in args.models:
        for ct in args.compute_types:
            r = subprocess.run(
                [sys.executable, str(SCRIPT_DIR / "bench_stt.py"),
                 "--_worker", m, ct, clips_json, args.device],
                capture_output=True, text=True,
            )
            line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            try:
                d = json.loads(line)
                print(f"{m:>10} {ct:>9} {d['rtf']:>8.4f} {d['wer']:>7.3f} {d['rss_mb']:>9.0f}")
            except Exception:
                print(f"{m:>10} {ct:>9}   ERROR  {r.stderr.strip().splitlines()[-1:]}")


if __name__ == "__main__":
    main()
