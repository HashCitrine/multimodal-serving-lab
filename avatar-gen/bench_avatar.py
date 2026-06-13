#!/usr/bin/env python3
"""Avatar/lip-sync backend 벤치.

같은 text/face 입력으로 static 또는 Wav2Lip backend의 end-to-end latency, lip-sync RTF,
GPU peak memory를 측정한다. 모델/체크포인트/출력 미디어는 Git에 포함하지 않는다.

사용:
    python bench_avatar.py --backend static --runs 3
    python bench_avatar.py --backend wav2lip --face /path/to/face.jpg --device cuda --runs 3
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from pipeline import load_config, run_pipeline

SCRIPT_DIR = Path(__file__).parent
DEFAULT_TEXTS = [
    "Hello, I am your tutor.",
    "Today we will practice short and natural English pronunciation.",
    "Please repeat after me and focus on the final consonant sound in each word.",
]


class GpuMemorySampler:
    def __init__(self, gpu_id: int = 0, interval_s: float = 0.2):
        self.gpu_id = gpu_id
        self.interval_s = interval_s
        self.peak_mb: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            value = self._query()
            if value is not None:
                self.peak_mb = value if self.peak_mb is None else max(self.peak_mb, value)
            time.sleep(self.interval_s)

    def _query(self) -> int | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_id}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except Exception:
            return None
        first = out.strip().splitlines()[0] if out.strip() else ""
        return int(first) if first.isdigit() else None


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def run_case(args: argparse.Namespace, cfg: dict[str, Any], text: str, index: int) -> dict[str, Any]:
    output = SCRIPT_DIR / "outputs" / f"bench_{args.backend}_{index}.mp4"
    speech_name = f"bench_{args.backend}_{index}.wav"
    return run_pipeline(
        cfg,
        text=text,
        face=args.face,
        backend_name=args.backend,
        device_pref=args.device,
        output=output,
        speech_name=speech_name,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Avatar/lip-sync latency and RTF benchmark")
    ap.add_argument("--backend", default="wav2lip", choices=["static", "wav2lip", "musetalk"])
    ap.add_argument("--face", help="립싱크 품질 검증용 실제 얼굴 이미지/영상 경로")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--text", action="append", help="측정 문장. 여러 번 지정 가능")
    args = ap.parse_args()

    texts = args.text or DEFAULT_TEXTS
    cfg = load_config(args.config)

    if args.backend in ("wav2lip", "musetalk") and not args.face:
        raise SystemExit(f"--backend {args.backend} requires --face with a real face image or video")

    print(f"backend={args.backend} device={args.device} runs={args.runs} warmup={args.warmup}")

    for i in range(args.warmup):
        run_case(args, cfg, texts[0], 10_000 + i)

    rows = []
    with GpuMemorySampler(gpu_id=args.gpu_id) as sampler:
        for run in range(args.runs):
            for idx, text in enumerate(texts):
                metrics = run_case(args, cfg, text, run * len(texts) + idx)
                rows.append(metrics)
                print(
                    f"run={run + 1} text={idx + 1} "
                    f"audio={metrics['audio_seconds']:.2f}s "
                    f"tts={metrics['tts_seconds']:.2f}s "
                    f"lip={metrics['lipsync_seconds']:.2f}s "
                    f"rtf={metrics['lipsync_rtf']:.3f} "
                    f"e2e={metrics['e2e_seconds']:.2f}s"
                )

    print("\nsummary")
    print(f"cases={len(rows)}")
    print(f"audio_s_median={median([r['audio_seconds'] for r in rows]):.2f}")
    print(f"tts_s_median={median([r['tts_seconds'] for r in rows]):.2f}")
    print(f"lipsync_s_median={median([r['lipsync_seconds'] for r in rows]):.2f}")
    print(f"lipsync_rtf_median={median([r['lipsync_rtf'] for r in rows]):.3f}")
    print(f"e2e_s_median={median([r['e2e_seconds'] for r in rows]):.2f}")
    if sampler.peak_mb is not None:
        print(f"gpu{args.gpu_id}_peak_mem_mb={sampler.peak_mb}")
    else:
        print(f"gpu{args.gpu_id}_peak_mem_mb=unavailable")


if __name__ == "__main__":
    main()
