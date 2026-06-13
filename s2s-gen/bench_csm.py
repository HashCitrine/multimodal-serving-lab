#!/usr/bin/env python3
"""csm-tts(상주 BentoML 서비스) RTF 벤치마크.

길이가 다른 문장을 csm-tts `/synthesize_meta` 로 합성하며 RTF·합성시간·오디오길이를 측정한다.
서비스가 모델을 **상주**시키므로(로드 비용 제외) 순수 합성 성능을 본다 —
tts-gen/bench_rtf.py(Piper) 와 같은 포맷이라 표현형 TTS(CSM) vs 경량 TTS(Piper) 를 직접 비교.

선행: csm-tts 서비스 기동
    CSM_DIR=../_external/csm uv run --extra csm bentoml serve bento_service:CSMTTS --port 3003

사용:
    uv run --extra csm python bench_csm.py                 # config.yaml csm_service_url
    uv run --extra csm python bench_csm.py --url http://127.0.0.1:3003 --runs 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

SENTENCES = [
    "Hello.",
    "This is a short sentence for benchmarking.",
    "Text to speech latency depends on input length, model size, and hardware backend.",
    ("Real time factor measures how long synthesis takes relative to the audio it "
     "produces; a value well below one means the model is faster than real time."),
]


def _meta(url: str, text: str) -> dict:
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(f"{url}/synthesize_meta", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser(description="csm-tts RTF 벤치")
    ap.add_argument("--url", help="csm-tts base URL(미지정 시 config.yaml csm_service_url)")
    ap.add_argument("--runs", type=int, default=3, help="문장별 반복 측정 수")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    url = args.url
    if not url:
        cfg = yaml.safe_load(open(SCRIPT_DIR / args.config, "r", encoding="utf-8")) or {}
        url = cfg.get("csm_service_url", "http://127.0.0.1:3003")
    url = url.rstrip("/")

    # 워밍업 + 서비스 확인
    try:
        warm = _meta(url, "warm up")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"csm-tts 서비스에 연결 실패({url}): {e}\n"
            "먼저 서비스를 기동하세요: "
            "CSM_DIR=../_external/csm uv run --extra csm bentoml serve bento_service:CSMTTS --port 3003"
        )

    print(f"url={url}  device={warm.get('device')}  runs={args.runs}\n")
    print(f"{'chars':>6} {'audio_s':>8} {'synth_s':>8} {'RTF':>7} {'xRT':>7}")
    all_rtf = []
    for text in SENTENCES:
        synth_times, audio_s = [], 0.0
        for _ in range(args.runs):
            m = _meta(url, text)
            synth_times.append(m["synth_seconds"])
            audio_s = m["audio_seconds"]
        synth_s = statistics.median(synth_times)
        rtf = synth_s / audio_s if audio_s else float("nan")
        all_rtf.append(rtf)
        print(f"{len(text):>6} {audio_s:>8.2f} {synth_s:>8.4f} {rtf:>7.4f} {1/rtf:>6.1f}x")

    print(f"\nmean RTF={statistics.mean(all_rtf):.4f}  "
          f"median RTF={statistics.median(all_rtf):.4f}  "
          f"(상주 서비스 — 모델 로드 비용 제외, 순수 합성)")


if __name__ == "__main__":
    main()
