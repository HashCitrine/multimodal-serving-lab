#!/usr/bin/env python3
"""벤치마크 하니스 — 동시성 sweep로 지연/처리량 측정.

서버(/infer)를 동시성 수준별로 부하 주며 per-request latency 분포(p50/p95/p99),
throughput(req/s), 서버가 본 평균 배치 크기를 측정한다. 모든 서브 프로젝트가
재사용한다(모달리티별 지표 RTF/tokens·sec는 각 프로젝트에서 확장).

사용:
    python bench.py --url http://127.0.0.1:8000 --concurrency 1 2 4 8 --requests 64
"""
from __future__ import annotations

import argparse
import asyncio
import time
from typing import List

import httpx


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


async def _worker(client, url, path, field, payload, n, latencies):
    for _ in range(n):
        t0 = time.perf_counter()
        r = await client.post(f"{url}{path}", json={field: payload}, timeout=60.0)
        r.raise_for_status()
        latencies.append((time.perf_counter() - t0) * 1000.0)


async def run_level(url, path, field, concurrency: int, total_requests: int, payload: str) -> dict:
    per = max(1, total_requests // concurrency)
    latencies: List[float] = []
    async with httpx.AsyncClient() as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[
            _worker(client, url, path, field, payload, per, latencies) for _ in range(concurrency)
        ])
        wall = time.perf_counter() - t0
    done = len(latencies)
    return {
        "concurrency": concurrency,
        "requests": done,
        "wall_s": round(wall, 3),
        "throughput_rps": round(done / wall, 1) if wall > 0 else 0.0,
        "p50_ms": round(percentile(latencies, 50), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
    }


async def main_async(args):
    # warmup
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{args.url}{args.path}", json={args.field: "warmup"}, timeout=60.0)
        except Exception as e:
            print(f"[bench] warmup 실패 — 서버가 떠 있나요? {e}")
            return

    print(f"target: {args.url}{args.path}  field={args.field!r}")
    print(f"{'conc':>5} {'reqs':>6} {'wall_s':>8} {'rps':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    rows = []
    for c in args.concurrency:
        row = await run_level(args.url, args.path, args.field, c, args.requests, args.payload)
        rows.append(row)
        print(f"{row['concurrency']:>5} {row['requests']:>6} {row['wall_s']:>8} "
              f"{row['throughput_rps']:>8} {row['p50_ms']:>8} {row['p95_ms']:>8} {row['p99_ms']:>8}")

    # 서버가 관측한 배치 메트릭
    async with httpx.AsyncClient() as client:
        try:
            m = (await client.get(f"{args.url}/metrics")).json()
            print(f"\n[server metrics] avg_batch_size={m.get('avg_batch_size')} "
                  f"max_observed_batch={m.get('max_observed_batch')} "
                  f"total_batches={m.get('total_batches')}")
        except Exception:
            pass


def parse_args():
    ap = argparse.ArgumentParser(description="serving spine 벤치마크")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--path", default="/infer", help="요청 경로 (BentoML: /synthesize_meta)")
    ap.add_argument("--field", default="input", help="JSON 입력 필드명 (BentoML: text)")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--requests", type=int, default=64, help="동시성 수준별 총 요청 수")
    ap.add_argument("--payload", default="hello serving spine")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
