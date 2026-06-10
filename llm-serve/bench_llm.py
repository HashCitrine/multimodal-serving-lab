#!/usr/bin/env python3
"""LLM 서빙 동시성 벤치 (OpenAI 호환 — provider 무관, 클라우드 포팅 가능).

동시성을 높이며 **집계 처리량(aggregate decode tok/s)**·TTFT·요청별 tok/s 를 잰다.
LLM은 배치 forward(연속 배칭)가 되므로 — Phase1(TTS 배칭 0×)·Phase2(STT int8 속도 0×)와 달리 —
동시성이 올라갈수록 집계 처리량이 오를 것으로 기대. 같은 스크립트를 vLLM endpoint(base_url)로만
바꿔 클라우드에서 재실행할 수 있다.

사용:
    python bench_llm.py --concurrency 1 2 4 8 --max-tokens 64
    python bench_llm.py --base-url http://<vllm>:8000/v1 --model meta-llama/Llama-3.2-1B-Instruct
"""
from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import yaml
from openai import AsyncOpenAI

SCRIPT_DIR = Path(__file__).parent
PROMPT = "Write a concise paragraph explaining what continuous batching is in LLM serving."


def load_config():
    return yaml.safe_load(open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8"))


async def one_request(client, model, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    n = 0
    stream = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROMPT}],
        stream=True, temperature=0, max_tokens=max_tokens,
    )
    async for ch in stream:
        d = ch.choices[0].delta.content
        if d:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
    return ttft or 0.0, n, time.perf_counter() - t0


async def run_level(client, model, concurrency, max_tokens):
    t0 = time.perf_counter()
    res = await asyncio.gather(*[one_request(client, model, max_tokens) for _ in range(concurrency)])
    wall = time.perf_counter() - t0
    total_tokens = sum(n for _, n, _ in res)
    per_req_ts = [n / (tot - ttft) if (tot - ttft) > 0 else 0 for ttft, n, tot in res]
    return {
        "concurrency": concurrency,
        "agg_tok_s": total_tokens / wall if wall else 0,
        "per_req_tok_s": statistics.median(per_req_ts),
        "ttft_ms": statistics.median(ttft for ttft, _, _ in res) * 1000,
    }


async def main_async(args):
    cfg = load_config()
    prov = cfg["provider"]
    base_url = args.base_url or prov["base_url"]
    model = args.model or prov["model"]
    client = AsyncOpenAI(base_url=base_url, api_key=prov.get("api_key", "EMPTY"))

    print(f"model={model}  base_url={base_url}  max_tokens={args.max_tokens}")
    await one_request(client, model, 8)  # warmup
    print(f"{'conc':>5} {'agg_tok/s':>10} {'req_tok/s':>10} {'TTFT_ms':>9}")
    for c in args.concurrency:
        r = await run_level(client, model, c, args.max_tokens)
        print(f"{r['concurrency']:>5} {r['agg_tok_s']:>10.1f} {r['per_req_tok_s']:>10.1f} {r['ttft_ms']:>9.0f}")


def parse_args():
    ap = argparse.ArgumentParser(description="OpenAI 호환 LLM 동시성 벤치")
    ap.add_argument("--base-url", dest="base_url", help="config 덮어쓰기 (vLLM 등)")
    ap.add_argument("--model")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=64, dest="max_tokens")
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
