#!/usr/bin/env python3
"""OpenAI 호환 LLM 채팅 CLI (provider 무관).

config.yaml 의 provider(base_url/model) 만 바꾸면 로컬 Ollama든 클라우드 vLLM이든 동일하게 동작.
스트리밍으로 TTFT(first token)와 decode tok/s 를 출력한다.

사용:
    python chat.py -p "Explain MLOps in one sentence."
    python chat.py -p "..." --model llama3.2:1b-instruct-q8_0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="OpenAI 호환 LLM 채팅")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("--model", help="config provider.model 덮어쓰기")
    ap.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    prov = cfg["provider"]
    model = args.model or prov["model"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))

    t0 = time.perf_counter()
    ttft = None
    n = 0
    print(f"[{model}] ", end="", flush=True)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": args.prompt}],
        stream=True, temperature=0, max_tokens=args.max_tokens,
    )
    for ch in stream:
        delta = ch.choices[0].delta.content
        if delta:
            if ttft is None:
                ttft = time.perf_counter() - t0
            n += 1
            print(delta, end="", flush=True)
    total = time.perf_counter() - t0
    decode = total - (ttft or 0)
    print(f"\n\n[metrics] TTFT={ttft*1000:.0f}ms  tokens≈{n}  "
          f"decode_tok/s≈{n/decode:.1f}  total={total:.2f}s")


if __name__ == "__main__":
    main()
