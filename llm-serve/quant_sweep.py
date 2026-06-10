#!/usr/bin/env python3
"""LLM 양자화 스윕 (로컬 Ollama) — Q4/Q8/FP16 의 tok/s·메모리·디스크·품질.

핵심 질문(Phase 2와 대비): LLM에서 양자화는 메모리뿐 아니라 **속도(decode tok/s)**까지 올리는가?
(LLM decode는 메모리 대역폭 바운드라, 더 작은 가중치 = 더 빠른 디코딩이 되는 경향.)
품질은 fp16 출력과의 일치도로 근사(temp=0, 같은 프롬프트).

정밀 timing/메모리는 Ollama 네이티브 API(/api/generate, /api/ps, /api/tags)를 쓴다.
사용: python quant_sweep.py
"""
from __future__ import annotations

import re
import statistics
from pathlib import Path

import httpx
import yaml

SCRIPT_DIR = Path(__file__).parent

PROMPTS = [
    "List three benefits of unit testing.",
    "What is the capital of France? Answer in one short sentence.",
    "Write a haiku about servers.",
    "Explain what a hash map is in two sentences.",
]


def load_config():
    return yaml.safe_load(open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8"))


def disk_size_gb(base, model):
    r = httpx.get(f"{base}/api/tags", timeout=30).json()
    for m in r.get("models", []):
        if m["name"] == model:
            return m["size"] / 1e9
    return None


def mem_gb(base, model):
    r = httpx.get(f"{base}/api/ps", timeout=30).json()
    for m in r.get("models", []):
        if m["name"] == model:
            return m.get("size", 0) / 1e9, m.get("size_vram", 0) / 1e9
    return None, None


def generate(base, model, prompt, num_predict=128):
    return httpx.post(f"{base}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict, "seed": 0},
    }, timeout=180).json()


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def agreement(a, b):
    """단어 단위 일치도(작은 편집거리 기반 유사도, 0~1)."""
    ra, rb = norm(a), norm(b)
    d = [[0] * (len(rb) + 1) for _ in range(len(ra) + 1)]
    for i in range(len(ra) + 1):
        d[i][0] = i
    for j in range(len(rb) + 1):
        d[0][j] = j
    for i in range(1, len(ra) + 1):
        for j in range(1, len(rb) + 1):
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+(ra[i-1] != rb[j-1]))
    dist = d[len(ra)][len(rb)]
    return 1 - dist / max(1, max(len(ra), len(rb)))


def main():
    cfg = load_config()
    base = cfg["ollama_base"]
    tags = cfg["quant_tags"]

    # fp16(마지막 태그)을 품질 기준으로
    ref_tag = tags[-1]
    ref_out = {}
    print("[*] 기준(fp16) 출력 생성 중...")
    for p in PROMPTS:
        ref_out[p] = generate(base, ref_tag, p, num_predict=80).get("response", "")

    print(f"\n{'quant':>28} {'disk(GB)':>9} {'mem(GB)':>8} {'prefill t/s':>12} "
          f"{'decode t/s':>11} {'qual(vs fp16)':>13}")
    for tag in tags:
        # 디코딩 tok/s: 긴 생성 1회 + 정밀 timing
        decode_ts, prefill_ts = [], []
        for _ in range(3):
            r = generate(base, tag, "Write a short paragraph about distributed systems.", 128)
            if r.get("eval_duration"):
                decode_ts.append(r["eval_count"] / r["eval_duration"] * 1e9)
            if r.get("prompt_eval_duration"):
                prefill_ts.append(r.get("prompt_eval_count", 0) / r["prompt_eval_duration"] * 1e9)
        mem, _vram = mem_gb(base, tag)
        disk = disk_size_gb(base, tag)
        # 품질: fp16 대비 일치도
        quals = []
        for p in PROMPTS:
            out = generate(base, tag, p, num_predict=80).get("response", "")
            quals.append(agreement(out, ref_out[p]))
        print(f"{tag:>28} {disk:>9.2f} {mem:>8.2f} "
              f"{statistics.median(prefill_ts):>12.0f} {statistics.median(decode_ts):>11.1f} "
              f"{statistics.mean(quals):>13.2f}")


if __name__ == "__main__":
    main()
