#!/usr/bin/env python3
"""
LTX-Video text-to-video generation on Apple Silicon (MPS).
Uses Lightricks/LTX-Video via HuggingFace Diffusers.
Target: 2-second video (49 frames @ 24fps), 512x768 resolution.
"""

import os
import sys
import time
import torch
from pathlib import Path
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
PROMPT = (
    "A stylish young woman walking along a busy New York City sidewalk in daytime, "
    "smooth continuous walking motion, yellow taxis passing by, urban skyscrapers, "
    "busy street crowd, camera follows from behind slightly, cinematic, photorealistic, "
    "4k quality, natural lighting, dynamic motion"
)
NEGATIVE_PROMPT = (
    "static image, no motion, freeze, blurry, distortion, low quality, "
    "multiple people overlapping, watermark, text, logo, bad anatomy, "
    "deformed, ugly, worst quality, cartoon, anime"
)

NUM_FRAMES    = 49      # 8*6+1 — LTX-Video 요구사항
FPS           = 24
WIDTH         = 512
HEIGHT        = 768
GUIDANCE      = 3.0     # LTX-Video 권장값
STEPS         = 25
SEED          = 42

CACHE_DIR     = Path(__file__).parent / "models" / "ltx-video"
OUTPUT_DIR    = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
    dtype  = torch.float32   # MPS는 bfloat16 미지원, float32 사용
    print(f"[Device] Apple Silicon MPS (float32)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    dtype  = torch.bfloat16
    print(f"[Device] CUDA (bfloat16)")
else:
    device = torch.device("cpu")
    dtype  = torch.float32
    print(f"[Device] CPU (float32) — 느릴 수 있음")

# ── Load pipeline ───────────────────────────────────────────────────────────────
print(f"\n[Model] LTX-Video 로딩 중 (첫 실행 시 ~6GB 다운로드)...")
t0 = time.time()

from diffusers import LTXPipeline

pipe = LTXPipeline.from_pretrained(
    "Lightricks/LTX-Video",
    torch_dtype=dtype,
    cache_dir=str(CACHE_DIR),
)

print(f"[Model] 로드 완료 ({time.time()-t0:.1f}s)")

# LTX-Video (~18GB) 는 MPS 전체 로드 불가 → CPU offload 방식 사용
# 레이어별로 CPU↔MPS/CPU 간 이동하며 메모리 절약
if device.type == "mps":
    print("[Offload] MPS 메모리 부족으로 CPU offload 모드 사용")
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

# ── Generate ────────────────────────────────────────────────────────────────────
print(f"\n[Generate] 설정:")
print(f"  Frames:    {NUM_FRAMES} ({NUM_FRAMES/FPS:.1f}초 @ {FPS}fps)")
print(f"  Size:      {WIDTH}x{HEIGHT}")
print(f"  Steps:     {STEPS}")
print(f"  Guidance:  {GUIDANCE}")
print(f"  Seed:      {SEED}")
print(f"  Prompt:    {PROMPT[:80]}...")
print()

generator = torch.Generator(device="cpu").manual_seed(SEED)
t1 = time.time()

output = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    num_frames=NUM_FRAMES,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=GUIDANCE,
    num_inference_steps=STEPS,
    generator=generator,
)

elapsed = time.time() - t1
print(f"\n[Generate] 완료 ({elapsed:.1f}s)")

# ── Export ──────────────────────────────────────────────────────────────────────
from diffusers.utils import export_to_video

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path  = OUTPUT_DIR / f"ltx_nyc_woman_{timestamp}.mp4"

frames = output.frames[0]
export_to_video(frames, str(out_path), fps=FPS)

print(f"\n[Output] 저장 완료: {out_path}")
print(f"[Output] 총 소요: {(time.time()-t0):.1f}s")
print(f"RESULT_PATH:{out_path}")
