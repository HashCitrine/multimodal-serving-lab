#!/usr/bin/env python3
"""Ghibli-Diffusion - 지브리 본연의 단순하고 따뜻한 스타일"""
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from pathlib import Path

MODEL_ID  = "nitrosocke/Ghibli-Diffusion"
CACHE_DIR = str(Path(__file__).parent / "models")
OUT_DIR   = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    cache_dir=CACHE_DIR,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("mps")
pipe.enable_attention_slicing()

prompt = (
    "ghibli style, hayao miyazaki, "
    "simple face, soft expression, round eyes, "
    "gentle smile, clean line art, "
    "hand drawn animation, watercolor background, "
    "warm color palette, 1girl, brown hair"
)
negative_prompt = (
    "detailed shading, sharp features, anime eyes, "
    "glowing eyes, complex texture, realistic, "
    "3d, photorealistic, blurry, watermark"
)

seeds = [100, 200, 300, 400, 500]
print(f"[*] 지브리 본연 스타일 {len(seeds)}장...")

for seed in seeds:
    generator = torch.Generator("mps").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,
        height=640,
        num_inference_steps=50,
        guidance_scale=7.0,
        generator=generator,
    ).images[0]
    out_path = OUT_DIR / f"ghibli_pure_s{seed}.png"
    image.save(out_path)
    print(f"[+] {out_path}")

print("[+] 완료!")
