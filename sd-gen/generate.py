#!/usr/bin/env python3
"""
Stable Diffusion 이미지 생성기
사용법: python generate.py -p "your prompt here"
"""

import argparse
import os
import sys
import random
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(config: dict):
    import torch

    pref = config["device"]["type"]
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return pref


def load_pipeline(config: dict, device: str):
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        DPMSolverMultistepScheduler,
    )

    model_id = config["model"]["id"]
    cache_dir = config["model"]["cache_dir"]
    use_fp16 = config["device"]["fp16"] and device in ("cuda", "mps")
    dtype = torch.float16 if use_fp16 else torch.float32

    print(f"[*] 모델 로딩: {model_id}  (device={device}, dtype={dtype})")

    # SDXL 모델 자동 감지
    is_sdxl = "xl" in model_id.lower()
    PipelineClass = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline

    pipe = PipelineClass.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
        safety_checker=None,       # NSFW 필터 끄려면 유지, 켜려면 제거
        requires_safety_checker=False,
    )

    # DPM++ 2M 스케줄러로 교체 (품질/속도 균형 좋음)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)

    # 메모리 최적화
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention()
    if device == "cpu":
        pipe.enable_attention_slicing()

    return pipe


def generate(pipe, config: dict, args) -> list:
    import torch

    gen_cfg = config["generation"]

    prompt = args.prompt or gen_cfg["prompt"]
    negative_prompt = args.negative_prompt or gen_cfg["negative_prompt"]
    width = args.width or gen_cfg["width"]
    height = args.height or gen_cfg["height"]
    steps = args.steps or gen_cfg["steps"]
    guidance = args.guidance or gen_cfg["guidance_scale"]
    num_images = args.num_images or gen_cfg["num_images"]
    seed = args.seed if args.seed is not None else gen_cfg["seed"]

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    print(f"[*] 프롬프트: {prompt}")
    print(f"[*] 네거티브: {negative_prompt}")
    print(f"[*] 크기: {width}x{height}  스텝: {steps}  CFG: {guidance}  시드: {seed}")

    generator = torch.Generator().manual_seed(seed)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        num_images_per_prompt=num_images,
        generator=generator,
    ).images

    return images, seed, prompt


def save_images(images, config: dict, seed: int, prompt: str):
    from PIL import PngImagePlugin

    out_cfg = config["output"]
    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = out_cfg["filename_prefix"]
    fmt = out_cfg["format"].lower()
    save_meta = out_cfg["save_metadata"]

    saved_paths = []
    for i, img in enumerate(images):
        suffix = f"_{i+1}" if len(images) > 1 else ""
        filename = f"{prefix}_{timestamp}{suffix}.{fmt}"
        filepath = out_dir / filename

        if fmt == "png" and save_meta:
            meta = PngImagePlugin.PngInfo()
            meta.add_text("prompt", prompt)
            meta.add_text("seed", str(seed))
            meta.add_text("model", config["model"]["id"])
            img.save(filepath, pnginfo=meta)
        else:
            img.save(filepath)

        print(f"[+] 저장됨: {filepath}")
        saved_paths.append(str(filepath))

    return saved_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Stable Diffusion 이미지 생성기")
    parser.add_argument("-p", "--prompt", type=str, help="생성 프롬프트")
    parser.add_argument("-n", "--negative-prompt", type=str, dest="negative_prompt",
                        help="네거티브 프롬프트")
    parser.add_argument("-W", "--width", type=int, help="이미지 너비")
    parser.add_argument("-H", "--height", type=int, help="이미지 높이")
    parser.add_argument("-s", "--steps", type=int, help="추론 스텝 수")
    parser.add_argument("-g", "--guidance", type=float, help="CFG scale")
    parser.add_argument("-N", "--num-images", type=int, dest="num_images",
                        help="생성할 이미지 수")
    parser.add_argument("--seed", type=int, help="고정 시드 (-1=랜덤)")
    parser.add_argument("-c", "--config", type=str, default="config.yaml",
                        help="설정 파일 경로")
    return parser.parse_args()


def main():
    args = parse_args()

    # config.yaml 위치를 스크립트 디렉토리 기준으로
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    config = load_config(str(config_path))

    # 출력/모델 경로를 스크립트 기준 상대경로로 해석
    for key in ("cache_dir",):
        p = config["model"][key]
        if not Path(p).is_absolute():
            config["model"][key] = str(script_dir / p)
    p = config["output"]["dir"]
    if not Path(p).is_absolute():
        config["output"]["dir"] = str(script_dir / p)

    device = get_device(config)
    pipe = load_pipeline(config, device)

    images, seed, prompt = generate(pipe, config, args)
    save_images(images, config, seed, prompt)

    print("[+] 완료!")


if __name__ == "__main__":
    main()
