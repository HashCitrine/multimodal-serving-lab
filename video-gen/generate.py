#!/usr/bin/env python3
"""Local video generation using AnimateDiff or Zeroscope on Apple Silicon (MPS)."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import imageio
import torch
import yaml
from tqdm import tqdm


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_device(config: dict) -> torch.device:
    dtype = config["device"]["type"]
    if dtype == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(dtype)


def generate_animatediff(args, config: dict, device: torch.device):
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

    cfg = config["animatediff"]
    num_frames = args.frames if args.frames else cfg["num_frames"]
    width = args.width if args.width else cfg["width"]
    height = args.height if args.height else cfg["height"]
    steps = args.steps if args.steps else cfg["steps"]
    guidance = args.guidance if args.guidance else cfg["guidance_scale"]
    fps = args.fps if args.fps else cfg["fps"]

    base_model_cache = os.path.expanduser(cfg["base_model_cache"])
    motion_cache = os.path.expanduser(cfg["motion_adapter_cache"])

    print(f"[AnimateDiff] Loading motion adapter: {cfg['motion_adapter']}")
    adapter = MotionAdapter.from_pretrained(
        cfg["motion_adapter"],
        torch_dtype=torch.float32,
        cache_dir=motion_cache,
    )

    print(f"[AnimateDiff] Loading base model: {cfg['base_model']}")
    pipe = AnimateDiffPipeline.from_pretrained(
        cfg["base_model"],
        motion_adapter=adapter,
        torch_dtype=torch.float32,
        cache_dir=base_model_cache,
    )

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe = pipe.to(device)

    seed = args.seed if args.seed != -1 else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"[AnimateDiff] Seed: {seed}")
    print(f"[AnimateDiff] Generating {num_frames} frames at {width}x{height}, {steps} steps...")

    negative = args.negative or "bad quality, worst quality, low resolution, blurry"

    output = pipe(
        prompt=args.prompt,
        negative_prompt=negative,
        num_frames=num_frames,
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
    )

    frames = output.frames[0]  # list of PIL images
    return frames, fps, seed


def generate_zeroscope(args, config: dict, device: torch.device):
    from diffusers import DiffusionPipeline

    cfg = config["zeroscope"]
    num_frames = args.frames if args.frames else cfg["num_frames"]
    width = args.width if args.width else cfg["width"]
    height = args.height if args.height else cfg["height"]
    steps = args.steps if args.steps else cfg["steps"]
    guidance = args.guidance if args.guidance else cfg["guidance_scale"]
    fps = args.fps if args.fps else cfg["fps"]

    cache_dir = os.path.expanduser(cfg["cache_dir"])

    print(f"[Zeroscope] Loading model: {cfg['model_id']}")
    pipe = DiffusionPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )

    pipe.enable_attention_slicing(1)
    pipe.enable_vae_slicing()

    # MPS: cpu offload 미지원 → 직접 MPS로, CPU fallback
    try:
        pipe = pipe.to(device)
    except Exception:
        print(f"[Zeroscope] Failed to move to {device}, falling back to CPU")
        pipe = pipe.to("cpu")
        device = torch.device("cpu")

    seed = args.seed if args.seed != -1 else torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    print(f"[Zeroscope] Seed: {seed}")
    print(f"[Zeroscope] Generating {num_frames} frames at {width}x{height}, {steps} steps...")

    negative = args.negative or "bad quality, worst quality, low resolution, blurry"

    output = pipe(
        prompt=args.prompt,
        negative_prompt=negative,
        num_frames=num_frames,
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
    )

    # output.frames: numpy array shape (batch, frames, H, W, C) or list
    import numpy as np
    frames_raw = output.frames
    if isinstance(frames_raw, np.ndarray):
        # shape: (B, F, H, W, C)
        frames_np = frames_raw[0]  # (F, H, W, C)
        frames = [frames_np[i] for i in range(frames_np.shape[0])]
    elif isinstance(frames_raw, torch.Tensor):
        frames_np = frames_raw[0].cpu().numpy()
        frames = [frames_np[i] for i in range(frames_np.shape[0])]
    elif isinstance(frames_raw, list):
        inner = frames_raw[0]
        if isinstance(inner, list):
            frames = inner
        elif isinstance(inner, np.ndarray) and inner.ndim == 4:
            frames = [inner[i] for i in range(inner.shape[0])]
        else:
            frames = frames_raw
    else:
        frames = frames_raw

    return frames, fps, seed


def save_video(frames, fps: int, output_path: str):
    """Save frames as mp4 using imageio."""
    import numpy as np

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    print(f"Saving {len(frames)} frames to {output_path} at {fps} FPS...")
    for frame in tqdm(frames, desc="Writing frames"):
        if hasattr(frame, "convert"):
            # PIL Image
            frame_np = np.array(frame)
        elif isinstance(frame, np.ndarray):
            frame_np = frame
        else:
            frame_np = np.array(frame)

        # Ensure uint8
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)

        writer.append_data(frame_np)
    writer.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Local video generation (AnimateDiff / Zeroscope)")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument("-n", "--negative", type=str, default=None, help="Negative prompt")
    parser.add_argument("-m", "--model", type=str, choices=["animatediff", "zeroscope"],
                        default="animatediff", help="Model selection (default: animatediff)")
    parser.add_argument("--steps", type=int, default=None, help="Inference steps (default: 25)")
    parser.add_argument("--guidance", type=float, default=None, help="CFG scale (default: 7.5)")
    parser.add_argument("--width", type=int, default=None, help="Width in pixels")
    parser.add_argument("--height", type=int, default=None, help="Height in pixels")
    parser.add_argument("--frames", type=int, default=None, help="Number of frames")
    parser.add_argument("--fps", type=int, default=None, help="Output FPS (default: 8)")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1=random)")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output filename")
    parser.add_argument("--config", type=str, default=None, help="Config file path")

    args = parser.parse_args()

    # Find config file
    script_dir = Path(__file__).parent
    config_path = args.config or str(script_dir / "config.yaml")
    config = load_config(config_path)

    # Resolve device
    device = resolve_device(config)
    print(f"Device: {device}")

    # Output path
    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.expanduser(config["output"]["dir"])
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"{timestamp}.mp4")

    # Generate
    if args.model == "animatediff":
        frames, fps, seed = generate_animatediff(args, config, device)
    else:
        frames, fps, seed = generate_zeroscope(args, config, device)

    # Save
    save_video(frames, fps, output_path)
    print(f"\nDone! seed={seed}, output={output_path}")


if __name__ == "__main__":
    main()
