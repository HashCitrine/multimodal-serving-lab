# Experiment Notes

## Image generation

- Started with `diffusers` and Stable Diffusion v1.5 for local text-to-image generation.
- Tested anime/illustration-oriented models such as Anything V5 and Ghibli-style LoRA/model variants.
- Tried prompt and negative-prompt tuning for cat and character images, including seed-based retries.
- Added one-off generation and upscaling helpers after repeated manual runs.

## Video generation

- Tried a simple image-sequence approach: generate a keyframe, then create nearby frames with img2img and assemble them with ffmpeg.
- This preserved visual continuity better than fresh text-to-image frames, but it did not create real semantic motion. Walking prompts mostly became static or warped frames.
- Tested AnimateDiff to get actual frame-to-frame motion from a motion adapter.
- On a 16 GB Apple Silicon machine, practical settings had to be reduced to fewer frames and lower resolution. This improved memory use but reduced quality and subject consistency.
- Tested Zeroscope and LTX-style scripts as alternative text-to-video paths.

## Main takeaways

- Local image generation was usable for iteration and prompt experiments.
- Local video generation was much more constrained: memory pressure, low resolution, and weak subject consistency were recurring issues.
- img2img frame interpolation is useful for small camera-like changes, not for convincing action.
- Motion-adapter pipelines can create real motion, but quality depends heavily on VRAM/RAM, resolution, frame count, and the base model.
- For production-quality video, hosted services or larger GPU setups are more realistic than this local setup.
