# local-ai-gen-experiments

Local image and video generation experiments with Stable Diffusion, AnimateDiff, Zeroscope, and LTX-style pipelines.

This repository is an archive of scripts and notes from experiments on Apple Silicon/MPS. It is not a polished product package. Model weights, virtual environments, and generated outputs are intentionally excluded.

## Contents

- `sd-gen/`: text-to-image, ComfyUI-oriented, one-off generation, and upscaling scripts
- `video-gen/`: text-to-video experiments using AnimateDiff, Zeroscope, and LTX-style generation
- `docs/experiments.md`: notes about what was tried and what worked poorly or well

## Setup

Each folder has its own README and `requirements.txt`.

```bash
cd sd-gen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate.py -p "a cinematic mountain landscape at sunset"
```

For video experiments:

```bash
cd video-gen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate.py -p "a cat walking in a garden"
```

## Notes

- Model files are downloaded or cached locally under `models/` and are ignored by Git.
- Generated media goes under `outputs/` and is ignored by Git.
- Apple Silicon MPS worked more reliably with float32 for these experiments; fp16 often produced black frames or unstable output.
- Video quality was strongly limited by local memory, model choice, and the maturity of open text-to-video pipelines at the time of testing.
