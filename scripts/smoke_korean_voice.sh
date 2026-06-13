#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/s2s-gen"
uv run \
  --with "melotts @ git+https://github.com/myshell-ai/MeloTTS.git" \
  --with "unidic-lite>=1.0.8" \
  --with "python-mecab-ko>=1.3.7" \
  python - <<'PY'
from pathlib import Path
import yaml
import wave

from melo_runtime import prepare_unidic_lite
from backends import build_backend

cfg = yaml.safe_load(Path("config.yaml").read_text())
cfg["backend"] = "melo"
cfg["language"] = "ko"

prepare_unidic_lite()
from melo.api import TTS  # noqa: F401

model = TTS(language="KR", device=cfg.get("melo_device", "auto"))
speakers = model.hps.data.spk2id
out = Path("/tmp/melo_ko_smoke.wav")
model.tts_to_file("안녕하세요.", speakers["KR"], str(out), quiet=True)
with wave.open(str(out), "rb") as w:
    if w.getnframes() <= 0:
        raise SystemExit("Melo Korean synthesis produced an empty wav")

backend = build_backend(cfg)
issues = backend.diagnostics()
if issues:
    raise SystemExit("Melo backend diagnostics failed:\n" + "\n".join(f"  - {i}" for i in issues))
print("Korean Melo smoke passed: Melo import/UniDic, backend diagnostics")
PY
