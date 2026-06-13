"""Lightweight runtime readiness checks shared by S2S CLI/backends."""
from __future__ import annotations

import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(value: str | Path, base: Path = BASE_DIR) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (base / p).resolve()


def piper_voice_paths(voice_dir: str | Path, voice: str, base: Path = BASE_DIR) -> tuple[Path, Path]:
    root = resolve_path(voice_dir, base)
    return root / f"{voice}.onnx", root / f"{voice}.onnx.json"


def missing_piper_voice_files(voice_dir: str | Path, voice: str, base: Path = BASE_DIR) -> list[Path]:
    onnx, cfg = piper_voice_paths(voice_dir, voice, base)
    return [p for p in (onnx, cfg) if not p.exists()]


def piper_voice_config_issue(voice_dir: str | Path, voice: str, base: Path = BASE_DIR) -> str:
    _, cfg = piper_voice_paths(voice_dir, voice, base)
    if not cfg.exists():
        return ""
    try:
        phoneme_type = (json.loads(cfg.read_text(encoding="utf-8")).get("phoneme_type") or "espeak").lower()
    except Exception as e:
        return f"Piper voice config unreadable: {cfg} ({e})"
    if phoneme_type not in ("espeak", "text"):
        return f"Piper voice config unsupported by current piper package: {cfg} (phoneme_type={phoneme_type!r})"
    return ""


def piper_voice_hint(voice: str) -> str:
    return f"cd ../tts-gen && uv run python synthesize.py --download --voice {voice}"


def require_piper_voice(voice_dir: str | Path, voice: str, *, purpose: str, base: Path = BASE_DIR) -> Path:
    missing = missing_piper_voice_files(voice_dir, voice, base)
    onnx, _ = piper_voice_paths(voice_dir, voice, base)
    if missing:
        files = "\n".join(f"  - {p}" for p in missing)
        raise RuntimeError(
            f"{purpose} Piper voice is not ready: {voice}\n"
            f"Missing files:\n{files}\n"
            f"Prepare it with:\n  {piper_voice_hint(voice)}"
        )
    issue = piper_voice_config_issue(voice_dir, voice, base)
    if issue:
        raise RuntimeError(f"{purpose} Piper voice is not usable: {voice}\n{issue}")
    return onnx
