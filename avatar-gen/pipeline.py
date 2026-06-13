#!/usr/bin/env python3
"""아바타 토킹헤드 end-to-end 파이프라인: text →(LLM)→ TTS →(lip-sync)→ mp4.

LLM은 OpenAI 호환 provider, TTS는 Piper, 립싱크는 교체형 backend를 사용한다.
`backend=static`은 정지 영상 폴백이고, `backend=wav2lip`은 외부 Wav2Lip repo/checkpoint를 호출한다.

사용:
    python pipeline.py --prompt "Greet a new English learner in one sentence." --face face.jpg
    python pipeline.py --text "Hello there!" --face face.jpg --backend wav2lip --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path
from typing import Any, Optional, Union


SCRIPT_DIR = Path(__file__).parent


def detect_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def resolve_path(value: Union[str, Path], base_dir: Path = SCRIPT_DIR) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_config(path: str) -> dict[str, Any]:
    import yaml

    p = resolve_path(path)
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def gen_text(cfg: dict[str, Any], prompt: str) -> str:
    from openai import OpenAI

    prov = cfg["llm"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))
    r = client.chat.completions.create(
        model=prov["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=prov.get("max_tokens", 80),
    )
    return r.choices[0].message.content.strip()


def synth_tts(cfg: dict[str, Any], text: str, out_wav: Path, use_cuda: bool) -> float:
    import numpy as np
    from piper import PiperVoice, SynthesisConfig

    voice_dir = resolve_path(cfg["tts"]["voice_dir"])
    voice = PiperVoice.load(str(voice_dir / f"{cfg['tts']['voice']}.onnx"), use_cuda=use_cuda)
    sr = voice.config.sample_rate
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return len(audio) / sr


def placeholder_face(path: Path) -> None:
    """얼굴 이미지를 안 줬을 때 static backend 검증용 더미 이미지 생성."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (40, 44, 52))
    d = ImageDraw.Draw(img)
    d.ellipse([146, 120, 366, 360], fill=(220, 200, 180))
    d.ellipse([200, 210, 240, 250], fill=(30, 30, 30))
    d.ellipse([272, 210, 312, 250], fill=(30, 30, 30))
    d.rectangle([225, 300, 287, 320], fill=(120, 60, 60))
    d.text((150, 380), "placeholder face", fill=(180, 180, 180))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_lipsync_backend(cfg: dict[str, Any], device: str):
    from backends import build_backend

    lip_cfg = dict(cfg["lipsync"])
    lip_cfg["device"] = device
    return build_backend(lip_cfg)


def run_pipeline(
    cfg: dict[str, Any],
    *,
    prompt: Optional[str] = None,
    text: Optional[str] = None,
    face: Optional[str] = None,
    backend_name: Optional[str] = None,
    device_pref: str = "auto",
    output: Union[str, Path] = "outputs/avatar.mp4",
    speech_name: str = "speech.wav",
) -> dict[str, Any]:
    if backend_name:
        cfg["lipsync"]["backend"] = backend_name

    device = detect_device(device_pref)
    tts_use_cuda = bool(cfg.get("tts", {}).get("use_cuda", False))
    out_dir = SCRIPT_DIR / "outputs"
    out_dir.mkdir(exist_ok=True)

    metrics: dict[str, Any] = {
        "device": device,
        "backend": cfg["lipsync"]["backend"],
        "tts_use_cuda": tts_use_cuda,
    }
    total_t0 = time.perf_counter()

    if text:
        spoken_text = text
        metrics["llm_seconds"] = 0.0
    elif prompt:
        t0 = time.perf_counter()
        spoken_text = gen_text(cfg, prompt)
        metrics["llm_seconds"] = time.perf_counter() - t0
    else:
        raise ValueError("prompt or text is required")
    metrics["text"] = spoken_text

    wav = out_dir / speech_name
    t0 = time.perf_counter()
    audio_seconds = synth_tts(cfg, spoken_text, wav, tts_use_cuda)
    metrics["tts_seconds"] = time.perf_counter() - t0
    metrics["audio_seconds"] = audio_seconds
    metrics["audio_path"] = str(wav)

    if face:
        face_path = str(resolve_path(face, Path.cwd()))
    else:
        placeholder = out_dir / "placeholder_face.png"
        placeholder_face(placeholder)
        face_path = str(placeholder)
    metrics["face_path"] = face_path

    backend = build_lipsync_backend(cfg, device)
    issues = backend.diagnostics()
    if issues:
        detail = "\n".join(f"- {issue}" for issue in issues)
        raise RuntimeError(f"backend '{backend.name}' is not ready:\n{detail}")

    out = resolve_path(output)
    t0 = time.perf_counter()
    backend.generate(face_path, str(wav), str(out))
    lipsync_seconds = time.perf_counter() - t0
    metrics["lipsync_seconds"] = lipsync_seconds
    metrics["lipsync_rtf"] = lipsync_seconds / audio_seconds if audio_seconds else None
    metrics["output_path"] = str(out)
    metrics["e2e_seconds"] = time.perf_counter() - total_t0
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"[*] device={metrics['device']}  lipsync backend={metrics['backend']}  tts_cuda={metrics['tts_use_cuda']}")
    print(f"[text] {metrics['text']}")
    if metrics["llm_seconds"]:
        print(f"[LLM] {metrics['llm_seconds']:.2f}s")
    print(f"[TTS] audio={metrics['audio_seconds']:.2f}s synth={metrics['tts_seconds']:.2f}s -> {metrics['audio_path']}")
    print(
        f"[lipsync:{metrics['backend']}] {metrics['lipsync_seconds']:.2f}s "
        f"RTF={metrics['lipsync_rtf']:.3f} -> {metrics['output_path']}"
    )
    print(f"[E2E] {metrics['e2e_seconds']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description="아바타 토킹헤드 파이프라인")
    ap.add_argument("--prompt", help="LLM에 줄 프롬프트 (응답을 말하게 함)")
    ap.add_argument("--text", help="LLM 건너뛰고 이 텍스트를 바로 말하게 함")
    ap.add_argument("--face", help="얼굴 이미지/영상 경로(없으면 static 검증용 placeholder 생성)")
    ap.add_argument("--backend", help="static | wav2lip | musetalk (config 덮어쓰기)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--json", action="store_true", help="metrics를 JSON으로 출력")
    ap.add_argument("-o", "--output", default="outputs/avatar.mp4")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    metrics = run_pipeline(
        cfg,
        prompt=args.prompt,
        text=args.text,
        face=args.face,
        backend_name=args.backend,
        device_pref=args.device,
        output=args.output,
    )
    if args.json:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print_metrics(metrics)


if __name__ == "__main__":
    main()
