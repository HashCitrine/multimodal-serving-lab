#!/usr/bin/env python3
"""아바타 토킹헤드 end-to-end 파이프라인: text →(LLM)→ TTS →(lip-sync)→ mp4.

각 단계는 앞 Phase의 자산을 재사용한다:
  - LLM:  llm-serve 의 OpenAI 호환 provider (로컬 Ollama ↔ 클라우드 vLLM, base_url 교체)
  - TTS:  tts-gen 의 Piper 보이스
  - 립싱크: backends/ (static=ffmpeg 폴백 / wav2lip=외부 모델)
device 는 auto 감지(cuda→mps→cpu). 무거운 립싱크 모델이 없어도 backend=static 으로 전 구간 검증된다.

사용:
    python pipeline.py --prompt "Greet a new English learner in one sentence." --face face.jpg
    python pipeline.py --text "Hello there!" --face face.jpg --backend static
"""
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import yaml

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


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def gen_text(cfg: dict, prompt: str) -> str:
    from openai import OpenAI

    prov = cfg["llm"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))
    r = client.chat.completions.create(
        model=prov["model"], messages=[{"role": "user", "content": prompt}],
        temperature=0.7, max_tokens=cfg["llm"].get("max_tokens", 80),
    )
    return r.choices[0].message.content.strip()


def synth_tts(cfg: dict, text: str, out_wav: Path, use_cuda: bool) -> float:
    from piper import PiperVoice, SynthesisConfig

    voice_dir = Path(cfg["tts"]["voice_dir"])
    if not voice_dir.is_absolute():
        voice_dir = SCRIPT_DIR / voice_dir
    voice = PiperVoice.load(str(voice_dir / f"{cfg['tts']['voice']}.onnx"), use_cuda=use_cuda)
    sr = voice.config.sample_rate
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return len(audio) / sr


def placeholder_face(path: Path) -> None:
    """얼굴 이미지를 안 줬을 때 파이프라인 검증용 더미 이미지 생성."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (512, 512), (40, 44, 52))
    d = ImageDraw.Draw(img)
    d.ellipse([146, 120, 366, 360], fill=(220, 200, 180))   # 얼굴
    d.ellipse([200, 210, 240, 250], fill=(30, 30, 30))      # 눈
    d.ellipse([272, 210, 312, 250], fill=(30, 30, 30))
    d.rectangle([225, 300, 287, 320], fill=(120, 60, 60))   # 입
    d.text((150, 380), "placeholder face", fill=(180, 180, 180))
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description="아바타 토킹헤드 파이프라인")
    ap.add_argument("--prompt", help="LLM에 줄 프롬프트 (응답을 말하게 함)")
    ap.add_argument("--text", help="LLM 건너뛰고 이 텍스트를 바로 말하게 함")
    ap.add_argument("--face", help="얼굴 이미지 경로(없으면 placeholder 생성)")
    ap.add_argument("--backend", help="static | wav2lip (config 덮어쓰기)")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("-o", "--output", default="outputs/avatar.mp4")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.backend:
        cfg["lipsync"]["backend"] = args.backend
    device = detect_device(args.device)
    use_cuda = device == "cuda"
    out_dir = (SCRIPT_DIR / "outputs"); out_dir.mkdir(exist_ok=True)

    print(f"[*] device={device}  lipsync backend={cfg['lipsync']['backend']}")

    # 1) text
    if args.text:
        text = args.text
    elif args.prompt:
        t0 = time.perf_counter()
        text = gen_text(cfg, args.prompt)
        print(f"[LLM] {text}   ({time.perf_counter()-t0:.2f}s)")
    else:
        raise SystemExit("--prompt 또는 --text 필요")

    # 2) TTS
    wav = out_dir / "speech.wav"
    t0 = time.perf_counter()
    dur = synth_tts(cfg, text, wav, use_cuda)
    print(f"[TTS] {dur:.2f}s 음성 → {wav}   ({time.perf_counter()-t0:.2f}s)")

    # 3) face
    if args.face:
        face = args.face
    else:
        face = str(out_dir / "placeholder_face.png")
        placeholder_face(Path(face))
        print(f"[face] 얼굴 미지정 → placeholder 생성: {face}")

    # 4) lip-sync
    from backends import build_backend

    backend = build_backend(cfg["lipsync"])
    if not backend.available():
        raise SystemExit(
            f"backend '{backend.name}' 사용 불가(의존성/체크포인트 확인). "
            f"우선 --backend static 으로 파이프라인을 검증하세요."
        )
    out = Path(args.output)
    if not out.is_absolute():
        out = SCRIPT_DIR / out
    t0 = time.perf_counter()
    backend.generate(face, str(wav), str(out))
    print(f"[lipsync:{backend.name}] → {out}   ({time.perf_counter()-t0:.2f}s)")
    print(f"[+] 완료: {out}")


if __name__ == "__main__":
    main()
