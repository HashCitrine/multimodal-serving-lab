#!/usr/bin/env python3
"""Piper(ONNX) 로컬 음성 합성 CLI.

사용법:
    python synthesize.py --download                 # 보이스 모델 1회 내려받기
    python synthesize.py -t "hello world"           # 합성 → outputs/ 에 wav 저장
    python synthesize.py -t "..." --length-scale 1.1
각 실행은 RTF(real-time factor)를 함께 출력한다.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent

PIPER_KO_VOICE = "ko_KR-kss-medium"
PIPER_KO_FILES = {
    ".onnx": "https://huggingface.co/neurlang/piper-onnx-kss-korean/resolve/main/piper-kss-korean.onnx?download=true",
    ".onnx.json": "https://huggingface.co/neurlang/piper-onnx-kss-korean/raw/main/piper-kss-korean.onnx.json",
}


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dir(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def download_voice(cfg: dict) -> None:
    name = cfg["voice"]["name"]
    model_dir = resolve_dir(cfg["voice"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    if name == PIPER_KO_VOICE:
        print(f"[*] 한국어 커뮤니티 보이스 다운로드: {name} → {model_dir}")
        print("[!] 라이선스: neurlang/piper-onnx-kss-korean 은 cc-by-nc-sa-4.0 입니다.")
        for suffix, url in PIPER_KO_FILES.items():
            dst = model_dir / f"{name}{suffix}"
            print(f"    - {dst.name}")
            urllib.request.urlretrieve(url, dst)
        _require_voice_files(model_dir, name)
        print("[+] 완료")
        return

    from piper.download_voices import download_voice as _dl

    print(f"[*] 보이스 다운로드: {name} → {model_dir}")
    _dl(name, model_dir)
    _require_voice_files(model_dir, name)
    print("[+] 완료")


def _require_voice_files(model_dir: Path, name: str) -> None:
    missing = [model_dir / f"{name}{suffix}" for suffix in (".onnx", ".onnx.json")
               if not (model_dir / f"{name}{suffix}").exists()]
    if missing:
        raise FileNotFoundError("Piper 보이스 파일 누락:\n" + "\n".join(f"  - {p}" for p in missing))
    cfg_path = model_dir / f"{name}.onnx.json"
    try:
        phoneme_type = (json.loads(cfg_path.read_text(encoding="utf-8")).get("phoneme_type") or "espeak").lower()
    except Exception:
        phoneme_type = ""
    if phoneme_type not in ("espeak", "text"):
        raise ValueError(
            f"Piper 보이스 설정이 현재 piper 패키지와 호환되지 않습니다: {cfg_path} "
            f"(phoneme_type={phoneme_type!r}). 이 보이스는 s2s melo 경로에서 쓰지 않습니다."
        )


def load_voice(cfg: dict, use_cuda: bool):
    from piper import PiperVoice

    name = cfg["voice"]["name"]
    model_path = resolve_dir(cfg["voice"]["model_dir"]) / f"{name}.onnx"
    try:
        _require_voice_files(resolve_dir(cfg["voice"]["model_dir"]), name)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n먼저 `python synthesize.py --download`"
            f"{' --voice ko_KR-kss-medium' if name == PIPER_KO_VOICE else ''}"
        ) from e
    print(f"[*] 모델 로딩: {name}  (cuda={use_cuda})")
    return PiperVoice.load(str(model_path), use_cuda=use_cuda)


def synthesize(voice, text: str, cfg: dict):
    from piper import SynthesisConfig

    syn = cfg["synthesis"]
    sc = SynthesisConfig(
        length_scale=syn.get("length_scale", 1.0),
        noise_scale=syn.get("noise_scale", 0.667),
        noise_w_scale=syn.get("noise_w_scale", 0.8),
    )
    # 워밍업 (첫 호출 지연 흡수)
    list(voice.synthesize("warm up", sc))

    t0 = time.perf_counter()
    chunks = list(voice.synthesize(text, sc))
    synth_s = time.perf_counter() - t0

    audio = np.concatenate([c.audio_int16_array for c in chunks])
    return audio, synth_s


def save_wav(audio: np.ndarray, sample_rate: int, cfg: dict, out_path: str | None) -> Path:
    if out_path:
        path = Path(out_path)
    else:
        out_dir = resolve_dir(cfg["output"]["dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"{cfg['output']['filename_prefix']}_{ts}.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(audio.tobytes())
    return path


def parse_args():
    ap = argparse.ArgumentParser(description="Piper 음성 합성")
    ap.add_argument("-t", "--text", type=str, help="합성할 텍스트")
    ap.add_argument("--voice", type=str, help="보이스 이름(config 덮어쓰기)")
    ap.add_argument("--length-scale", type=float, dest="length_scale",
                    help="발화 속도(>1 느리게)")
    ap.add_argument("-o", "--output", type=str, help="출력 wav 경로")
    ap.add_argument("--download", action="store_true", help="보이스 모델만 내려받고 종료")
    ap.add_argument("--cuda", action="store_true", help="onnxruntime CUDA EP 사용")
    ap.add_argument("-c", "--config", type=str, default="config.yaml")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.voice:
        cfg["voice"]["name"] = args.voice
    if args.length_scale is not None:
        cfg["synthesis"]["length_scale"] = args.length_scale

    if args.download:
        download_voice(cfg)
        return

    use_cuda = args.cuda or cfg.get("device", {}).get("use_cuda", False)
    voice = load_voice(cfg, use_cuda)
    sr = voice.config.sample_rate

    text = args.text or cfg["synthesis"]["text"]
    audio, synth_s = synthesize(voice, text, cfg)
    audio_s = len(audio) / sr
    rtf = synth_s / audio_s if audio_s else float("nan")

    path = save_wav(audio, sr, cfg, args.output)
    print(f"[+] 저장: {path}")
    print(f"[metrics] audio={audio_s:.2f}s synth={synth_s:.3f}s "
          f"RTF={rtf:.3f} (x{1/rtf:.1f} realtime), sr={sr}, chars={len(text)}")


if __name__ == "__main__":
    main()
