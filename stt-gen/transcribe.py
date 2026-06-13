#!/usr/bin/env python3
"""faster-whisper(STT) 로컬 전사 CLI.

입력 오디오가 없으면 Piper(TTS)로 문장을 합성해 곧바로 전사하는 **TTS↔STT 왕복**을
지원한다(외부 오디오 불필요, 정답 텍스트를 알므로 WER도 계산).

사용법:
    python transcribe.py --from-tts "hello world"              # 합성→전사, RTF·WER 출력
    python transcribe.py -a clip.wav                            # 파일 전사
    python transcribe.py --from-tts "..." --compute-type float32
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def norm(s: str):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def _is_ascii_text(s: str) -> bool:
    """WER 계산은 [a-z0-9] 정규화 기반 — 한글/일본어 등은 토큰이 모두 사라져 무의미하다."""
    return s.isascii()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / max(1, len(r))


def tts_to_wav(text: str, cfg: dict) -> tuple[str, float]:
    from piper import PiperVoice, SynthesisConfig

    voice_dir = Path(cfg["audio"]["tts_voice_dir"])
    if not voice_dir.is_absolute():
        voice_dir = SCRIPT_DIR / voice_dir
    voice = PiperVoice.load(str(voice_dir / f"{cfg['audio']['tts_voice']}.onnx"))
    sr = voice.config.sample_rate
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(f.name, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return f.name, len(audio) / sr


def main():
    ap = argparse.ArgumentParser(description="faster-whisper 전사")
    ap.add_argument("-a", "--audio", type=str, help="입력 wav 경로")
    ap.add_argument("--from-tts", type=str, dest="from_tts", help="TTS로 합성 후 전사할 텍스트")
    ap.add_argument("--model", type=str, help="모델 (예: base.en, small.en, medium.en)")
    ap.add_argument("--compute-type", type=str, dest="compute_type",
                    help="CPU: int8|float32 / GPU: float16|int8_float16")
    ap.add_argument("--device", type=str, help="auto | cpu | cuda")
    ap.add_argument("--language", type=str,
                    help="전사 언어 (auto=자동감지 | ko | en | ja | zh ...). 한국어 등은 다국어 모델 필요")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model_name = args.model or cfg["model"]["name"]
    compute_type = args.compute_type or cfg["model"]["compute_type"]
    device = args.device or cfg["model"].get("device", "auto")
    beam = cfg["model"].get("beam_size", 1)
    language = args.language or cfg["model"].get("language", "auto")
    lang = None if (not language or language == "auto") else language

    from faster_whisper import WhisperModel
    print(f"[*] 모델 로딩: {model_name} (device={device}, compute_type={compute_type}, language={language})")
    if lang and model_name.endswith(".en"):
        print(f"[warn] '{model_name}' 은 영어 전용입니다 — language={lang} 전사를 위해 다국어 모델"
              f"(예: {model_name[:-3]} 또는 large-v3)을 사용하세요.")
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    ref = None
    if args.from_tts:
        audio_path, dur = tts_to_wav(args.from_tts, cfg)
        ref = args.from_tts
    elif args.audio:
        audio_path = args.audio
        try:
            with wave.open(audio_path, "rb") as w:
                dur = w.getnframes() / w.getframerate()
        except wave.Error as e:
            print(f"[warn] wav 메타데이터를 읽지 못했습니다: {e}")
            dur = 0.0
    else:
        raise SystemExit("입력이 필요합니다: -a <wav> 또는 --from-tts <text>")

    # 워밍업
    list(model.transcribe(audio_path, beam_size=beam, language=lang,
                          task="transcribe", condition_on_previous_text=False)[0])
    t0 = time.perf_counter()
    segs, info = model.transcribe(audio_path, beam_size=beam, language=lang,
                                  task="transcribe", condition_on_previous_text=False)
    text = "".join(s.text for s in segs).strip()
    dt = time.perf_counter() - t0

    print(f"[전사] {text}")
    print(f"[language] requested={language} detected={getattr(info, 'language', 'unknown')} "
          f"prob={getattr(info, 'language_probability', 0.0):.4f}")
    if dur:
        line = f"[metrics] audio={dur:.2f}s transcribe={dt:.3f}s RTF={dt/dur:.4f}"
    else:
        line = f"[metrics] audio=unknown transcribe={dt:.3f}s RTF=unknown"
    # WER 은 ASCII 정규화 기반이라 영어 외 언어에는 의미가 없어 생략한다.
    if ref is not None and _is_ascii_text(ref):
        line += f" WER={wer(ref, text):.3f}"
    print(line)


if __name__ == "__main__":
    main()
