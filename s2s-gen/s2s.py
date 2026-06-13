#!/usr/bin/env python3
"""S2S(speech-to-speech) 단일 턴 CLI — 캐스케이드와 음성 네이티브 모델을 같은 인터페이스로 실행.

입력 음성(wav) → 선택한 backend → 응답 음성(wav). 단계 경계가 없는 S2S와, STT→LLM→TTS
캐스케이드를 '같은 in_wav→out_wav+metrics' 형태로 돌려 지연/관찰가능성 트레이드오프를 본다.

마이크 없이 자체 검증: --ask "질문" 이면 Piper로 질문 음성을 합성해 입력으로 쓴다.

사용:
    python s2s.py --backend cascade --ask "How do you spell necessary?"
    python s2s.py --backend moshi   --audio question.wav
    python s2s.py --backend csm     --audio question.wav --device cuda
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import yaml

from backends import build_backend

SCRIPT_DIR = Path(__file__).parent


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def synth_question(cfg: dict, text: str, out: Path) -> float:
    """Piper로 질문 음성을 합성(마이크 입력 대체). 반환: 음성 길이(초)."""
    from piper import PiperVoice, SynthesisConfig
    tts = cfg.get("tts", {})
    voice = PiperVoice.load(str(resolve(tts.get("voice_dir", "../tts-gen/models")) / f"{tts.get('voice', 'en_US-lessac-medium')}.onnx"))
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    sr = voice.config.sample_rate
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
    return len(audio) / sr


def main():
    ap = argparse.ArgumentParser(description="S2S 단일 턴 (cascade|moshi|csm)")
    ap.add_argument("--backend", choices=["cascade", "csm"], help="config의 backend 덮어쓰기 (moshi는 라이브 전용 — lab-ui Moshi 카드)")
    ap.add_argument("--ask", help="질문 텍스트(Piper로 음성 합성해 입력으로 사용)")
    ap.add_argument("--audio", help="질문 음성 wav 직접 입력")
    ap.add_argument("--device", help="auto|cpu|cuda|mps (config 덮어쓰기)")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    if not args.ask and not args.audio:
        raise SystemExit("--ask <text> 또는 --audio <wav> 필요")

    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    if args.device:
        cfg["device"] = args.device

    out_dir = (SCRIPT_DIR / "outputs"); out_dir.mkdir(exist_ok=True)

    # 0) 입력 음성 준비 (마이크 대체)
    if args.audio:
        in_wav = args.audio
        with wave.open(in_wav, "rb") as w:
            in_dur = w.getnframes() / w.getframerate()
    else:
        in_wav = str(out_dir / "question.wav")
        in_dur = synth_question(cfg, args.ask, Path(in_wav))

    backend = build_backend(cfg)
    if not backend.available():
        print(f"[{backend.name}] 백엔드가 아직 준비되지 않았습니다:")
        for issue in backend.diagnostics():
            print(f"  - {issue}")
        raise SystemExit(1)

    out_wav = str(out_dir / f"answer_{backend.name}.wav")
    m = backend.generate(in_wav, out_wav)

    print(f"\n[backend] {backend.name}   [입력 음성] {in_dur:.2f}s")
    if m.get("question"):
        print(f"[인식] \"{m['question']}\"")
    if m.get("text"):
        print(f"[응답] \"{m['text']}\"")
    print(f"[응답 음성] {m.get('out_s', 0):.2f}s → {out_wav}")
    print("\n=== 지연 예산 (한 턴) ===")
    if "stt_s" in m:  # cascade는 단계별 분해 제공
        print(f"  STT          : {m['stt_s']*1000:6.0f} ms")
        print(f"  LLM TTFT     : {m['llm_ttft_s']*1000:6.0f} ms")
        print(f"  LLM total    : {m['llm_s']*1000:6.0f} ms")
        print(f"  TTS          : {m['tts_s']*1000:6.0f} ms")
        print(f"  ─────────────────────────")
    print(f"  TTFA         : {m.get('ttfa_s', 0)*1000:6.0f} ms  (첫 응답 오디오까지)")
    print(f"  E2E          : {m.get('e2e_s', 0)*1000:6.0f} ms")
    print(f"  RTF          : {m.get('rtf', 0):.3f}  (e2e/응답길이, 작을수록 빠름)")


if __name__ == "__main__":
    main()
