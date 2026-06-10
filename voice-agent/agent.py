#!/usr/bin/env python3
"""음성 에이전트: speech-in → STT → LLM → TTS → speech-out.

앞 Phase 자산을 한 대화 턴으로 묶고, **단계별 지연 예산(latency budget)**을 측정한다 —
대화형 음성 응답에서 어디가 병목인지(STT vs LLM prefill/decode vs TTS) 보는 새 최적화 축.

마이크 없이 자체 검증: --ask "질문" 이면 Piper로 질문 음성을 합성해 입력으로 쓴다(전 구간 왕복).

사용:
    python agent.py --ask "How do I say hello in Korean?"
    python agent.py --audio question.wav
"""
from __future__ import annotations

import argparse
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


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def piper_voice(cfg, use_cuda=False):
    from piper import PiperVoice
    return PiperVoice.load(str(resolve(cfg["tts"]["voice_dir"]) / f"{cfg['tts']['voice']}.onnx"),
                           use_cuda=use_cuda)


def tts_to_wav(voice, text: str, out: Path) -> float:
    from piper import SynthesisConfig
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    sr = voice.config.sample_rate
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
    return len(audio) / sr


def main():
    ap = argparse.ArgumentParser(description="STT→LLM→TTS 음성 에이전트")
    ap.add_argument("--ask", help="질문 텍스트(Piper로 음성 합성해 입력으로 사용)")
    ap.add_argument("--audio", help="질문 음성 wav 직접 입력")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    if not args.ask and not args.audio:
        raise SystemExit("--ask <text> 또는 --audio <wav> 필요")

    cfg = load_config(args.config)
    out_dir = (SCRIPT_DIR / "outputs"); out_dir.mkdir(exist_ok=True)
    voice = piper_voice(cfg)

    # 0) 입력 음성 준비 (마이크 대체)
    if args.audio:
        in_wav = args.audio
        with wave.open(in_wav, "rb") as w:
            in_dur = w.getnframes() / w.getframerate()
    else:
        in_wav = str(out_dir / "question.wav")
        in_dur = tts_to_wav(voice, args.ask, Path(in_wav))

    budget = {}

    # 1) STT
    from faster_whisper import WhisperModel
    stt = WhisperModel(cfg["stt"]["model"], device=cfg["stt"].get("device", "auto"),
                       compute_type=cfg["stt"].get("compute_type", "int8"))
    list(stt.transcribe(in_wav, beam_size=1)[0])  # warmup
    t0 = time.perf_counter()
    segs, _ = stt.transcribe(in_wav, beam_size=1)
    question = "".join(s.text for s in segs).strip()
    budget["stt_s"] = time.perf_counter() - t0

    # 2) LLM (TTFT + 디코딩 분리 측정)
    from openai import OpenAI
    prov = cfg["llm"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))
    sys_prompt = cfg["llm"].get("system", "You are a concise, friendly English tutor. Answer in 1-2 sentences.")
    # 워밍업: 모델을 메모리에 올려 '대화 중 정상상태' 지연을 측정(첫 턴 콜드스타트 제외)
    client.chat.completions.create(model=prov["model"],
        messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0)
    t0 = time.perf_counter(); ttft = None; answer = ""
    stream = client.chat.completions.create(
        model=prov["model"],
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": question}],
        stream=True, temperature=0.5, max_tokens=prov.get("max_tokens", 80))
    for ch in stream:
        d = ch.choices[0].delta.content
        if d:
            if ttft is None:
                ttft = time.perf_counter() - t0
            answer += d
    budget["llm_ttft_s"] = ttft or 0.0
    budget["llm_total_s"] = time.perf_counter() - t0

    # 3) TTS
    ans_wav = out_dir / "answer.wav"
    t0 = time.perf_counter()
    ans_dur = tts_to_wav(voice, answer.strip(), ans_wav)
    budget["tts_s"] = time.perf_counter() - t0

    total = budget["stt_s"] + budget["llm_total_s"] + budget["tts_s"]

    print(f"\n[입력 음성] {in_dur:.2f}s  →  [STT] \"{question}\"")
    print(f"[LLM] \"{answer.strip()}\"")
    print(f"[TTS] {ans_dur:.2f}s 응답 음성 → {ans_wav}")
    print("\n=== 지연 예산 (한 대화 턴) ===")
    print(f"  STT          : {budget['stt_s']*1000:6.0f} ms")
    print(f"  LLM TTFT     : {budget['llm_ttft_s']*1000:6.0f} ms  (첫 토큰까지)")
    print(f"  LLM total    : {budget['llm_total_s']*1000:6.0f} ms")
    print(f"  TTS          : {budget['tts_s']*1000:6.0f} ms")
    print(f"  ─────────────────────────")
    print(f"  E2E 응답지연 : {total*1000:6.0f} ms  (사용자 발화 종료→응답 음성 생성)")
    # 병목 지목
    stage = max([("STT", budget["stt_s"]), ("LLM", budget["llm_total_s"]), ("TTS", budget["tts_s"])],
                key=lambda x: x[1])
    print(f"  병목 단계    : {stage[0]} ({stage[1]/total*100:.0f}%)")


if __name__ == "__main__":
    main()
