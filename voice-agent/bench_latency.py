#!/usr/bin/env python3
"""대화 턴 지연 예산 벤치 — 여러 질문에 대해 STT/LLM/TTS 단계 지연의 중앙값을 측정.

모델은 1회만 로드(정상상태 가정)하고 N개 질문을 돌려, 대화형 음성 응답의 병목을 안정적으로 본다.
사용: python bench_latency.py
"""
from __future__ import annotations

import statistics
import tempfile
import time
import wave
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent

QUESTIONS = [
    "What is the plural of child?",
    "How do you spell the word necessary?",
    "Give me a synonym for happy.",
    "What is the past tense of run?",
    "Is it correct to say I have went?",
]


def resolve(rel):
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def main():
    cfg = yaml.safe_load(open(SCRIPT_DIR / "config.yaml", "r", encoding="utf-8"))
    from piper import PiperVoice, SynthesisConfig
    from faster_whisper import WhisperModel
    from openai import OpenAI

    voice = PiperVoice.load(str(resolve(cfg["tts"]["voice_dir"]) / f"{cfg['tts']['voice']}.onnx"))
    sr = voice.config.sample_rate
    stt = WhisperModel(cfg["stt"]["model"], device=cfg["stt"].get("device", "auto"),
                       compute_type=cfg["stt"].get("compute_type", "int8"))
    prov = cfg["llm"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))
    sys_prompt = prov.get("system", "Answer in one short sentence.")

    def synth(text, path):
        audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())

    # 워밍업
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    synth("warm up", tmp)
    list(stt.transcribe(tmp, beam_size=1)[0])
    client.chat.completions.create(model=prov["model"],
        messages=[{"role": "user", "content": "hi"}], max_tokens=1, temperature=0)

    stt_t, llm_ttft, llm_t, tts_t, e2e = [], [], [], [], []
    for q in QUESTIONS:
        qwav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        synth(q, qwav)
        t0 = time.perf_counter()
        segs, _ = stt.transcribe(qwav, beam_size=1)
        question = "".join(s.text for s in segs).strip()
        s_stt = time.perf_counter() - t0

        t0 = time.perf_counter(); ttft = None; ans = ""
        stream = client.chat.completions.create(model=prov["model"],
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
            stream=True, temperature=0.3, max_tokens=prov.get("max_tokens", 80))
        for ch in stream:
            d = ch.choices[0].delta.content
            if d:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                ans += d
        s_llm = time.perf_counter() - t0

        t0 = time.perf_counter()
        synth(ans.strip() or "ok", tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
        s_tts = time.perf_counter() - t0

        stt_t.append(s_stt); llm_ttft.append(ttft or 0); llm_t.append(s_llm)
        tts_t.append(s_tts); e2e.append(s_stt + s_llm + s_tts)

    med = lambda x: statistics.median(x) * 1000
    print(f"questions={len(QUESTIONS)}  (정상상태, 모델 1회 로드)\n")
    print(f"  STT       median : {med(stt_t):6.0f} ms")
    print(f"  LLM TTFT  median : {med(llm_ttft):6.0f} ms")
    print(f"  LLM total median : {med(llm_t):6.0f} ms")
    print(f"  TTS       median : {med(tts_t):6.0f} ms")
    print(f"  ────────────────────────")
    print(f"  E2E       median : {med(e2e):6.0f} ms")
    stages = {"STT": med(stt_t), "LLM": med(llm_t), "TTS": med(tts_t)}
    top = max(stages, key=stages.get)
    print(f"  병목: {top} ({stages[top]/med(e2e)*100:.0f}% of E2E)")


if __name__ == "__main__":
    main()
