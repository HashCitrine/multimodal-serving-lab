#!/usr/bin/env python3
"""음성 에이전트: speech-in → STT → LLM → TTS → speech-out.

앞선 서빙 실험 자산을 한 대화 턴으로 묶고, **단계별 지연 예산(latency budget)**을 측정한다 —
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


def resolve_lang(cfg: dict, lang) -> dict:
    """언어 코드 → {whisper_lang, stt_model, voice, system}. lab-ui/live_voice.resolve_lang 와 동일 규칙."""
    code = (lang or cfg.get("language") or "auto").strip()
    entry = (cfg.get("languages", {}) or {}).get(code, {}) if code != "auto" else {}
    return {
        "code": code,
        "whisper_lang": None if code == "auto" else (entry.get("whisper_lang") or code),
        "stt_model": entry.get("stt_model") or "",
        "voice": entry.get("voice") or "",
        "system": entry.get("system") or "",
    }


def piper_voice(cfg, use_cuda=False, voice=None):
    from piper import PiperVoice
    name = voice or cfg["tts"]["voice"]
    return PiperVoice.load(str(resolve(cfg["tts"]["voice_dir"]) / f"{name}.onnx"),
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
    ap.add_argument("--language", help="대화 언어(auto|ko|en|ja|zh). STT 언어·응답 언어·기본 보이스를 함께 결정")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    if not args.ask and not args.audio:
        raise SystemExit("--ask <text> 또는 --audio <wav> 필요")

    cfg = load_config(args.config)
    out_dir = (SCRIPT_DIR / "outputs"); out_dir.mkdir(exist_ok=True)
    L = resolve_lang(cfg, args.language)
    voice_name = L["voice"] or cfg["tts"]["voice"]
    voice = piper_voice(cfg, voice=voice_name)

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
    stt_model = L["stt_model"] or cfg["stt"]["model"]
    whisper_lang = L["whisper_lang"]
    if whisper_lang and whisper_lang != "en" and stt_model.endswith(".en"):
        stt_model = stt_model[:-3]  # 영어 전용 모델을 다국어 변형으로 승격
    from faster_whisper import WhisperModel
    stt = WhisperModel(stt_model, device=cfg["stt"].get("device", "auto"),
                       compute_type=cfg["stt"].get("compute_type", "int8"))
    list(stt.transcribe(in_wav, beam_size=1, language=whisper_lang)[0])  # warmup
    t0 = time.perf_counter()
    segs, _ = stt.transcribe(in_wav, beam_size=1, language=whisper_lang)
    question = "".join(s.text for s in segs).strip()
    budget["stt_s"] = time.perf_counter() - t0

    # 2) LLM (TTFT + 디코딩 분리 측정)
    from openai import OpenAI
    prov = cfg["llm"]
    client = OpenAI(base_url=prov["base_url"], api_key=prov.get("api_key", "EMPTY"))
    sys_prompt = L["system"] or cfg["llm"].get("system", "You are a concise, friendly English tutor. Answer in 1-2 sentences.")
    if L["code"] == "auto" and not L["system"]:
        sys_prompt += " Respond in the same language as the user."
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
