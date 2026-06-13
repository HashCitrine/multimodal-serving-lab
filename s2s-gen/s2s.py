#!/usr/bin/env python3
"""S2S(speech-to-speech) 단일 턴 CLI — 캐스케이드와 음성 네이티브 모델을 같은 인터페이스로 실행.

입력 음성(wav) → 선택한 backend → 응답 음성(wav). 단계 경계가 없는 S2S와, STT→LLM→TTS
캐스케이드를 '같은 in_wav→out_wav+metrics' 형태로 돌려 지연/관찰가능성 트레이드오프를 본다.

마이크 없이 자체 검증: --ask "질문" 이면 Piper로 질문 음성을 합성해 입력으로 쓴다.

사용:
    python s2s.py --backend cascade --ask "How do you spell necessary?"
    python s2s.py --backend melo    --ask "파이썬 리스트와 튜플의 차이는?"
    python s2s.py --backend csm     --audio question.wav --device cuda
"""
from __future__ import annotations

import argparse
import re
import wave
from pathlib import Path

import numpy as np
import yaml

from backends import build_backend
from runtime_checks import require_piper_voice

SCRIPT_DIR = Path(__file__).parent
_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else (SCRIPT_DIR / p)


def infer_text_language(text: str) -> str:
    return "ko" if _HANGUL_RE.search(text or "") else "auto"


def apply_text_language_hint(cfg: dict, text: str) -> None:
    if cfg.get("language", "auto") == "auto":
        inferred = infer_text_language(text)
        if inferred != "auto":
            cfg["language"] = inferred


def synth_question(cfg: dict, text: str, out: Path) -> float:
    """Piper로 질문 음성을 합성(마이크 입력 대체). 반환: 음성 길이(초)."""
    if cfg.get("backend") == "melo":
        return synth_question_melo(cfg, text, out)

    from piper import PiperVoice, SynthesisConfig
    tts = cfg.get("tts", {})
    lang = cfg.get("language", "auto")
    entry = (cfg.get("languages", {}) or {}).get(lang, {}) if lang != "auto" else {}
    voice_name = entry.get("voice") or tts.get("voice", "en_US-lessac-medium")
    try:
        voice_path = require_piper_voice(
            tts.get("voice_dir", "../tts-gen/models"),
            voice_name,
            purpose=f"--ask input synthesis for language={lang}",
            base=SCRIPT_DIR,
        )
    except RuntimeError as e:
        raise SystemExit(str(e) + "\nOr pass --audio <wav> to skip Piper question synthesis.") from e
    voice = PiperVoice.load(str(voice_path))
    audio = np.concatenate([c.audio_int16_array for c in voice.synthesize(text, SynthesisConfig())])
    sr = voice.config.sample_rate
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(audio.tobytes())
    return len(audio) / sr


def synth_question_melo(cfg: dict, text: str, out: Path) -> float:
    """Melo 백엔드 검증용 질문 음성 합성. 한국어 Piper 보이스 호환성에 의존하지 않는다."""
    from melo_runtime import prepare_unidic_lite

    lang = cfg.get("language", "auto")
    entry = (cfg.get("languages", {}) or {}).get(lang, {}) if lang != "auto" else {}
    melo_lang = (entry.get("melo_lang") or cfg.get("melo_default_language", "EN")).upper()
    if melo_lang == "EN" and infer_text_language(text) == "ko":
        raise SystemExit(
            "Melo --ask detected Korean text but resolved language=EN. "
            "Run with --language ko or keep language=auto so Korean text can be inferred before synthesis."
        )
    if melo_lang == "EN":
        try:
            import nltk
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except Exception as e:
            raise SystemExit(
                "Melo English synthesis requires NLTK resource averaged_perceptron_tagger_eng.\n"
                "Install it with:\n"
                "  uv run --with nltk python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"\n"
                "For Korean text, run with --language ko."
            ) from e
    device = cfg.get("melo_device", cfg.get("device", "auto"))
    speaker = str(cfg.get("melo_speaker", "") or "")
    speed = float(cfg.get("melo_speed", 1.0))

    prepare_unidic_lite()
    try:
        from melo.api import TTS
    except Exception as e:
        raise SystemExit(
            "Melo --ask input synthesis requires MeloTTS packages.\n"
            "Run through lab-ui s2s melo, or use:\n"
            "  uv run --with 'melotts @ git+https://github.com/myshell-ai/MeloTTS.git' "
            "--with unidic-lite python s2s.py --backend melo ..."
        ) from e

    model = TTS(language=melo_lang, device=device)
    speakers = model.hps.data.spk2id
    keys = list(speakers.keys())

    def pick(key: str):
        return speakers[key] if key in keys else None

    if speaker:
        spk = int(speaker) if speaker.isdigit() else (pick(speaker) if pick(speaker) is not None else speakers[keys[0]])
    elif melo_lang == "EN":
        spk = pick("EN-Default") if pick("EN-Default") is not None else speakers[keys[0]]
    else:
        spk = pick(melo_lang) if pick(melo_lang) is not None else speakers[keys[0]]
    model.tts_to_file(text or "ok", spk, str(out), speed=speed, quiet=True)
    with wave.open(str(out), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def main():
    ap = argparse.ArgumentParser(description="S2S 단일 턴 (cascade|csm|melo)")
    ap.add_argument("--backend", choices=["cascade", "csm", "melo"], help="config의 backend 덮어쓰기 (moshi는 라이브 전용 — lab-ui Moshi 카드)")
    ap.add_argument("--ask", help="질문 텍스트(Piper로 음성 합성해 입력으로 사용)")
    ap.add_argument("--audio", help="질문 음성 wav 직접 입력")
    ap.add_argument("--language", help="대화 언어(auto|ko|en|ja|zh). STT·LLM·기본 TTS 언어를 함께 결정")
    ap.add_argument("--device", help="auto|cpu|cuda|mps (config 덮어쓰기)")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()
    if not args.ask and not args.audio:
        raise SystemExit("--ask <text> 또는 --audio <wav> 필요")

    cfg = load_config(args.config)
    if args.backend:
        cfg["backend"] = args.backend
    if args.language and args.language != "auto":
        cfg["language"] = args.language
    elif args.language == "auto":
        cfg["language"] = "auto"
    if args.ask and cfg.get("backend") == "melo":
        apply_text_language_hint(cfg, args.ask)
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
    if m.get("detected_language"):
        print(f"[STT language] detected={m['detected_language']} prob={m.get('language_probability', 0):.4f}")
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
