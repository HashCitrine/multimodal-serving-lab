# voice-agent — STT→LLM→TTS 음성 에이전트 (지연 예산 측정)

앞선 서빙 실험의 세 서빙(STT·LLM·TTS)을 **한 대화 턴**으로 묶는다: speech-in → STT → LLM → TTS → speech-out.
대화형 음성 튜터/에이전트 맥락. 새 최적화 축 = **end-to-end 지연 예산(latency budget)**.

마이크 없이도 검증: `--ask "질문"` 이면 Piper로 질문 음성을 합성해 입력으로 쓴다(전 구간 로컬 왕복).

## 구성
```
voice-agent/
├── agent.py          # 한 턴 실행 + 단계별 지연 예산 출력
├── bench_latency.py  # 여러 질문 정상상태 중앙값(병목 안정 측정)
├── config.yaml       # stt / llm(provider) / tts 설정
└── pyproject.toml
```

## 실행
```bash
uv sync                                   # + 앞선 서빙 실험 모델(Piper 보이스, Ollama LLM, whisper)
uv run python agent.py --ask "How do you spell necessary?"
uv run python agent.py --audio question.wav
uv run python bench_latency.py
```

## 측정 결과 (M5 Max, 로컬, base.en + llama3.2:1b-q4 + Piper)
정상상태(모델 1회 로드), 5문항 중앙값:

| 단계 | 지연 | 비중 |
|--|--:|--:|
| STT | 344 ms | **55%** |
| LLM (total) | 196 ms (TTFT 142) | 31% |
| TTS | 71 ms | 11% |
| **E2E** | **630 ms** | 100% |

- **정상상태 병목은 STT**(55%). 단, **콜드스타트(첫 턴)** 에선 LLM이 70%+ — 모델 로드+prefill 때문.
  → 최적화 처방이 다르다: ①모델을 항상 워밍 유지(콜드스타트 회피) ②그다음 STT를 줄임(더 작은 whisper,
  GPU, 스트리밍 부분전사). LLM은 TTFT가 작아(142ms) 이미 충분.
- 또 하나의 "측정 없이는 모른다" 사례: *대화 지연은 LLM 탓일 거라 가정하기 쉽지만, 워밍 후엔 STT가 지배*.

## 다음(선택)
- 스트리밍 오버랩(LLM 토큰 → 문장 단위로 TTS를 흘려보내 체감 지연↓), STT 스트리밍 부분전사
- GPU(STT float16)로 STT 단축 측정(클라우드)
