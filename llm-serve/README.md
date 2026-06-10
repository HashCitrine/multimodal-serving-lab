# llm-serve — LLM 서빙·양자화 최적화 (로컬 Ollama, 클라우드 vLLM 대비 구조)

LLM 서빙을 **0→1로** 다루되, 로컬↔클라우드 전환이 쉬운 구조로 만든다.

## 클라우드 전환 구조 (핵심)
로컬(Ollama)도 클라우드(vLLM)도 **OpenAI 호환 API**를 말한다. 그래서 `config.yaml`의
`provider.base_url` 한 줄만 바꾸면 같은 코드/벤치가 그대로 돈다.
- 지금(로컬): `base_url: http://localhost:11434/v1` (Ollama)
- 나중(클라우드): `base_url: http://<vllm-host>:8000/v1` (vLLM) — 모델명만 교체
> Provider-agnostic OpenAI-compatible interface pattern. LLM은 TTS/STT와 달리 BentoML로 직접
> 감싸기보다 **전용 서빙 런타임(vLLM/Ollama=llama.cpp)** 을 쓰는 게 표준 — 모달리티마다 맞는 프레임워크 선택.

## 구성
```
llm-serve/
├── chat.py         # OpenAI 호환 채팅 CLI (TTFT·decode tok/s)
├── quant_sweep.py  # Q4/Q8/FP16 양자화 → tok/s·메모리·디스크·품질(vs fp16)
├── bench_llm.py    # 동시성 sweep → 집계 throughput·TTFT (vLLM에도 그대로 사용)
├── config.yaml     # provider(base_url/model) + 양자화 태그
└── requirements.txt
```
로컬 백엔드: `ollama serve` + `ollama pull llama3.2:1b-instruct-q4_K_M` 등.

## 빠른 시작
```bash
pip install -r requirements.txt
ollama pull llama3.2:1b-instruct-q4_K_M    # (q8_0, fp16 도 동일 패턴)
python chat.py -p "Explain MLOps in one sentence."
python quant_sweep.py
python bench_llm.py --concurrency 1 2 4 8
# 클라우드 전환: python bench_llm.py --base-url http://<vllm>:8000/v1 --model meta-llama/Llama-3.2-1B-Instruct
```

## 측정 결과 (M5 Max, 로컬 Ollama, llama3.2:1b) — 자세히는 `../docs/experiments.md`
**1) 양자화 (Q4/Q8/FP16)** — *Phase 2(STT)와 대비*
| quant | disk | decode tok/s | 품질(vs fp16) |
|--|--:|--:|--:|
| q4_K_M | 0.81GB | **372** | 0.52 |
| q8_0 | 1.32GB | 282 | 0.82 |
| fp16 | 2.48GB | 177 | 1.00 |
- LLM은 decode가 **메모리 대역폭 바운드** → 양자화가 **메모리뿐 아니라 속도까지** 올린다(q4 = fp16의 2.1×).
  STT(Phase 2)에선 int8이 CPU에서 오히려 느렸던 것과 정반대. **품질은 q4에서 눈에 띄게 저하, q8이 균형점.**

**2) 동시성/배칭** — *Phase 1(TTS)과 대비*
| 동시성 | 집계 tok/s |
|--:|--:|
| 1 | 235 |
| 8 | 342 |
- 연속 배칭으로 집계 처리량 **1.45× 상승**(235→342). 배칭이 0× 였던 TTS와 대비. 1B+로컬이라 폭이 작고,
  **GPU+vLLM에서 훨씬 큰 이득**(여기가 클라우드를 쓰는 지점).

## 다음(클라우드 전환 시)
- vLLM endpoint로 `base_url` 교체 후 `bench_llm.py` 재실행 → continuous batching·PagedAttention 효과 측정
- AWQ/GPTQ 양자화, 더 큰 모델, KV-cache·컨텍스트 길이별 처리량
