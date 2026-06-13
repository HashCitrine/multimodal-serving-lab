# 실험 기록

## 이미지 생성

- `diffusers`와 Stable Diffusion v1.5를 기반으로 로컬 text-to-image 생성을 시작했습니다.
- Anything V5, Ghibli 계열 LoRA/모델처럼 애니메이션·일러스트 스타일에 가까운 모델을 테스트했습니다.
- 고양이, 캐릭터 이미지 생성을 위해 positive prompt, negative prompt, seed를 바꿔가며 반복 실험했습니다.
- 반복 실행이 많아지면서 단발 생성용 스크립트와 업스케일 보조 스크립트를 추가했습니다.

## 비디오 생성

- 먼저 키프레임 이미지를 만든 뒤 `img2img`로 비슷한 프레임을 여러 장 생성하고, `ffmpeg`로 이어 붙이는 방식을 시도했습니다.
- 이 방식은 매 프레임을 새로 생성하는 것보다 시각적 연속성은 나았지만, 실제 의미 있는 동작을 만들지는 못했습니다. 걷기 같은 프롬프트는 정적인 이미지가 흔들리거나 왜곡되는 결과가 많았습니다.
- 실제 프레임 간 모션을 만들기 위해 AnimateDiff와 motion adapter를 테스트했습니다.
- 16GB Apple Silicon 환경에서는 프레임 수와 해상도를 낮춰야 실행 가능했습니다. 메모리 사용은 줄었지만 품질과 피사체 일관성은 크게 떨어졌습니다.
- 대안적인 text-to-video 경로로 Zeroscope와 LTX 계열 스크립트도 테스트했습니다.

## 정리

- 로컬 이미지 생성은 프롬프트 실험과 반복 작업에 충분히 사용할 만했습니다.
- 로컬 비디오 생성은 훨씬 제약이 컸습니다. 메모리 부족, 낮은 해상도, 약한 피사체 일관성이 반복적인 문제였습니다.
- `img2img` 프레임 연결은 작은 카메라 움직임처럼 보이는 변화에는 쓸 수 있지만, 설득력 있는 실제 동작을 만들기에는 한계가 있습니다.
- motion adapter 기반 파이프라인은 실제 움직임을 만들 수 있지만, 결과 품질은 RAM/VRAM, 해상도, 프레임 수, base model에 크게 좌우됩니다.
- 제품 수준의 비디오 품질을 기대한다면 이 로컬 환경보다는 hosted service나 더 큰 GPU 환경이 현실적입니다.

## 서빙 내부 원리 학습 (serve/) — 직접 구현 baseline + 동적 배칭

> 실제 서빙은 BentoML/Triton/vLLM로 한다. 이 절은 그 프레임워크들의 배칭을 제대로
> 이해·설정하기 위해 dynamic batching·큐·지연/처리량을 **직접 구현해 계측**한 기록이며,
> 이후 프레임워크 결과와 나란히 비교하는 기준선으로 쓴다.

추론 서버의 핵심 요소(요청 큐, 동적 마이크로배칭, 메트릭, 그레이스풀 셧다운)를
프레임워크 없이 직접 구현했다. 모달리티는 `ModelAdapter` 인터페이스 뒤에 끼워 교체한다.

### 동적 배칭 효과 측정 (echo 더미 어댑터, 배치당 50ms 지연 흉내)
환경: Apple Silicon / Python 3.9, `max_batch_size=8`, `max_wait_ms=10`, workers=1.
스파인 자체의 배칭 동작만 보기 위해 모델 대신 50ms 슬립 어댑터로 측정.

| 동시성 | throughput(req/s) | p50(ms) | p95(ms) | p99(ms) |
|-------:|------------------:|--------:|--------:|--------:|
| 1 | 14.8 | 67.5 | 70.0 | 70.2 |
| 2 | 28.8 | 69.6 | 72.8 | 73.0 |
| 4 | 55.6 | 72.0 | 75.0 | 75.7 |
| 8 | 111.5 | 72.0 | 78.9 | 80.7 |

- 동시성 1→8에서 처리량 약 **7.5배**(14.8→111.5 rps), p50은 67→72ms로 거의 불변.
  요청이 한 배치로 묶여 같은 50ms 안에 함께 처리되기 때문(서버 관측 평균 배치 2.1, 최대 8).
- 즉 **지연을 거의 늘리지 않고 처리량을 끌어올리는** 동적 배칭의 이득을 직접 재현·계측.
- 메모: `max_wait_ms`를 키우면 배치가 커져 처리량↑ 대신 꼬리 지연(p99)↑ 트레이드오프.
  단, 이 이득은 **모델이 진짜 배치 forward를 지원할 때**의 이야기다(아래 TTS 결과 참고).

## TTS (tts-gen/) — Piper 음성 합성 서빙·최적화

Piper(ONNX) `en_US-lessac-medium` 을 BentoML과 직접 구현 baseline 양쪽으로 서빙하며
RTF와 동시성 처리량을 측정. 환경: Apple Silicon, CPU/onnxruntime 1.19.

### 1) 단발 합성 RTF (문장 길이별, runs=5 median)
RTF = 합성시간 / 생성오디오길이 (낮을수록 좋음, <1 = 실시간보다 빠름).

| 입력 chars | audio(s) | synth(s) | RTF | 실시간 배수 |
|-----------:|---------:|---------:|------:|-----------:|
| 6   | 0.64  | 0.0187 | 0.0293 | 34.2× |
| 42  | 2.39  | 0.0544 | 0.0227 | 44.0× |
| 81  | 5.33  | 0.1056 | 0.0198 | 50.5× |
| 226 | 12.79 | 0.2663 | 0.0208 | 48.0× |

- 평균 RTF ≈ **0.023 (약 40–50× 실시간)**. 입력이 길수록 고정 오버헤드(모델 init/phonemize)가
  분산돼 RTF가 좋아진다 → 매우 짧은 발화를 잦게 치면 단위 비용이 비싸다.

### 2) 서빙 처리량 — BentoML vs 직접 구현 baseline (같은 모델, 동시성 sweep, 48 req)
| 동시성 | baseline rps | baseline p50(ms) | BentoML rps | BentoML p50(ms) |
|------:|------:|------:|------:|------:|
| 1 | 10.8 | 94 | 11.9 | 85 |
| 2 | 11.4 | 177 | 11.7 | 171 |
| 4 | 11.8 | 343 | 11.6 | 348 |
| 8 | 12.5 | 639 | 11.5 | 688 |

- **두 서버 모두 ~11–12 rps에서 평탄**, 지연은 동시성에 비례해 선형 증가.
- **동적 배칭이 처리량을 못 올린다**: Piper는 per-utterance 합성이라 배치를 모아도 내부에서
  순차 처리되고(진짜 배치 forward 아님), onnxruntime이 이미 멀티코어를 포화시킨다.
  worker 스레드를 1→4로 늘려도 동일(~11 rps) — CPU가 요청당 이미 포화.
- 병목이 **모델(CPU)** 이라 서버 구현 차이는 처리량에 거의 영향 없음. BentoML의 가치는 여기서
  처리량이 아니라 **패키징·Docker화·헬스/동시성·관측성**에 있다.

### 3) 처리량 향상 검증 — intra-op 스레드 제한 × BentoML workers(replica)
"배칭이 안 되면 무엇이 되는가"를 실측. 18코어 머신, 동시성 8, 64 req, BentoML `workers`(프로세스 복제)와
인스턴스당 onnxruntime `intra_op_num_threads` 조합.

| 구성 | workers | intra-op | throughput(rps) | p50(ms) |
|--|------:|------:|------:|------:|
| A 기본 | 1 | 기본(전체) | 12.5 | 638 |
| B 스레드만 제한 | 1 | 2 | 6.8 | 1170 |
| C 제한+복제 | 4 | 2 | 13.3 | 597 |
| **D 제한+복제** | **8** | **2** | **25.5** | **252** |

- **D: 처리량 2× (12.5→25.5 rps), p50도 638→252ms 동반 개선.** 동적 배칭이 0× 였던 것과 대비.
- B가 핵심 교훈: **스레드 제한만 하면 단일 요청이 느려져 오히려 손해**(6.8 rps). 제한은 *복제와 결합*해야
  의미가 있다 — 인스턴스당 코어 점유를 줄여 더 많은 replica가 18코어를 동시에 효율적으로 채우게 한다.
- 이론 상한 추정: intra=2 요청 ≈ 0.29 core·s → 18코어/0.29 ≈ 62 rps. workers를 더 늘리면 상승하나
  컨텍스트 스위칭·메모리(인스턴스당 모델 사본)로 수익 체감. 프로덕션에선 K8s CPU requests/limits로
  인스턴스당 코어를 핀하고 HPA로 replica를 스케일하는 형태가 된다.

### 정리 (서빙 판단)
- **"배칭 = 처리량↑"은 배치 forward가 되는 모델(LLM·이미지 diffusion 등)에서만 성립.** CPU 바운드
  per-utterance TTS의 처리량 레버는 **인스턴스당 스레드 제한 + 프로세스 복제(replica) 스케일아웃**이며,
  이를 실측으로 2× 확인했다. (그리고 BentoML `workers`로 이 복제를 코드 한 줄로 얻는다 = 프레임워크 가치)
- 다음 최적화(선택): 스트리밍 합성 TTFB, workers×intra 더 넓은 스윕(상한 탐색).
  (int8 양자화는 STT/LLM 단계에서 더 효과가 큰 모델로 다룸 — 아래 참고.)

## STT (stt-gen/) — faster-whisper 서빙·int8 양자화 최적화

faster-whisper(CTranslate2)를 BentoML로 서빙. 핵심 질문: **"int8 양자화는 언제 이득인가?"**
Piper(TTS)로 만든 클립을 전사하는 **TTS↔STT 왕복**으로 검증(외부 오디오 불필요, 정답을 알아 WER 계산).
환경: Apple Silicon 18코어, CPU, 클립 4개(총 ~14s), beam=1.

### 모델크기 × compute_type — RTF·WER·메모리
| model | compute | RTF | WER | mem(MB) |
|--|--|------:|------:|------:|
| base.en | float32 | 0.086 | 0.053 | 836 |
| base.en | int8 | 0.095 | 0.053 | **702** |
| small.en | float32 | 0.177 | 0.053 | 2162 |
| small.en | int8 | 0.245 | 0.053 | **1149** |
| medium.en | float32 | 0.515 | 0.053 | 4623 |
| medium.en | int8 | 0.702 | 0.053 | **2094** |

- **int8의 이득은 속도가 아니라 메모리**: 전 모델에서 ~50% 감소(medium 4.6GB→2.1GB, 절대 절감은 큰
  모델일수록 큼). **WER는 완전히 동일(정확도 손실 0)**.
- CPU에선 int8이 오히려 **약간 느림**(양자화/역양자화 오버헤드 + float32 GEMM이 이미 최적화). int8의
  *속도* 이득은 보통 **GPU(int8 텐서코어)**나 메모리 대역폭 바운드 상황에서 나타남 → 클라우드 단계 확인.
- 서빙 의미: 메모리 반감 = **GPU당 모델/replica 밀도 2×**. TTS 실험의 '복제로 처리량↑' 레버와 직결
  (메모리를 줄여야 더 많은 replica를 같은 GPU에 얹는다).

### 서빙·왕복 검증
- BentoML `WhisperSTT`: `POST /transcribe` (wav 업로드) → 텍스트+RTF. 헬스/동시성/Docker 동일 패턴.
- TTS↔STT 왕복(Piper→Whisper)으로 음성 입↔출 파이프라인 확인(스피킹 학습/음성 에이전트 맥락).

### 정리 (TTS 실험 + STT 실험 종합)
- **최적화는 도움 되는 '축'이 제각각이다**: TTS 실험 복제 = *throughput*, STT 실험 int8 = *memory*.
  배칭은 CPU per-utterance TTS에서 0×, int8은 CPU에서 속도 0× but 메모리 −50%.
- 즉 **"이 기법이 이 모델·이 하드웨어에서 어느 축에 듣는지"를 가정하지 말고 측정**해야 한다 — 이게
  서빙/최적화 엔지니어의 핵심 판단. (LLM 실험에서 대비되는 결론을 확인 — 아래.)

## LLM (llm-serve/) — 서빙·양자화 최적화 (로컬 Ollama, 클라우드 vLLM 대비 구조)

로컬↔클라우드 전환을 **OpenAI 호환 API**로 추상화(로컬 Ollama=llama.cpp ↔ 클라우드 vLLM, `base_url`만 교체).
환경: M5 Max, 로컬 Ollama, `llama3.2:1b` (q4_K_M/q8_0/fp16).

### 1) 양자화 스윕 — *STT 실험과 정반대 결론*
| quant | disk(GB) | mem(GB) | prefill t/s | decode t/s | 품질(vs fp16) |
|--|--:|--:|--:|--:|--:|
| q4_K_M | 0.81 | 5.87 | 8779 | **372.4** | 0.52 |
| q8_0 | 1.32 | 6.38 | 7471 | 282.0 | 0.82 |
| fp16 | 2.48 | 7.54 | 5097 | 177.2 | 1.00 |

- **양자화가 메모리뿐 아니라 decode 속도까지 올린다**(q4 = fp16의 **2.1×**). LLM 자기회귀 decode는
  **메모리 대역폭 바운드** → 가중치가 작을수록 토큰당 읽을 메모리가 줄어 빨라진다.
- → STT 실험에서 int8이 CPU에서 *느렸던* 것과 정반대. **같은 'int8/저비트 양자화'라도 모델 구조
  (메모리 바운드 LLM vs 연산 바운드 인코더)·하드웨어에 따라 속도 영향이 반대로 나온다.**
- **품질 비용**: q4는 fp16과 어휘 일치도 0.52로 눈에 띄게 갈라짐, q8은 0.82로 근접 → **q8이 속도/품질 균형점**.

### 2) 동시성/연속 배칭 — *TTS 실험과 정반대 결론*
| 동시성 | 집계 tok/s | 요청별 decode t/s | TTFT(ms) |
|--:|--:|--:|--:|
| 1 | 235 | 369 | 99 |
| 2 | 283 | 377 | 198 |
| 4 | 323 | 394 | 382 |
| 8 | 342 | 394 | 748 |

- **집계 처리량이 동시성과 함께 상승(235→342, 1.45×)**, 요청별 decode 속도는 유지. LLM은 배치 forward가
  되므로 연속 배칭이 효과적 — TTS(배칭 0×)·STT와 대비. (1B+로컬이라 폭이 작음. **GPU+vLLM에서 훨씬 큼**
  — 여기가 클라우드를 쓰는 지점. 같은 `bench_llm.py`를 vLLM base_url로 재실행하면 측정 가능.)

### 정리 (TTS·STT·LLM 실험 종합) — "측정 없이는 모른다"의 3대 대조
| | 레버 | 결과 | 이유 |
|--|--|--|--|
| TTS (CPU) | 동적 배칭 | throughput **0×** → 복제로 2× | per-utterance·연산 바운드, 배치 forward 없음 |
| STT (CPU) | int8 양자화 | 속도 **0×**(약간 ↓), 메모리 **−50%** | 연산 바운드, float32 GEMM 이미 최적 |
| LLM (Metal) | 양자화 + 배칭 | 속도 **2.1×** + throughput **1.45×** | decode 메모리 대역폭 바운드 + 배치 forward 가능 |
- **같은 이름의 최적화(배칭/양자화)가 모델·하드웨어에 따라 정반대로 작용**한다. 가정하지 말고 측정하는 것이
  서빙/최적화 엔지니어의 핵심 — 세 모달리티를 직접 서빙·계측해 이 차이를 실측으로 확인했다.

## 음성 에이전트 (voice-agent/) — STT→LLM→TTS end-to-end 지연 예산

세 서빙을 한 대화 턴으로 묶어(speech-in→STT→LLM→TTS→speech-out) **응답 지연의 단계별 예산**을 측정.
대화형 음성 튜터/에이전트 맥락. 환경: M5 Max, 로컬(base.en + llama3.2:1b-q4 + Piper), 5문항 중앙값.

| 단계 | 정상상태(warm) | 콜드스타트(첫 턴) |
|--|--:|--:|
| STT | 344 ms (**55%**) | 346 ms |
| LLM total (TTFT) | 196 ms (142) | 1061 ms (TTFT 982) |
| TTS | 71 ms | 112 ms |
| **E2E** | **630 ms** | ~1520 ms |
| 병목 | **STT** | **LLM** |

- **콜드스타트는 LLM(70%+)이 병목**(모델 로드+prefill), **워밍 후 정상상태는 STT(55%)가 병목**으로 뒤바뀐다.
  → 처방이 다르다: ①모델 워밍 유지로 콜드스타트 제거 → ②그 다음 STT 단축(작은 whisper/GPU/스트리밍 부분전사).
- "대화 지연 = LLM 탓"이라 가정하기 쉽지만 **워밍 후엔 STT가 지배** — 4번째 "측정 없이는 모른다" 사례.
- 체감 지연 추가 최적화(선택): LLM 토큰을 문장 단위로 흘려 TTS와 오버랩(스트리밍).


## S2S (s2s-gen/) — 캐스케이드 · 표현형 TTS · full-duplex

음성 대화 파이프라인은 두 갈래다. **캐스케이드**(STT→LLM→TTS)는 단계마다 추적 로그가 남아 관찰가능성·디버깅이 좋고, **음성 네이티브(S2S)**는 단계 경계가 없어 지연·자연스러움에 강하다. `s2s-gen`은 파일→파일(`in_wav → out_wav + metrics`) 백엔드로 둘을 같은 축(TTFA·E2E·RTF)에서 비교하고, full-duplex 모델은 라이브 경로로 붙인다.

로컬 실측으로 확정된 모델별 실제 형태:

- **`cascade`** — faster-whisper + OpenAI 호환 LLM + Piper. 무거운 모델 없이 어디서나 동작하는 기준선.
- **`csm`** — Sesame CSM(`sesame/csm-1b`)은 단독 S2S가 아니라 **conversational TTS**다(`generate(text, speaker, context) → waveform`, MPS 지원). 그래서 cascade 의 STT+LLM 은 그대로 쓰고 **TTS 단계만 CSM 으로 교체**한다(Piper 대비 운율·표현력 비교). 외부 clone(`CSM_DIR`) + HF 게이트 동의 필요.
- **`melo`** — MeloTTS 는 한국어·영어·일본어·중국어 등을 지원하는 다국어 TTS다. CSM 과 같은 구조로 cascade 의 STT+LLM 은 유지하고 **TTS 단계만 MeloTTS 로 교체**한다. Live Voice 에서는 `language=ko`, `pipeline_mode=s2s`, `s2s_backend=melo` 로 한국어 응답 음성을 실험한다.
- **`moshi`** — Kyutai Moshi(full-duplex)는 파일→파일 오프라인 API가 없다. 로컬 실행은 **라이브 웹 데모뿐**(`moshi_mlx.local_web`, :8998)이라 backend 가 아니라 lab-ui "Moshi 라이브" 카드로 띄운다.

```bash
cd s2s-gen
python s2s.py --backend cascade --ask "How do you spell necessary?"             # 어디서나 동작
CSM_DIR=../_external/csm uv run python s2s.py --backend csm --ask "..."          # STT→LLM→CSM(MPS)
uv run --with "melotts @ git+https://github.com/myshell-ai/MeloTTS.git" --with unidic-lite python s2s.py --backend melo --language ko --ask "파이썬 리스트와 튜플의 차이는?"
uv run --python 3.12 --with moshi_mlx python -m moshi_mlx.local_web -q 4         # Moshi 라이브 → :8998
```

한국어 경로는 `tts-gen`의 `ko_KR-kss-medium` Piper 보이스로 질문 음성을 만들고, faster-whisper 다국어 모델(`small`)로 전사한 뒤, LLM 프롬프트를 한국어로 바꾸고, 응답 음성은 cascade(Piper) 또는 s2s+melo(MeloTTS `KR`)로 합성한다. `ko_KR-kss-medium` 은 `neurlang/piper-onnx-kss-korean` 커뮤니티 보이스를 다운로드하며 라이선스는 cc-by-nc-sa-4.0 이다.

cascade 기준선 측정(로컬 MPS, base.en + llama3.2:1b-q4 + Piper, 1턴): STT 411 ms / LLM TTFT 1577 ms / TTS 390 ms → TTFA·E2E 약 2.4 s, RTF 0.73. 캐스케이드는 비스트리밍이라 첫 오디오가 TTS 종료 후 나와 TTFA=E2E다. CSM/MeloTTS 로 TTS 를 바꾸면 같은 STT/LLM 위에서 운율·표현력 또는 다국어 음성을 비교한다 — 같은 `_synthesize` 교체점에서 직접 비교한다.

이 비교의 의의는 "지연 vs 관찰가능성 vs 표현력/언어 지원"을 같은 측정 축에서 본다는 점이다. 캐스케이드는 어디서 틀렸는지(STT 오인식/LLM 응답/TTS) 단계별로 짚을 수 있고, CSM/MeloTTS 는 그 경계를 유지한 채 TTS 품질 또는 한국어 음성을 올리며, Moshi 는 경계 자체를 없애 full-duplex 자연스러움을 얻는다.

### 인프로세스 vs 서빙 — 모델마다 손익이 반대

같은 모델을 "이 프로세스에 캐싱(인프로세스)"하느냐 "BentoML 상주 서비스로 띄워 HTTP 호출(서빙)"하느냐는 **모델 무게에 따라 손익이 갈린다**. CSM 은 conversational TTS 라 STT(whisper-stt)·TTS(piper-tts)·CSM(csm-tts)을 모두 같은 서비스 패턴으로 띄울 수 있어, 이 트레이드오프를 한 자리에서 측정한다(lab-ui "모델 서비스" 카드 + Live Voice `serving_mode` 토글).

| 모델 | 단위 연산 비용 | 인프로세스 | 서빙(HTTP 상주) | 판단 |
|---|---|---|---|---|
| whisper-stt | 수십~수백 ms (base.en int8 RTF≈0.28) | 직접 호출, 오버헤드 0 | HTTP 왕복 + wav 멀티파트 직렬화가 연산과 맞먹음 | 인프로세스 유리 |
| piper-tts | 수십 ms (RTF≈0.05) | 직접 호출 | HTTP + wav 바이트 전송 오버헤드가 연산보다 큼 | 인프로세스 유리 |
| csm-tts | **로드 수~수십 초** + 합성 수 초 | 매 호출 재로드면 실시간 불가 | **1회 로드·상주 → 이후 합성만** | **서빙이 압도** |

요지: 경량 모델은 서비스화가 오히려 턴 지연을 늘릴 수 있고(왕복·직렬화 오버헤드 > 연산), 무거운 모델은 "재로드 제거"가 워낙 커서 서비스화가 결정적이다. 그래서 lab-ui 기본값은 STT/Piper 인프로세스 + CSM 서비스이고, 토글로 둘을 같은 turn 파이프라인·메트릭(E2E·RTF) 위에서 실측 비교한다. 서비스 자체의 정상상태 합성 성능(로드 비용 제외)은 `bench_csm.py`(csm-tts) / `bench_rtf.py`(piper-tts)로 따로 본다.


## 아바타 립싱크 (avatar-gen/) — text→TTS→lip-sync 실행·벤치

토킹헤드 아바타 실험은 단계로 나눈다. `static` backend는 ffmpeg로 정지 얼굴과 음성을 합성해 전체 배선을 검증하는 폴백이고, `wav2lip`·`musetalk` backend는 외부 repo/checkpoint를 호출해 실제 입모양 동기화 mp4를 만든다. `musetalk`은 잠재공간 인페인팅 기반 실시간 립싱크(GPU 권장)로, wav2lip 대비 품질/RTF를 비교한다.

측정 스크립트는 `bench_avatar.py`다. 같은 입력으로 audio length, TTS latency, lip-sync latency, lip-sync RTF, end-to-end latency, NVIDIA GPU peak memory를 기록한다. 모델 weights, 얼굴 입력, 오디오, mp4 결과물은 Git에 포함하지 않는다.

```bash
cd avatar-gen
python bench_avatar.py --backend static --runs 3
python bench_avatar.py --backend wav2lip --face /path/to/face.jpg --device cuda --runs 3 --gpu-id 0
python bench_avatar.py --backend musetalk --face /path/to/face.jpg --device cuda --runs 3 --gpu-id 0
```

아직 Wav2Lip·MuseTalk 실측값은 환경 준비 후 채운다. 현재 목표는 제품 품질 아바타가 아니라, 음성 합성 결과를 실제 lip-sync 모델까지 연결하고 병목을 수치로 분리하는 것이다.

저지연 대화용 아바타는 무거운 뉴럴 비디오 대신 **오디오 구동 viseme**(브라우저)로도 붙일 수 있다. lab-ui Live Voice는 응답 오디오 진폭으로 입모양을 움직이는 viseme-lite 레이어를 제공하는데, 음성 파이프라인과 독립이라 cascade·S2S 출력 모두 동일하게 커버한다(거의 무지연). 서버측 MuseTalk는 포토리얼이 필요할 때의 무거운 옵션이다.
