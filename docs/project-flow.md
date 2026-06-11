# 프로젝트 흐름

`multimodal-serving-lab`의 실험 구성, 서브 프로젝트별 로직 흐름, 실험 판단 기준을 정리한다.
용어 설명은 [`glossary.md`](glossary.md), 실제 수치는 [`experiments.md`](experiments.md)를 참고한다.

## 문서 책임

| 문서 | 책임 |
|---|---|
| `project-flow.md` | 현재 문서. 전체 Phase와 서브 프로젝트별 구현 흐름을 설명 |
| `glossary.md` | 이 문서에서 사용하는 용어를 독립적인 사전 형태로 설명 |
| `experiments.md` | 실험 수치, 관찰 결과, 한계와 결론을 기록 |
| 각 디렉터리 `README.md` | 설치, 실행 명령, 개별 프로젝트별 상세 사용법 |

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 목적 | 로컬/클라우드 환경에서 멀티모달 모델을 실행, 서빙, 최적화하며 실제 병목을 측정 |
| 범위 | 이미지, 비디오, TTS, STT, LLM, 음성 에이전트, talking head 아바타 |
| 중심 질문 | 같은 최적화 기법이 모델/하드웨어별로 어떤 축에 효과가 있는가 |
| 산출물 | 실행 스크립트, BentoML 서비스, 직접 구현 serving baseline, 벤치 결과, 통합 파이프라인 |
| 제외 | 모델 파일, 생성 결과물, 완성형 제품 UI |

## 2. 전체 실험 흐름

| 순서 | 디렉터리 | 모달리티 | 핵심 역할 | 다음 단계로 이어지는 자산 |
|---:|---|---|---|---|
| 1 | `sd-gen/` | Text to Image | Stable Diffusion 로컬 이미지 생성 | 이미지 생성 경험, 모델 캐시 |
| 2 | `video-gen/` | Text to Video | AnimateDiff/Zeroscope 비디오 생성 | 모션/프레임 제약 이해 |
| 3 | `serve/` | Serving Runtime | 큐, 스케줄러, 동적 배칭 직접 구현 | 공통 benchmark baseline |
| 4 | `tts-gen/` | TTS | Piper 합성, BentoML 서빙, RTF 측정 | Piper voice, 음성 합성 API |
| 5 | `stt-gen/` | STT | faster-whisper 전사, WER/RTF/메모리 측정 | Whisper STT, TTS-STT 왕복 검증 |
| 6 | `llm-serve/` | LLM | OpenAI 호환 provider, 양자화/동시성 벤치 | Ollama/vLLM 전환 구조 |
| 7 | `voice-agent/` | Speech/Text | STT to LLM to TTS 한 턴 지연 예산 | 대화형 latency breakdown |
| 8 | `avatar-gen/` | Avatar Video | Text/LLM/TTS/lip-sync mp4 파이프라인 | talking head 통합 구조 |

## 3. 큰 데이터 흐름

```text
sd-gen
  prompt -> Stable Diffusion -> image

video-gen
  prompt -> video diffusion pipeline -> frames -> mp4

serve
  HTTP request -> queue -> scheduler -> batch -> adapter -> response

tts-gen
  text -> Piper -> wav -> RTF / BentoML API

stt-gen
  wav -> faster-whisper -> text -> RTF / WER / memory

llm-serve
  prompt -> OpenAI-compatible API -> streamed tokens -> TTFT / token/s

voice-agent
  speech -> STT -> text -> LLM -> answer text -> TTS -> speech

avatar-gen
  prompt/text -> LLM optional -> TTS wav -> lip-sync backend -> mp4
```

## 4. 서브 프로젝트별 로직 흐름

### 4.1 `serve/` - 직접 구현한 serving baseline

| 항목 | 내용 |
|---|---|
| 목적 | 서빙 런타임 내부 원리 학습 |
| 핵심 파일 | `server.py`, `scheduler.py`, `adapters/base.py`, `bench.py` |
| 입력 | `POST /infer` JSON payload |
| 출력 | adapter inference 결과, batch size, latency |
| 핵심 지표 | req/s, p50/p95/p99, 평균 batch size |

```text
client
  -> FastAPI /infer
  -> Scheduler.submit()
  -> asyncio.Queue
  -> max_wait_ms 동안 요청 수집
  -> batch 확정
  -> ThreadPoolExecutor에서 adapter.infer(batch)
  -> 요청별 Future에 결과 전달
  -> HTTP response
```

| 구성 | 역할 |
|---|---|
| `server.py` | FastAPI endpoint와 lifespan 관리 |
| `scheduler.py` | queue, micro-batching, executor 실행 |
| `ModelAdapter` | 모델별 load/warmup/infer/unload 표준화 |
| `bench.py` | 동시성 sweep와 latency/throughput 측정 |

**확인 포인트**

- `max_wait_ms`가 커지면 batch가 커질 수 있지만 tail latency도 늘 수 있다.
- adapter가 실제 batch forward를 하지 못하면 dynamic batching 이득은 제한된다.
- `serve/`는 프레임워크 대체가 아니라 프레임워크 동작을 이해하기 위한 기준선이다.

### 4.2 `sd-gen/` - Stable Diffusion 이미지 생성

| 항목 | 내용 |
|---|---|
| 목적 | Text-to-image 로컬 생성 실험 |
| 핵심 파일 | `generate.py`, `config.yaml`, `upscale.py` |
| 입력 | prompt, negative prompt, seed, 크기, steps |
| 출력 | `outputs/*.png` |
| 핵심 지표 | 생성 성공 여부, 품질, 재현성, 실행 시간 |

```text
config.yaml / CLI
  -> device 선택(cuda/mps/cpu)
  -> StableDiffusionPipeline 또는 StableDiffusionXLPipeline 로드
  -> scheduler를 DPMSolverMultistepScheduler로 교체
  -> prompt / negative prompt / seed 적용
  -> image 생성
  -> PNG 저장 및 metadata 기록
```

| 설정 | 의미 |
|---|---|
| `model.id` | HuggingFace model id 또는 로컬 모델 경로 |
| `generation.prompt` | 만들고 싶은 이미지 설명 |
| `generation.negative_prompt` | 피하고 싶은 특징 |
| `generation.steps` | denoising 반복 횟수 |
| `generation.guidance_scale` | prompt를 따르는 강도 |
| `generation.seed` | 재현성을 위한 난수 seed |
| `device.fp16` | half precision 사용 여부 |

**확인 포인트**

- Apple Silicon/MPS에서는 fp16이 불안정할 수 있어 config 기본값이 `false`다.
- seed를 고정하면 같은 prompt 설정을 재현하기 쉽다.
- `upscale.py`는 diffusion upscaler가 아니라 Pillow 기반 후처리 스크립트다.

### 4.3 `video-gen/` - 로컬 비디오 생성

| 항목 | 내용 |
|---|---|
| 목적 | Text-to-video 모델의 로컬 실행 가능성과 제약 확인 |
| 핵심 파일 | `generate.py`, `generate_ltx.py`, `config.yaml` |
| 입력 | prompt, model 선택, frames, fps, 해상도 |
| 출력 | `outputs/*.mp4` |
| 핵심 지표 | 생성 가능 여부, 프레임 일관성, 메모리 사용, 품질 |

**AnimateDiff**

```text
prompt
  -> MotionAdapter 로드
  -> SD 1.5 base model 로드
  -> AnimateDiffPipeline 구성
  -> num_frames만큼 frame 생성
  -> imageio로 mp4 저장
```

**Zeroscope**

```text
prompt
  -> DiffusionPipeline 로드
  -> MPS 이동 시도, 실패 시 CPU fallback
  -> frame tensor/list 생성
  -> 저장 가능한 frame list로 정규화
  -> mp4 저장
```

| 관찰 | 의미 |
|---|---|
| 이미지 생성보다 메모리 제약이 큼 | frame 수와 해상도가 비용을 크게 늘림 |
| 피사체 일관성이 약함 | 로컬 저해상도 설정에서는 품질 한계가 뚜렷함 |
| MPS float32가 안정적 | fp16은 black frame 등 문제가 날 수 있음 |

### 4.4 `tts-gen/` - Piper TTS 합성 및 BentoML 서빙

| 항목 | 내용 |
|---|---|
| 목적 | Text-to-speech 모델을 로컬 CLI와 BentoML API로 서빙 |
| 핵심 파일 | `synthesize.py`, `bento_service.py`, `bench_rtf.py`, `stream_ttfb.py` |
| 입력 | text, voice, length scale |
| 출력 | WAV 또는 RTF metadata |
| 핵심 지표 | RTF, req/s, p50/p95, worker별 처리량 |

**CLI 합성**

```text
text
  -> PiperVoice 로드
  -> warmup synthesis
  -> voice.synthesize(text)
  -> audio chunks concatenate
  -> WAV 저장
  -> audio_seconds / synth_seconds / RTF 출력
```

**BentoML 서빙**

```text
HTTP request
  -> BentoML service worker
  -> PiperTTS._synthesize_int16()
  -> /synthesize: WAV 반환
  -> /synthesize_meta: RTF metadata 반환
```

| 설정 | 의미 | 기대 효과 |
|---|---|---|
| `BENTO_WORKERS` | BentoML process replica 수 | 병렬 요청 처리량 증가 |
| `ORT_INTRA_OP` | ONNX Runtime 인스턴스당 thread 수 | replica별 CPU 점유 제한 |
| `PIPER_CUDA` | CUDA execution provider 사용 | NVIDIA 환경에서 GPU 사용 |
| `length_scale` | 발화 속도 제어 | 음성 길이와 합성 시간 변화 |

| 관찰 | 해석 |
|---|---|
| 단일 인스턴스는 약 11-12 rps에서 평탄 | 요청 하나가 CPU를 이미 많이 사용 |
| dynamic batching 이득이 작음 | Piper가 batch forward 모델이 아님 |
| thread 제한 + worker 복제가 효과적 | replica가 CPU 코어를 더 고르게 사용 |

### 4.5 `stt-gen/` - faster-whisper STT와 양자화 벤치

| 항목 | 내용 |
|---|---|
| 목적 | Speech-to-text 전사와 int8 양자화 효과 측정 |
| 핵심 파일 | `transcribe.py`, `bench_stt.py`, `bento_service.py` |
| 입력 | WAV 파일 또는 `--from-tts` 텍스트 |
| 출력 | 전사 text, RTF, WER, memory |
| 핵심 지표 | RTF, WER, RSS memory |

**단발 전사**

```text
audio wav
  -> WhisperModel 로드
  -> warmup transcribe
  -> model.transcribe()
  -> segments 병합
  -> text / RTF 출력
```

**TTS-STT round-trip**

```text
reference text
  -> Piper TTS로 임시 WAV 생성
  -> faster-whisper 전사
  -> reference text와 hypothesis 비교
  -> WER 계산
```

**벤치**

```text
고정 문장들
  -> Piper clip 생성
  -> model x compute_type 조합 생성
  -> 조합별 subprocess 실행
  -> RTF / WER / peak RSS 수집
  -> 표 출력
```

| 설정 | 의미 |
|---|---|
| `model.name` | `base.en`, `small.en`, `medium.en` 등 Whisper 모델 크기 |
| `model.compute_type` | `float32`, `int8`, `float16`, `int8_float16` |
| `model.device` | `auto`, `cpu`, `cuda` |
| `beam_size` | 디코딩 탐색 폭 |

| 관찰 | 해석 |
|---|---|
| CPU int8은 메모리를 크게 줄임 | 더 큰 모델이나 더 많은 replica 적재에 유리 |
| CPU int8이 항상 빠르지는 않음 | float32 GEMM이 이미 최적화되어 있을 수 있음 |
| WER 변화는 작았음 | 해당 테스트셋에서는 정확도 손실이 거의 없었음 |

### 4.6 `llm-serve/` - LLM provider, 양자화, 동시성 측정

| 항목 | 내용 |
|---|---|
| 목적 | 로컬 Ollama와 클라우드 vLLM을 OpenAI 호환 API로 추상화 |
| 핵심 파일 | `chat.py`, `bench_llm.py`, `quant_sweep.py`, `config.yaml` |
| 입력 | prompt, model, base_url |
| 출력 | streaming text, TTFT, token/s, 품질 근사치 |
| 핵심 지표 | TTFT, prefill token/s, decode token/s, aggregate token/s |

**Chat**

```text
prompt
  -> OpenAI client(base_url, model)
  -> streaming chat.completions.create()
  -> 첫 delta 도착 시 TTFT 기록
  -> delta 출력
  -> decode token/s 계산
```

**동시성 벤치**

```text
concurrency level
  -> 같은 prompt를 N개 동시 요청
  -> 요청별 TTFT / total time / token count 수집
  -> aggregate token/s 계산
  -> 동시성별 표 출력
```

**양자화 스윕**

```text
quant tags(q4, q8, fp16)
  -> Ollama /api/generate로 timing 측정
  -> /api/tags로 disk size 조회
  -> /api/ps로 memory 조회
  -> fp16 출력과 단어 유사도 비교
  -> 속도 / 메모리 / 품질 표 출력
```

| 환경 | `base_url` 예 | 특징 |
|---|---|---|
| 로컬 Ollama | `http://localhost:11434/v1` | 로컬 개발/실험에 편함 |
| 클라우드 vLLM | `http://<vllm-host>:8000/v1` | continuous batching, GPU 활용에 유리 |

| 관찰 | 해석 |
|---|---|
| q4/q8이 decode 속도에도 영향 | LLM decode는 memory bandwidth-bound 성격이 강함 |
| q4는 품질 비용이 큼 | 속도와 품질의 균형은 q8 쪽이 나을 수 있음 |
| 동시성 증가 시 aggregate token/s 상승 | LLM은 continuous batching 이득을 기대할 수 있음 |

### 4.7 `voice-agent/` - STT to LLM to TTS 대화 턴

| 항목 | 내용 |
|---|---|
| 목적 | 음성 입력부터 음성 응답까지 end-to-end latency breakdown 측정 |
| 핵심 파일 | `agent.py`, `bench_latency.py`, `config.yaml` |
| 입력 | `--ask` 텍스트 또는 `--audio` WAV |
| 출력 | 답변 WAV, 단계별 지연 예산 |
| 핵심 지표 | STT latency, LLM TTFT, LLM total, TTS latency, E2E latency |

```text
--ask text 또는 --audio wav
  -> 입력 WAV 준비
  -> faster-whisper STT
  -> question text
  -> OpenAI-compatible LLM streaming
  -> answer text
  -> Piper TTS
  -> answer.wav
  -> 단계별 latency budget 출력
```

| 항목 | 의미 |
|---|---|
| STT | 사용자 음성을 텍스트로 바꾸는 시간 |
| LLM TTFT | LLM 첫 token까지의 시간 |
| LLM total | 전체 답변 text 생성 시간 |
| TTS | 답변 text를 음성으로 합성하는 시간 |
| E2E | 사용자 발화 종료부터 응답 음성 생성까지의 총합 |

| 상태 | 병목 | 의미 |
|---|---|---|
| Cold start | LLM | 모델 로드/초기화 비용이 큼 |
| Warm state | STT | 정상상태에서는 음성 전사가 더 큰 비중을 차지할 수 있음 |

### 4.8 `avatar-gen/` - Text/LLM/TTS/lip-sync 아바타 파이프라인

| 항목 | 내용 |
|---|---|
| 목적 | 말하는 아바타 영상 생성 파이프라인 검증 |
| 핵심 파일 | `pipeline.py`, `backends/base.py`, `backends/static.py`, `backends/wav2lip.py` |
| 입력 | `--text` 또는 `--prompt`, face image, backend |
| 출력 | `outputs/avatar.mp4` |
| 핵심 지표 | end-to-end 동작 여부, backend 교체 가능성, 환경 호환성 |

```text
--text
  -> TTS
  -> lip-sync backend
  -> mp4

--prompt
  -> LLM으로 말할 text 생성
  -> TTS
  -> lip-sync backend
  -> mp4
```

| Backend | 역할 | 의존성 | 용도 |
|---|---|---|---|
| `static` | 정지 이미지 + 오디오를 mp4로 합성 | ffmpeg | 전체 배선 검증, fallback |
| `wav2lip` | 외부 Wav2Lip repo를 호출 | 외부 repo, checkpoint | 실제 lip-sync |

**확인 포인트**

- `static`은 lip-sync 품질 검증용이 아니라 파이프라인 배선 검증용이다.
- 무거운 lip-sync 모델은 repo에 포함하지 않고 외부 경로로 주입한다.
- Wav2Lip/SadTalker 계열은 CUDA 가정 코드가 많아 MPS 환경에서는 추가 패치가 필요할 수 있다.

## 5. 최적화 판단 요약

| 실험 | 적용한 레버 | 관찰 | 해석 |
|---|---|---|---|
| `serve/` echo | Dynamic batching | 처리량 증가 | batch forward가 되는 상황의 기준선 |
| `tts-gen/` Piper | Dynamic batching | 처리량 이득 제한 | per-utterance CPU-bound 모델 |
| `tts-gen/` Piper | Worker replica | 처리량 증가 | thread 제한과 복제를 결합해야 효과적 |
| `stt-gen/` Whisper | int8 quantization | 메모리 감소, 속도 이득 제한 | CPU에서는 int8이 항상 빠르지 않음 |
| `llm-serve/` LLM | q4/q8 quantization | decode 속도 증가, 품질 비용 | memory bandwidth-bound decode |
| `llm-serve/` LLM | 동시성/continuous batching | aggregate token/s 증가 | LLM serving에 적합한 배칭 구조 |
| `voice-agent/` | 단계별 latency 측정 | cold와 warm 병목이 다름 | 최적화 순서는 상태별로 달라짐 |

## 6. 확장 방향

| 영역 | 다음 작업 | 확인할 지표 |
|---|---|---|
| TTS | Streaming synthesis, TTFB 측정 | 첫 audio chunk latency, RTF |
| STT | Streaming partial transcription, GPU float16/int8_float16 | partial latency, RTF, WER |
| LLM | vLLM endpoint, 큰 모델, 긴 context | TTFT, decode token/s, aggregate token/s |
| Avatar | Wav2Lip 외 SadTalker/MuseTalk/LatentSync backend 추가 | 성공률, 품질, 처리 시간 |
| Serving | Triton/BentoML/vLLM 결과와 baseline 비교 | throughput, p95/p99, resource usage |
