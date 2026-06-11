# 핵심 용어 사전

`multimodal-serving-lab`에서 반복적으로 등장하는 멀티모달, 서빙, 최적화 용어를 정리한다.
프로젝트 흐름은 [`project-flow.md`](project-flow.md)를 함께 참고한다.

## 사용 방법

| 상황 | 보는 방법 |
|---|---|
| 흐름 문서에서 모르는 단어가 나올 때 | 해당 용어를 이 문서에서 검색 |
| 새 서브 프로젝트를 추가할 때 | 입력/출력 지표를 어떤 용어로 기록할지 맞춤 |

## 모달리티와 모델 작업

| 용어 | 의미 | 이 저장소에서의 예 |
|---|---|---|
| 모달리티 | 모델이 다루는 데이터 형식 | 텍스트, 이미지, 비디오, 음성, 아바타 |
| 멀티모달 | 여러 모달리티를 한 파이프라인에서 연결 | `voice-agent/`, `avatar-gen/` |
| TTS | Text-to-Speech, 텍스트를 음성으로 변환 | Piper |
| STT | Speech-to-Text, 음성을 텍스트로 변환 | faster-whisper |
| LLM | Large Language Model, 텍스트 생성 모델 | Ollama/vLLM provider |
| Lip-sync | 음성에 맞춰 얼굴 입모양을 동기화 | Wav2Lip backend |
| Talking head | 얼굴 이미지/영상이 말하는 형태의 결과물 | `avatar-gen/outputs/avatar.mp4` |

## 서빙 구성 요소

| 용어 | 의미 | 코드 위치 |
|---|---|---|
| Inference | 학습된 모델로 결과를 생성하는 추론 | `generate.py`, `synthesize.py`, `transcribe.py` |
| Serving | 추론을 API/서비스 형태로 제공 | `bento_service.py`, `serve/server.py` |
| Runtime | 요청 처리, 배칭, 실행, 응답까지의 실행 환경 | `serve/`, BentoML, Ollama, vLLM |
| Queue | 요청을 임시로 쌓아 두는 버퍼 | `serve/scheduler.py` |
| Scheduler | 요청을 언제 어떤 batch로 실행할지 결정 | `serve/scheduler.py` |
| Adapter | 모델별 추론 함수를 공통 인터페이스로 감싼 래퍼 | `serve/adapters/` |
| Worker | 실제 추론을 수행하는 실행 단위 | BentoML workers, ThreadPool workers |
| Replica | 같은 서비스 인스턴스를 여러 개 띄운 복제본 | `BENTO_WORKERS` |

## 프레임워크와 런타임

| 이름 | 역할 | 이 저장소에서의 위치 |
|---|---|---|
| FastAPI | Python API 서버 프레임워크 | `serve/server.py` |
| BentoML | 모델 API/컨테이너 패키징 프레임워크 | `tts-gen/`, `stt-gen/` |
| Triton | 범용 inference server | 직접 구현은 없고 비교 개념으로 언급 |
| Ollama | 로컬 LLM 실행 runtime | `llm-serve/config.yaml` |
| vLLM | LLM serving 특화 runtime | 클라우드 전환 대상 |
| OpenAI 호환 API | OpenAI 형식의 `/v1/chat/completions` API | Ollama와 vLLM 전환 지점 |

## 배칭과 처리량

| 용어 | 의미 | 주의점 |
|---|---|---|
| Dynamic batching | 짧은 시간 동안 요청을 모아 batch inference 수행 | 모델이 batch forward를 지원해야 효과가 큼 |
| Micro-batching | 작은 시간 창 안에서 소규모 batch를 만드는 방식 | `max_wait_ms`와 `max_batch_size`가 핵심 |
| Continuous batching | LLM 요청을 생성 중에도 계속 섞어 처리 | vLLM 같은 LLM runtime에서 중요 |
| Batch forward | 여러 입력을 한 번의 forward pass로 처리 | Piper처럼 내부 순차 처리면 이득이 작음 |
| Throughput | 단위 시간당 처리량 | req/s, token/s |
| Latency | 요청 하나의 응답 시간 | p50/p95/p99로 분포를 봐야 함 |

## 측정 지표

| 지표 | 풀네임 | 의미 | 사용 위치 |
|---|---|---|---|
| RTF | Real-Time Factor | 처리 시간 / 오디오 길이. 1보다 작으면 실시간보다 빠름 | `tts-gen/`, `stt-gen/` |
| WER | Word Error Rate | STT 결과와 정답 문장의 단어 오류율 | `stt-gen/` |
| TTFT | Time To First Token | LLM 요청 후 첫 token까지 걸린 시간 | `llm-serve/`, `voice-agent/` |
| token/s | Tokens per second | LLM prefill/decode 처리 속도 | `llm-serve/` |
| p50 | 50th percentile | latency 중앙값 | `serve/bench.py` |
| p95/p99 | Tail latency | 느린 요청 구간의 지연 | `serve/bench.py` |
| RSS | Resident Set Size | 프로세스가 실제 점유한 메모리 | `stt-gen/bench_stt.py` |

## 최적화 관련 용어

| 용어 | 의미 | 프로젝트에서 확인한 점 |
|---|---|---|
| Quantization | 가중치/연산 정밀도를 낮춰 메모리나 속도를 개선 | 효과는 모델/하드웨어별로 다름 |
| int8 | 8-bit integer 양자화 | STT CPU에서는 메모리 절감 효과가 컸음 |
| q4/q8 | LLM 저비트 양자화 태그 | LLM decode 속도에도 영향을 줬음 |
| fp16 | 16-bit floating point | LLM 품질 기준 또는 CUDA에서 자주 사용 |
| CPU-bound | CPU 연산량이 병목 | Piper TTS |
| Memory bandwidth-bound | 메모리 읽기 속도가 병목 | LLM decode |
| Cold start | 첫 요청에서 모델 로드/캐시 비용이 드는 상태 | voice-agent 첫 턴 |
| Warmup | 더미 요청으로 초기 비용을 미리 흡수 | TTS/STT/LLM 공통 |
