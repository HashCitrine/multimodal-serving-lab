# multimodal-serving-lab

생성·멀티모달 AI 모델을 로컬 환경에서 직접 실행·서빙·최적화해 본 기록을 정리한 저장소입니다.

이미지/비디오에서 시작해 음성(TTS/STT), LLM, 토킹헤드 아바타까지 모달리티를 넓히며, 각 모델을 **서빙 프레임워크(BentoML / Triton / vLLM)** 로 패키징·최적화·배포하는 0→1 과정을 실험합니다. 별도의 `serve/`는 서빙 런타임의 **내부 동작(동적 배칭·큐·지연/처리량)을 직접 구현해 이해**하기 위한 학습용 baseline이며, 프레임워크 결과와 나란히 비교하는 기준선으로 씁니다. Apple Silicon/MPS를 기본으로 하고, 무거운 실험은 클라우드 GPU에서 진행합니다. 완성된 제품이 아니라 "무엇이 가능했고 어떤 한계가 있었는지"를 남기는 실험 아카이브입니다.

모델 파일, 가상환경, 생성 결과물(이미지/영상/오디오/모션)은 저장소에 포함하지 않습니다.

## 구성

- `serve/`: **서빙 내부 원리 학습용 baseline + 벤치마크 하니스** — 직접 구현한 큐·동적 배칭·메트릭(프레임워크 비교 기준선). 실제 서빙은 BentoML/Triton/vLLM 사용
- `sd-gen/`: Stable Diffusion 기반 이미지 생성, ComfyUI 연동, 단발 생성, 업스케일
- `video-gen/`: AnimateDiff, Zeroscope, LTX 계열 비디오 생성
- `tts-gen/`: 음성 합성(Piper) 서빙·최적화 — BentoML 패키징, RTF·동시성 벤치 (Phase 1 완료)
- `stt-gen/`: 음성 인식(faster-whisper) 서빙·최적화 — int8 양자화 RTF/WER/메모리, TTS↔STT 왕복 (Phase 2 완료)
- `llm-serve/`: LLM 서빙·양자화 최적화 — 로컬 Ollama(OpenAI 호환)↔클라우드 vLLM 전환 구조, 양자화/배칭 벤치 (Phase 3 완료)
- `voice-agent/`: STT→LLM→TTS 음성 에이전트 — 대화 턴 end-to-end 지연 예산 측정(병목: warm=STT, cold=LLM)
- `avatar-gen/`: 토킹헤드·립싱크 아바타 파이프라인(text→LLM→TTS→lip-sync) — 사전 스캐폴드, static 백엔드로 지금 동작·립싱크 모델 교체형 (Phase 4)
- `docs/project-flow-and-terms.md`: 흐름·용어·실험 기록 문서 안내
- `docs/project-flow.md`: 전체 Phase 흐름·서브 프로젝트별 로직 해설
- `docs/glossary.md`: 멀티모달·서빙·최적화 핵심 용어 사전
- `docs/experiments.md`: 시도한 내용·결과·한계, 벤치 수치 기록

## 실행 환경 (device 이식성)
모든 서브 프로젝트는 **device 파라미터화**(Apple Silicon/MPS·CPU 기본, NVIDIA/CUDA 전환 가능):
- TTS(Piper): `--cuda` 또는 `PIPER_CUDA=1` + `onnxruntime-gpu`
- STT(faster-whisper): `device: auto|cuda` (GPU는 `compute_type: float16|int8_float16`)
- LLM: OpenAI 호환 `base_url` 교체(로컬 Ollama ↔ 클라우드 vLLM)
- avatar: `device: auto`(cuda→mps→cpu) 자동 감지

## 실행 예시

각 디렉터리에 별도의 `README.md`와 `requirements.txt`가 있습니다.

이미지 생성:

```bash
cd sd-gen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate.py -p "a cinematic mountain landscape at sunset"
```

비디오 생성:

```bash
cd video-gen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate.py -p "a cat walking in a garden"
```

## 정리

- 모델 캐시는 각 디렉터리의 `models/` 아래에 생성되며 Git에 포함하지 않습니다.
- 생성된 이미지와 영상은 `outputs/` 아래에 저장되며 Git에 포함하지 않습니다.
- Apple Silicon MPS에서는 `float32`가 더 안정적이었습니다. `fp16`은 검은 이미지나 불안정한 결과가 나오는 경우가 있었습니다.
- 이미지 생성은 로컬 실험용으로 충분히 쓸 만했지만, 비디오 생성은 메모리, 해상도, 프레임 수, 모델 성숙도 때문에 품질 한계가 컸습니다.
- `img2img`로 프레임을 이어 붙이는 방식은 작은 카메라 움직임에는 쓸 수 있지만, 걷기 같은 실제 동작을 만들기에는 한계가 있었습니다.
