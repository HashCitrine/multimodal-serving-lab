# avatar-gen — 토킹헤드/립싱크 아바타 파이프라인

`text →(LLM)→ TTS →(lip-sync)→ mp4` 를 한 파이프라인으로 묶는다. LLM은 OpenAI 호환 provider, TTS는 Piper, 립싱크는 교체형 backend를 사용한다. `backend=static`은 정지 영상 폴백이고, `backend=wav2lip`은 외부 Wav2Lip repo/checkpoint를 호출한다.

모델 파일, 얼굴 입력, 오디오, 생성 mp4는 저장소에 포함하지 않는다. Wav2Lip 오픈소스 모델은 연구/개인 실험 용도로만 다룬다.

## 빠른 시작

```bash
pip install -r requirements.txt          # + 시스템 ffmpeg
python pipeline.py --text "Hello, I am your tutor." --backend static
```

실제 립싱크는 외부 Wav2Lip 환경을 준비한 뒤 실행한다.

```bash
export WAV2LIP_DIR=/path/to/Wav2Lip
export WAV2LIP_CKPT=/path/to/Wav2Lip/checkpoints/wav2lip_gan.pth
python pipeline.py --text "Hello, I am your tutor." --face /path/to/face.jpg --backend wav2lip --device cuda
```

## Wav2Lip 준비

Wav2Lip 본체와 weights는 별도 설치한다. 공개 repo에는 경로만 주입한다.

- repo: `Rudrabha/Wav2Lip` 또는 Python 3.10 대응 fork
- main checkpoint: `wav2lip_gan.pth` 또는 `wav2lip.pth`
- face detector weight: `face_detection/detection/sfd/s3fd.pth`
- 시스템 dependency: `ffmpeg`
- CUDA 환경: PyTorch CUDA wheel + Wav2Lip requirements

원본 inference 형식은 다음과 같다.

```bash
cd /path/to/Wav2Lip
python inference.py \
  --checkpoint_path checkpoints/wav2lip_gan.pth \
  --face /path/to/face.jpg \
  --audio /path/to/speech.wav \
  --outfile result.mp4
```

`config.yaml`의 `pads`, `resize_factor`, `nosmooth`, `fps`, `gpu_id`를 통해 inference 옵션을 넘길 수 있다. 입 위치가 어긋나면 `pads`와 `nosmooth`를 먼저 조정한다.

## 벤치

`bench_avatar.py`는 같은 입력으로 audio length, TTS latency, lip-sync latency, lip-sync RTF, end-to-end latency, NVIDIA GPU peak memory를 측정한다.

```bash
python bench_avatar.py --backend static --runs 3
python bench_avatar.py --backend wav2lip --face /path/to/face.jpg --device cuda --runs 3 --gpu-id 0
```

결과 mp4/wav는 `outputs/`에 생성되며 Git에 포함하지 않는다.

## 구성

```text
avatar-gen/
├── pipeline.py        # text→LLM→TTS→lipsync, 단계별 latency 출력
├── bench_avatar.py    # static/Wav2Lip backend latency·RTF·GPU memory 벤치
├── config.yaml        # llm/tts/lipsync backend 설정
├── requirements.txt
└── backends/
    ├── base.py        # LipSyncBackend 인터페이스
    ├── static.py      # ffmpeg 정지영상 폴백
    └── wav2lip.py     # 외부 Wav2Lip 셸아웃
```

## 위치 설정

이 실험의 목표는 제품 품질 아바타가 아니라, 음성 합성과 얼굴 입력을 실제 lip-sync 모델까지 연결하고 병목을 측정하는 것이다. Mac/MPS에서 막히던 레거시 의존성 문제는 CUDA 환경에서 우회할 수 있으므로, RTX 계열 GPU에서는 Wav2Lip 실행과 RTF 측정에 집중한다.
