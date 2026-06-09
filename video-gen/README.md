# video-gen

Apple Silicon MPS 환경에서 실행해 본 로컬 비디오 생성 실험 스크립트입니다.

## 지원 모델

### AnimateDiff v3 (기본값)
- Stable Diffusion v1.5 기반 텍스트-to-비디오 생성
- Motion Adapter를 통한 자연스러운 동작 생성
- 기본 설정: 80프레임, 8fps → **10초 영상**
- 512x512 해상도

### Zeroscope V2 576w
- 전용 비디오 생성 모델
- 기본 설정: 24프레임, 8fps → **3초 영상**
- 576x320 해상도 (16:9)

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 사용법

```bash
# 기본 (AnimateDiff, 80프레임, 10초 영상)
./venv/bin/python generate.py -p "a cat walking in a garden"

# Zeroscope 모델 사용
./venv/bin/python generate.py -p "ocean waves crashing on rocks" -m zeroscope

# 커스텀 설정
./venv/bin/python generate.py -p "a girl dancing in the rain" \
  --steps 30 --guidance 8.0 --frames 48 --fps 12 --seed 42

# 네거티브 프롬프트 + 출력 파일 지정
./venv/bin/python generate.py \
  -p "cinematic drone shot of mountains at sunset" \
  -n "blurry, low quality, distorted" \
  -o outputs/mountains.mp4

# Zeroscope 고품질 설정
./venv/bin/python generate.py -p "astronaut floating in space" \
  -m zeroscope --steps 40 --guidance 12.0
```

## Mac M4 예상 소요 시간

| 모델 | 프레임 | 해상도 | 스텝 | 예상 시간 |
|------|--------|--------|------|-----------|
| AnimateDiff | 80 | 512x512 | 25 | 5~10분 |
| AnimateDiff | 16 | 512x512 | 25 | 1~2분 |
| Zeroscope | 24 | 576x320 | 25 | 3~7분 |

> 첫 실행 시 모델 다운로드로 추가 시간 소요 (AnimateDiff motion adapter ~1.5GB, Zeroscope ~3.5GB)

## 주의사항

- **MPS float32 필수**: Apple Silicon MPS에서 float16 사용 시 블랙 이미지 버그 발생. `torch_dtype=torch.float32`로 고정됨
- **메모리**: 16GB RAM 기준 AnimateDiff 80프레임이 한계. OOM 발생 시 `--frames` 줄이기
- **첫 실행**: 모델 다운로드 필요 (HuggingFace에서 자동 다운로드, `models/` 폴더에 캐시)
- **SD v1.5 모델**: AnimateDiff의 base model은 기본적으로 `../sd-gen/models/` 캐시를 재사용
- **출력**: `outputs/` 폴더에 `YYYYMMDD_HHMMSS.mp4` 형식으로 자동 저장

## 설정 파일

`config.yaml`에서 기본값 변경 가능. CLI 옵션이 config 값보다 우선.
