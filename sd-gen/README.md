# Stable Diffusion 이미지 생성기

`diffusers` 기반 로컬 이미지 생성 스크립트.

---

## 설치

```bash
cd sd-gen

# 의존성 설치 (.venv 자동 생성, Python 3.11 은 uv 가 확보)
uv sync

# NVIDIA GPU라면 xformers 추가 (VRAM 절약)
uv sync --extra gpu
```

---

## 사용법

```bash
# config.yaml 기본 프롬프트로 생성
uv run python generate.py

# 프롬프트 직접 지정
uv run python generate.py -p "astronaut riding a horse on the moon, photorealistic"

# 네거티브 프롬프트 + 크기 지정
uv run python generate.py \
  -p "portrait of a samurai, detailed, cinematic" \
  -n "blurry, cartoon, watermark" \
  -W 768 -H 768 \
  --steps 40 \
  --seed 42

# 여러 장 생성
uv run python generate.py -p "fantasy castle" -N 4
```

---

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-p` | 프롬프트 | config.yaml 값 |
| `-n` | 네거티브 프롬프트 | config.yaml 값 |
| `-W` | 너비 | 512 |
| `-H` | 높이 | 512 |
| `-s` | 추론 스텝 | 30 |
| `-g` | CFG scale | 7.5 |
| `-N` | 이미지 수 | 1 |
| `--seed` | 고정 시드 | -1(랜덤) |
| `-c` | 설정 파일 경로 | config.yaml |

---

## 모델 변경

`config.yaml`의 `model.id`를 수정:

```yaml
model:
  id: "stabilityai/stable-diffusion-xl-base-1.0"  # SDXL (고품질, VRAM 8GB+)
  id: "stabilityai/stable-diffusion-2-1"           # SD 2.1
  id: "runwayml/stable-diffusion-v1-5"             # SD 1.5 (가볍고 빠름)
  id: "./models/my-local-model"                     # 로컬 모델
```

첫 실행 시 HuggingFace에서 자동 다운로드 (`models/` 폴더에 캐시).

---

## 디바이스별 권장 설정

| 환경 | 권장 설정 |
|------|----------|
| NVIDIA GPU (VRAM 6GB+) | `device.type: cuda`, `fp16: true` |
| Apple Silicon (M1/M2/M3) | `device.type: mps`, `fp16: true` |
| CPU only | `device.type: cpu`, `fp16: false`, width/height: 512 |

---

## 출력

`outputs/` 폴더에 `sd_YYYYMMDD_HHMMSS.png` 형식으로 저장.
PNG 파일에 프롬프트·시드·모델 정보가 메타데이터로 포함됩니다.
