# vision-ai-experiments

이미지 생성과 비디오 생성 모델을 로컬 환경에서 실험한 기록을 정리한 저장소입니다.

Stable Diffusion, AnimateDiff, Zeroscope, LTX 계열 파이프라인을 Apple Silicon/MPS 환경에서 실행해 보며 만든 스크립트와 메모를 보관합니다. 완성된 제품이나 범용 라이브러리라기보다는, 어떤 방식이 가능했고 어떤 한계가 있었는지 남기기 위한 실험 아카이브입니다.

모델 파일, 가상환경, 생성 결과물은 저장소에 포함하지 않습니다.

## 구성

- `sd-gen/`: Stable Diffusion 기반 이미지 생성, ComfyUI 연동 실험, 단발 생성, 업스케일 스크립트
- `video-gen/`: AnimateDiff, Zeroscope, LTX 계열 비디오 생성 실험
- `docs/experiments.md`: 이전에 시도한 내용과 결과, 한계 정리

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
