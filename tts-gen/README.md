# tts-gen — Piper TTS 서빙·최적화 (BentoML)

Piper(ONNX) 음성 합성을 **0→1로 서빙**하는 실험. 모델 합성 → 프레임워크(BentoML)
패키징 → RTF/동시성 벤치 → (직접 구현 baseline과 비교)까지 한 흐름으로 다룬다.

## 구성
```
tts-gen/
├── synthesize.py     # CLI: 합성 + RTF 출력 (--download 로 보이스 받기)
├── bench_rtf.py      # 문장 길이별 RTF 벤치
├── bento_service.py  # BentoML 서비스 (@bentoml.service): /synthesize, /synthesize_meta
├── bentofile.yaml    # bentoml build → containerize (Docker)
├── config.yaml
└── pyproject.toml
```
같은 모델을 직접 구현 baseline 서버로도 서빙: `../serve/config.piper.yaml`
(어댑터 `../serve/adapters/piper_tts.py`).

## 빠른 시작
```bash
uv sync                                          # NVIDIA GPU: uv sync --extra gpu
uv run python synthesize.py --download           # 보이스 1회 다운로드(en_US-lessac-medium, ~63MB)
uv run python synthesize.py -t "hello world"     # outputs/ 에 wav + RTF 출력
uv run python bench_rtf.py --runs 5              # 길이별 RTF

# BentoML 서빙
bentoml serve bento_service:PiperTTS             # http://127.0.0.1:3000 (/healthz)
curl -s -X POST localhost:3000/synthesize -H 'Content-Type: application/json' \
     -d '{"text":"served by bentoml"}' -o out.wav
# Docker 이미지: bentoml build && bentoml containerize piper-tts:latest

# 서빙 비교 (동시성 sweep)
uv run python ../serve/bench.py --url http://127.0.0.1:3000 --path /synthesize_meta --field text
```

## 측정 결과 (Apple Silicon 18코어, CPU/onnxruntime) — 자세히는 `../docs/experiments.md`
- **RTF ~0.02–0.03 (약 40–50× 실시간)**. 입력이 길수록 고정 오버헤드가 분산돼 RTF가 좋아짐.
- 단일 인스턴스 처리량은 BentoML·baseline 모두 **~11–12 rps에서 평탄** → Piper는 per-utterance·CPU
  바운드라 **동적 배칭이 처리량을 못 올린다**(진짜 배치 forward 아님 + onnxruntime이 이미 코어 포화).
- **그럼 무엇이 올리나(실측)**: 인스턴스당 `intra_op=2`로 제한 + **BentoML `workers=8`(프로세스 복제)**
  → **처리량 12.5 → 25.5 rps (2×), p50 638 → 252ms** 동시 개선. (스레드 제한만 하면 6.8 rps로 오히려
  손해 — *복제와 결합*해야 18코어를 효율적으로 채움.)
- **결론**: 배치 forward 안 되는 CPU 바운드 TTS의 처리량 레버는 **인스턴스당 스레드 제한 + replica
  스케일아웃**. BentoML `workers`로 이 복제를 한 줄로 얻는 게 프레임워크의 가치.

### 처리량 스윕 재현
```bash
# A 기본 / D 제한+복제 비교
BENTO_WORKERS=1               bentoml serve bento_service:PiperTTS --port 3000   # ~12 rps
BENTO_WORKERS=8 ORT_INTRA_OP=2 bentoml serve bento_service:PiperTTS --port 3000  # ~25 rps
uv run python ../serve/bench.py --url http://127.0.0.1:3000 --path /synthesize_meta --field text --concurrency 8 --requests 64
```

## 다음 최적화(선택)
- ONNX 그래프 양자화(int8) 후 RTF/품질 트레이드오프
- 스트리밍 합성(첫 오디오까지 TTFB) 측정
- workers×intra 더 넓은 스윕으로 처리량 상한 탐색
