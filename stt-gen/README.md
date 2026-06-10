# stt-gen — faster-whisper STT 서빙·양자화 최적화 (BentoML)

faster-whisper(CTranslate2) 음성 인식을 **0→1로 서빙**하며, **int8 양자화**의
정확도(WER)·속도(RTF)·메모리 트레이드오프를 실측한다. Piper(TTS) 출력을 입력으로 써
**TTS↔STT 음성 왕복**으로 검증한다(외부 오디오 불필요, 정답을 알아 WER 계산).

## 구성
```
stt-gen/
├── transcribe.py     # CLI: --from-tts "..." (합성→전사) / -a clip.wav
├── bench_stt.py      # 모델크기 × compute_type(int8/float32) → RTF·WER·메모리 스윕
├── bento_service.py  # BentoML 서비스: POST /transcribe (wav 업로드)
├── bentofile.yaml    # bentoml build → containerize
├── config.yaml
└── requirements.txt
```
같은 모델을 직접 구현 baseline 서버로도 서빙: 어댑터 `../serve/adapters/whisper_stt.py`.

## 빠른 시작
```bash
pip install -r requirements.txt
python transcribe.py --from-tts "hello from whisper"        # TTS↔STT 왕복 + RTF/WER
python bench_stt.py --models base.en small.en medium.en --compute-types float32 int8

# BentoML 서빙 (모델/양자화는 환경변수)
STT_MODEL=base.en STT_COMPUTE=int8 bentoml serve bento_service:WhisperSTT  # :3000
curl -s -X POST localhost:3000/transcribe -F 'audio=@clip.wav'
```

## 측정 결과 (Apple Silicon 18코어, CPU) — 자세히는 `../docs/experiments.md`
모델크기 × compute_type, TTS 클립 4개(총 ~14s):

| model | compute | RTF | WER | mem(MB) |
|--|--|--:|--:|--:|
| base.en | float32 | 0.086 | 0.053 | 836 |
| base.en | int8 | 0.095 | 0.053 | **702** |
| small.en | float32 | 0.177 | 0.053 | 2162 |
| small.en | int8 | 0.245 | 0.053 | **1149** |
| medium.en | float32 | 0.515 | 0.053 | 4623 |
| medium.en | int8 | 0.702 | 0.053 | **2094** |

- **int8의 이득은 속도가 아니라 메모리** (~50% 감소, medium 4.6GB→2.1GB). CPU에선 int8이 오히려
  약간 느림(양자화/역양자화 오버헤드, float32 GEMM이 이미 잘 최적화됨). **WER는 동일(정확도 손실 없음)**.
- 의미: 메모리 반감 → **GPU당 더 많은 replica/더 큰 모델을 적재**(Phase 1의 '복제' 처리량 레버와 직결).
  속도 이득의 int8은 보통 **GPU(int8 텐서코어)**에서 나타남 → 클라우드 단계에서 확인 예정.
- Phase 1(TTS)의 '복제' 레버와 묶으면: *"각 최적화는 도움 되는 축이 다르다(throughput vs memory),
  가정 말고 측정한다"* 는 서빙 판단으로 이어짐.

## 다음(선택)
- GPU에서 int8 속도 이득 측정(클라우드), large-v3 양자화, 스트리밍(부분 결과) 전사
