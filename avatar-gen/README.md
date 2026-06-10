# avatar-gen — 토킹헤드/립싱크 아바타 파이프라인 (Phase 4, 사전 스캐폴드)

`text →(LLM)→ TTS →(lip-sync)→ mp4` 를 한 파이프라인으로 묶는다. 앞 Phase 자산을 재사용:
LLM(llm-serve provider), TTS(tts-gen Piper), 립싱크(교체형 backend). **device 자동 감지(cuda→mps→cpu).**

## 지금 바로 되는 것 / 나중에 붙이는 것
- ✅ **지금**: `backend=static`(ffmpeg)으로 text→LLM→TTS→영상 **전 구간이 동작**(검증됨). 입모양은 정지.
- 🔌 **나중**: `backend=wav2lip` 등 실제 립싱크 모델을 외부 설치 후 config만 바꾸면 입모양까지. (모델은 무거워
  레포에 포함하지 않음 — 경로만 주입.)

```bash
pip install -r requirements.txt          # + 시스템 ffmpeg
python pipeline.py --prompt "Greet a learner in one sentence." --backend static
python pipeline.py --text "Hello!" --face me.jpg --backend wav2lip   # 모델 준비 후
```

## "Mac이라 안 되는 것" 아님 — NVIDIA를 흔히 쓰는 진짜 이유
SadTalker/Wav2Lip 류가 NVIDIA에서 많이 도는 건 **성능이 아니라 생태계(CUDA) 때문**이다:
- 오래된 코드(2020~2023)라 특정 torch/CUDA 버전·커스텀 CUDA 커널·`insightface`/`gfpgan` 등이 **CUDA 가정**
- 스크립트가 `--device cuda` 하드코딩, MPS 분기 미고려

즉 **M5 Max 성능이 부족한 게 아니라, 레포가 Apple Silicon을 1급 지원하지 않아 손이 더 간다.** 성능만 보면 충분.

### 환경별 현실
| 백엔드 | Mac(MPS) | NVIDIA | 비고 |
|--|--|--|--|
| static(ffmpeg) | ✅ | ✅ | 립싱크 아님(파이프라인 검증/폴백) |
| Wav2Lip | △ 가능(의존성·device 패치) | ✅ 그대로 | 순수 PyTorch라 MPS 시도 가치 큼 |
| SadTalker | △ 까다로움(gfpgan/face-align 패치) | ✅ | 의존성 무거움 |
| MuseTalk / LatentSync | △~✕ | ✅ | 최신·무거움, GPU 권장 |

### 준비 절차 (Wav2Lip) — 외부 레포 경로 주입
Wav2Lip을 사용하려면 외부 레포와 체크포인트를 별도로 준비한 뒤 `config.yaml`에 경로를 주입한다.
- 레포: `<external-wav2lip-dir>` (justinjohn0306 포크, RetinaFace 얼굴검출)
- 체크포인트: `checkpoints/wav2lip_gan.pth`(436MB), `checkpoints/mobilenet.pth`(RetinaFace)
- 의존성: librosa·opencv·numba·tqdm·batch-face·torchvision
- torch 2.5.1 (weights_only 이슈 없음), MPS 사용 가능

**실행 예시**
```bash
# 1) 직접 검증 (CPU. 5초 음성 기준 1~3분)
cd <external-wav2lip-dir>
python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth \
    --face <실제얼굴.jpg> --audio <speech.wav> --outfile result.mp4
# 2) avatar-gen 파이프라인으로 통합 실행 (config 의 wav2lip_dir/ckpt 주석 해제 후)
cd multimodal-serving-lab/avatar-gen
python pipeline.py --text "Hello, I am your tutor." --face <실제얼굴.jpg> --backend wav2lip
```
- **MPS 가속**: `inference.py` 148행 `device = 'cuda' if torch.cuda.is_available() else 'cpu'` 을
  `... else ('mps' if torch.backends.mps.is_available() else 'cpu')` 로 패치. (일부 op가 MPS 미지원이면
  CPU 폴백 — 그땐 CPU로 충분히 동작.)
- **주의**: RetinaFace 는 **실제 얼굴 사진**에서만 검출됨(그림/placeholder 불가). 정면 얼굴 권장.

## 솔직한 위치 설정
이 단계는 **최적화 경험이 아니라 '무거운 모델을 환경에 맞춰 돌리는 통합/트러블슈팅' 경험**이다(특히
레거시 모델의 MPS 포팅). Phase 1~3의 서빙·최적화 서사와는 결이 다르며, 모션 모달리티 커버리지를 위한 것.

## 구성
```
avatar-gen/
├── pipeline.py        # text→LLM→TTS→lipsync, device 자동감지
├── config.yaml        # llm/tts/lipsync(backend) 설정
├── requirements.txt
└── backends/
    ├── base.py        # LipSyncBackend 인터페이스
    ├── static.py      # ffmpeg 정지영상(지금 동작)
    └── wav2lip.py     # 외부 Wav2Lip 셸아웃(준비되면 동작)
```
