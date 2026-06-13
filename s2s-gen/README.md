# s2s-gen — 음성 대화 파이프라인: 캐스케이드 · 표현형 TTS · full-duplex

음성 대화 파이프라인은 두 갈래다. **캐스케이드**(STT→LLM→TTS)는 단계마다 추적 로그가 남아
관찰가능성·디버깅이 좋고, **음성 네이티브(S2S)**는 단계 경계가 없어 지연·자연스러움에 강하다.
이 모듈은 파일→파일(`in_wav → out_wav + metrics`) 백엔드로 둘을 같은 축에서 비교하고, full-duplex
모델은 별도 라이브 경로로 붙인다.

로컬 실측으로 확정된 현실:

- **`cascade`** — STT→LLM→TTS(Piper). 무거운 모델 없이 어디서나 동작하는 기준선.
- **`csm`** — Sesame CSM(`sesame/csm-1b`)은 단독 S2S가 아니라 **conversational TTS**다. 그래서
  cascade 의 STT+LLM 은 그대로 쓰고 **TTS 단계만 CSM 으로 교체**한다(Piper 대비 운율·표현력 비교).
  Apple Silicon **MPS** 에서 로컬 실행된다.
- **`moshi`** — Kyutai Moshi(full-duplex)는 파일→파일 오프라인 API가 없다. 로컬 실행은 **라이브 웹
  데모뿐**이라 이 모듈의 backend 로는 돌리지 않고, lab-ui "Moshi 라이브" 카드 또는 아래 명령으로 띄운다.

## 빠른 시작 (cascade)

```bash
uv sync
# 마이크 없이 자체 검증: 질문을 Piper로 합성해 입력으로 사용
uv run python s2s.py --backend cascade --ask "How do you spell necessary?"
```

cascade 는 로컬 Ollama(`ollama serve`)와 tts-gen Piper 보이스를 사용한다(앞선 서빙 실험 자산 재사용).

## CSM(표현형 TTS) 준비 — 로컬 MPS 실행

CSM 본체(레포+가중치)는 무겁고 HF 게이트(라이선스 동의 필요)라 저장소에 포함하지 않는다.
외부 clone 경로(`CSM_DIR`)와 HF 로그인이 필요하다.

```bash
# 1) 게이트 동의(각 페이지에서 'Agree and access repository') + 로그인
#    https://huggingface.co/sesame/csm-1b
#    https://huggingface.co/meta-llama/Llama-3.2-1B
huggingface-cli login

# 2) CSM repo 클론 (가중치는 첫 실행 시 HF에서 받음)
git clone https://github.com/SesameAILabs/csm ../_external/csm

# 3) CSM 의존성 설치(분리 extra — cascade 환경은 영향 없음)
uv sync --extra csm
uv pip install -r ../_external/csm/requirements.txt   # moshi/torchtune/torchao/silentcipher 보강

# 4) 실행 (STT→LLM→CSM, MPS)
CSM_DIR=../_external/csm uv run python s2s.py --backend csm --ask "How do you spell necessary?"
```

`config.yaml` 의 `csm_dir`/`csm_speaker`/`csm_max_audio_ms`, 환경변수 `CSM_DIR` 로 제어한다.
준비가 안 됐으면(토큰·게이트·clone 누락) 실행 전 친절한 안내가 뜬다. 현재 `sesame/csm-1b` 는
config.json 이 transformers 형식이라, 백엔드가 `ckpt.pt` + 표준 ModelArgs 로 직접 로딩해 우회한다.

## Moshi(full-duplex) 준비 — 라이브 웹 데모

Moshi 는 실시간 풀듀플렉스라 파일→파일이 아니다. lab-ui "Moshi 라이브" 카드로 기동하거나:

```bash
uv run --python 3.12 --with moshi_mlx python -m moshi_mlx.local_web -q 4
# → http://localhost:8998 에서 마이크로 실시간 대화 (말 끊고 들어가기/맞장구)
```

가중치(`kyutai/moshika-mlx-q4`)는 첫 기동 시 자동 다운로드된다(비게이트).

## 벤치

`bench_s2s.py`는 같은 질문 세트로 backend(cascade|csm)의 TTFA·E2E·RTF 중앙값을 측정한다.

```bash
uv run python bench_s2s.py --backend cascade
CSM_DIR=../_external/csm uv run python bench_s2s.py --backend csm
```

결과 wav 는 `outputs/`에 생성되며 Git 에 포함하지 않는다.

## 구성

```text
s2s-gen/
├── s2s.py             # 단일 턴 CLI(cascade|csm): in_wav→out_wav, 지연 예산 출력
├── bench_s2s.py       # TTFA·E2E·RTF 중앙값 벤치
├── config.yaml        # backend 선택 + cascade(STT/LLM/TTS) + csm 슬롯
└── backends/
    ├── base.py        # S2SBackend(abstract): generate(in_wav,out_wav)->metrics
    ├── cascade.py     # STT→LLM→TTS 기준선(_synthesize 교체점)
    ├── csm.py         # cascade 상속 + TTS만 CSM(표현형, MPS)
    └── moshi.py       # full-duplex 라이브 전용 안내 스텁(lab-ui 런처 사용)
```
