# serve — 서빙 내부 원리 학습용 baseline + 벤치마크 하니스

> **포지셔닝(중요)**: 이건 프로덕션 서빙 스택이 **아닙니다**. 실제 모델 서빙은
> **BentoML / Triton / vLLM** 같은 프레임워크로 합니다(루트 README 참고).
> 이 폴더는 dynamic batching·큐·지연/처리량 트레이드오프 같은 **서빙 런타임의 내부
> 동작을 직접 구현해 계측·이해**하기 위한 학습용 baseline입니다. 프레임워크의 배칭
> 설정(예: Triton `dynamic_batching`, vLLM continuous batching)을 *제대로 쓰기 위한*
> 토대 지식이며, 프레임워크 결과와 **나란히 비교**하는 기준선으로 재사용합니다.

고수준 서빙 프레임워크를 쓰지 않고 추론 서버의 핵심 요소를 직접 구현했습니다.
어떤 모달리티든 **어댑터**로 끼워 동일하게 벤치할 수 있습니다.

## 구성
```
serve/
├── server.py            # FastAPI: 큐→동적 배칭→어댑터→응답, /health /metrics /infer
├── scheduler.py         # 동적 마이크로배칭 스케줄러 (FastAPI와 분리, 단독 테스트 가능)
├── bench.py             # 동시성 sweep 벤치 (p50/p95/p99, throughput, 평균 배치 크기)
├── config.yaml          # 서버/스케줄러/어댑터 설정
└── adapters/
    ├── base.py          # ModelAdapter 인터페이스 (load/warmup/infer/unload)
    ├── echo.py          # 더미 어댑터 (모델 없이 스파인 검증용)
    └── __init__.py      # 이름→어댑터 레지스트리 (tts/stt/llm/avatar 여기 등록)
```

## 핵심 개념
- **동적 마이크로배칭**: 첫 요청 후 `max_wait_ms` 동안(또는 `max_batch_size` 도달까지)
  뒤따르는 요청을 모아 한 번에 추론 → 처리량↑. 블로킹 추론은 ThreadPool에서 실행해
  이벤트 루프를 막지 않음.
- **어댑터 인터페이스**: 모달리티 교체를 표준화(런타임 모델 교체와 동일한 발상).
- **메트릭**: 평균/최대 배치 크기, 큐 길이, 총 배치 수 등을 `/metrics`로 노출.
- **그레이스풀 셧다운 / 요청 취소**: 종료 시 스케줄러 정리, 클라이언트 disconnect 시 코루틴 취소.

## 실행
```bash
pip install -r requirements.txt
python server.py                       # config.yaml (기본 echo 어댑터)
# 다른 터미널에서
curl -s localhost:8000/health
python bench.py --concurrency 1 2 4 8 --requests 64
```

## 메모
- 지금은 `echo` 더미 어댑터만 등록되어 있어 무거운 모델 없이도 스파인을 검증할 수 있습니다.
- 이후 단계에서 `tts/stt/llm/avatar` 어댑터를 `adapters/`에 추가하고 `REGISTRY`에 등록하면,
  `config.yaml`의 `adapter.name`만 바꿔 서빙 대상을 교체할 수 있습니다.
- 실측 벤치 수치는 `../docs/experiments.md`에 기록합니다(로컬/학습 환경 기준).
