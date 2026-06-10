#!/usr/bin/env python3
"""추론 서버 (FastAPI).

큐 → 동적 마이크로배칭(Scheduler) → 어댑터 추론 → 응답. 헬스체크, 메트릭,
그레이스풀 셧다운, 요청 취소(클라이언트 disconnect 시 코루틴 취소) 지원.

실행:
    pip install -r requirements.txt
    python server.py                 # config.yaml 사용
    # 또는
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI
from pydantic import BaseModel

from adapters import build_adapter
from scheduler import Scheduler


def load_config(path: Optional[str] = None) -> dict:
    # 우선순위: 인자 > SERVE_CONFIG 환경변수 > 기본 config.yaml
    chosen = path or os.environ.get("SERVE_CONFIG")
    cfg_path = Path(chosen) if chosen else Path(__file__).parent / "config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = Path(__file__).parent / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
STATE: dict = {"scheduler": None, "started_at": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    sch_cfg = CONFIG.get("scheduler", {})
    adapter = build_adapter(CONFIG.get("adapter", {"name": "echo"}))
    scheduler = Scheduler(
        adapter,
        max_batch_size=int(sch_cfg.get("max_batch_size", 8)),
        max_wait_ms=float(sch_cfg.get("max_wait_ms", 10.0)),
        workers=int(sch_cfg.get("workers", 1)),
    )
    await scheduler.start()
    STATE["scheduler"] = scheduler
    STATE["started_at"] = time.time()
    print(f"[serve] adapter='{adapter.name}' batch<={scheduler.max_batch_size} "
          f"wait={scheduler.max_wait_ms}ms workers={sch_cfg.get('workers', 1)}")
    try:
        yield
    finally:
        await scheduler.stop()  # 그레이스풀 셧다운
        print("[serve] stopped")


app = FastAPI(title="multimodal-serving-lab serving spine", lifespan=lifespan)


class InferRequest(BaseModel):
    input: Any


class InferResponse(BaseModel):
    output: Any
    batch_size: int
    latency_ms: float


@app.get("/health")
async def health():
    sch: Scheduler = STATE["scheduler"]
    return {
        "status": "ok" if sch is not None else "starting",
        "model": sch.adapter.name if sch else None,
        "queue_size": sch.queue_size if sch else None,
        "uptime_s": round(time.time() - STATE["started_at"], 1) if STATE["started_at"] else None,
    }


@app.get("/metrics")
async def metrics():
    sch: Scheduler = STATE["scheduler"]
    if sch is None:
        return {"status": "starting"}
    return {
        "total_requests": sch.total_requests,
        "total_batches": sch.total_batches,
        "avg_batch_size": round(sch.avg_batch_size, 3),
        "max_observed_batch": sch.max_observed_batch,
        "queue_size": sch.queue_size,
    }


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    sch: Scheduler = STATE["scheduler"]
    t0 = time.perf_counter()
    # 클라이언트가 끊으면 이 코루틴이 취소되어 자동으로 요청이 버려진다(요청 취소).
    output, bsz = await sch.submit({"input": req.input})
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return InferResponse(output=output, batch_size=bsz, latency_ms=round(latency_ms, 2))


def main():
    import uvicorn
    srv = CONFIG.get("server", {})
    uvicorn.run(app, host=srv.get("host", "127.0.0.1"), port=int(srv.get("port", 8000)),
                log_level="warning")


if __name__ == "__main__":
    main()
