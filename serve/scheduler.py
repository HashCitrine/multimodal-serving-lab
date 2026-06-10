"""동적 마이크로배칭 스케줄러.

FastAPI와 분리된 순수 asyncio 컴포넌트라 단독 테스트가 가능하다.
첫 요청이 들어오면 max_wait_ms 동안(또는 max_batch_size에 도달할 때까지)
뒤따르는 요청을 모아 한 배치로 어댑터에 넘긴다. 어댑터 추론은 블로킹이므로
ThreadPoolExecutor에서 실행해 이벤트 루프를 막지 않는다.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Tuple

from adapters.base import ModelAdapter


class Scheduler:
    def __init__(
        self,
        adapter: ModelAdapter,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
        workers: int = 1,
    ):
        self.adapter = adapter
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self._executor = ThreadPoolExecutor(max_workers=workers)
        self._queue: "asyncio.Queue" = asyncio.Queue()
        self._task = None
        self._running = False
        # 메트릭
        self.total_requests = 0
        self.total_batches = 0
        self.total_batched_items = 0
        self.max_observed_batch = 0

    async def start(self) -> None:
        self.adapter.load()
        self.adapter.warmup()
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.adapter.unload()
        self._executor.shutdown(wait=False)

    async def submit(self, x: Any) -> Tuple[Any, int]:
        """요청 1건을 큐에 넣고 (출력, 처리된 배치 크기)를 반환."""
        loop = asyncio.get_event_loop()
        fut: "asyncio.Future" = loop.create_future()
        await self._queue.put((x, fut))
        return await fut

    async def _loop(self) -> None:
        loop = asyncio.get_event_loop()
        while self._running:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                break
            batch = [first]
            deadline = time.perf_counter() + self.max_wait_ms / 1000.0
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(nxt)
                except asyncio.TimeoutError:
                    break

            inputs = [b[0] for b in batch]
            bsz = len(batch)
            # 메트릭 갱신
            self.total_requests += bsz
            self.total_batches += 1
            self.total_batched_items += bsz
            if bsz > self.max_observed_batch:
                self.max_observed_batch = bsz

            try:
                outputs = await loop.run_in_executor(
                    self._executor, self.adapter.infer, inputs
                )
            except Exception as e:  # 배치 전체 실패 → 각 요청에 예외 전파
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
                continue

            for (_, fut), out in zip(batch, outputs):
                if not fut.done():  # 클라이언트가 취소(disconnect)했으면 이미 done
                    fut.set_result((out, bsz))

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def avg_batch_size(self) -> float:
        return (self.total_batched_items / self.total_batches) if self.total_batches else 0.0
