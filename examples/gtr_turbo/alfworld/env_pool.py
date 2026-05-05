"""Async pool of AlfWorldWorker Ray actors.

Singleton pool managing multiple ALFWorld Ray actors for parallel
environment execution. Workers are created once (expensive THOR
initialization) and reused across training iterations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import ray

from examples.gtr_turbo.alfworld.env_worker import AlfWorldWorker

logger = logging.getLogger(__name__)


class AlfWorldEnvPool:
    """Singleton async pool of AlfWorldWorker Ray actors."""

    _instance: AlfWorldEnvPool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: dict[str, Any]) -> AlfWorldEnvPool:
        """Return the singleton pool, creating it on first call."""
        if cls._instance is not None:
            return cls._instance
        async with cls._lock:
            if cls._instance is not None:
                return cls._instance
            pool = cls()
            await pool._initialize(config)
            cls._instance = pool
            return cls._instance

    async def _initialize(self, config: dict[str, Any]) -> None:
        """Create all Ray actor workers from config."""
        num_workers = config.get("num_workers", 16)
        config_file = config.get("alfworld_config_file", "base_config.yaml")
        max_steps = config.get("max_turns", 50)
        image_size = config.get("image_size", None)
        if isinstance(image_size, list):
            image_size = tuple(image_size)
        legacy_build_path = config.get("legacy_build_path")
        xvfb_display_base = config.get("xvfb_display_base")
        use_gpu_xorg = bool(config.get("use_gpu_xorg", False))
        xorg_display_base = config.get("xorg_display_base")
        xorg_num_displays = config.get("xorg_num_displays")
        self._prewarm_workers = bool(config.get("prewarm_workers", True))
        render_image = config.get("render_image")
        render_depth_image = config.get("render_depth_image")
        render_class_image = config.get("render_class_image")
        render_object_image = config.get("render_object_image")
        worker_warmup_reset = bool(config.get("worker_warmup_reset", False))

        resources = config.get("resources_per_worker", {"num_cpus": 2, "num_gpus": 0.25})

        logger.info("Initializing AlfWorldEnvPool with %d workers...", num_workers)

        self._config_file = config_file
        self._max_steps = max_steps
        self._image_size = image_size
        self._legacy_build_path = legacy_build_path
        self._xvfb_display_base = xvfb_display_base
        self._use_gpu_xorg = use_gpu_xorg
        self._xorg_display_base = xorg_display_base
        self._xorg_num_displays = xorg_num_displays
        self._render_image = render_image
        self._render_depth_image = render_depth_image
        self._render_class_image = render_class_image
        self._render_object_image = render_object_image
        self._worker_warmup_reset = worker_warmup_reset
        self._RemoteWorker = ray.remote(**resources)(AlfWorldWorker)

        self._workers: list[ray.ObjectRef] = []
        for i in range(num_workers):
            self._workers.append(self._create_worker(i))

        self._available: asyncio.PriorityQueue[tuple[float, int, int]] = asyncio.PriorityQueue()
        self._worker_scores = [0.0 for _ in range(num_workers)]
        self._worker_reset_ema = [0.0 for _ in range(num_workers)]
        self._worker_step_ema = [0.0 for _ in range(num_workers)]
        self._worker_use_counts = [0 for _ in range(num_workers)]
        self._queue_seq = 0
        self._closed = False
        for i in range(num_workers):
            self._put_available(i)

        self._num_workers = num_workers
        if self._prewarm_workers:
            start = time.monotonic()
            logger.info("Prewarming %d AlfWorld workers...", num_workers)
            ready_refs = [worker.ready.remote() for worker in self._workers]
            ready = await asyncio.to_thread(ray.get, ready_refs)
            logger.info("Prewarmed AlfWorld workers in %.1fs: %s",
                        time.monotonic() - start, ready)
        logger.info("AlfWorldEnvPool initialized with %d workers", num_workers)

    def _create_worker(self, worker_id: int):
        return self._RemoteWorker.options(scheduling_strategy="SPREAD").remote(
            worker_id=worker_id,
            config_file=self._config_file,
            max_steps=self._max_steps,
            image_size=self._image_size,
            legacy_build_path=self._legacy_build_path,
            xvfb_display_base=self._xvfb_display_base,
            use_gpu_xorg=self._use_gpu_xorg,
            xorg_display_base=self._xorg_display_base,
            xorg_num_displays=self._xorg_num_displays,
            render_image=self._render_image,
            render_depth_image=self._render_depth_image,
            render_class_image=self._render_class_image,
            render_object_image=self._render_object_image,
            worker_warmup_reset=self._worker_warmup_reset,
        )

    def _put_available(self, worker_id: int) -> None:
        self._queue_seq += 1
        self._available.put_nowait((self._worker_scores[worker_id], self._queue_seq, worker_id))

    async def acquire(self) -> tuple[ray.ObjectRef, int]:
        """Acquire an available worker. Blocks (async) if none available."""
        _, _, worker_id = await self._available.get()
        return self._workers[worker_id], worker_id

    def record_timing(self, worker_id: int, *, reset_sec: float | None = None, step_sec: float | None = None) -> None:
        """Update simple per-worker EMA used to prefer faster warm workers."""
        alpha = 0.2
        if reset_sec is not None:
            old = self._worker_reset_ema[worker_id]
            self._worker_reset_ema[worker_id] = reset_sec if old <= 0 else (1 - alpha) * old + alpha * reset_sec
        if step_sec is not None:
            old = self._worker_step_ema[worker_id]
            self._worker_step_ema[worker_id] = step_sec if old <= 0 else (1 - alpha) * old + alpha * step_sec
        self._worker_scores[worker_id] = self._worker_reset_ema[worker_id] + 5.0 * self._worker_step_ema[worker_id]
        self._worker_use_counts[worker_id] += 1

    def release(self, worker_id: int) -> None:
        """Release a worker back to the pool."""
        if self._closed:
            return
        self._put_available(worker_id)

    async def restart(self, worker_id: int):
        """Replace a broken worker actor and return the new actor handle."""
        old_worker = self._workers[worker_id]
        try:
            ray.kill(old_worker, no_restart=True)
        except Exception:
            logger.warning("Failed to kill ALFWorld worker %d during restart", worker_id, exc_info=True)
        worker = self._create_worker(worker_id)
        self._workers[worker_id] = worker
        if self._prewarm_workers:
            await asyncio.to_thread(ray.get, worker.ready.remote())
        self._worker_scores[worker_id] = 0.0
        self._worker_reset_ema[worker_id] = 0.0
        self._worker_step_ema[worker_id] = 0.0
        logger.info("Restarted ALFWorld worker %d", worker_id)
        return worker

    def discard(self, worker_id: int) -> None:
        """Remove a failed worker from circulation without blocking for replacement."""
        if self._closed:
            return
        try:
            ray.kill(self._workers[worker_id], no_restart=True)
        except Exception:
            logger.warning("Failed to kill discarded ALFWorld worker %d", worker_id, exc_info=True)
        self._workers[worker_id] = self._create_worker(worker_id)
        self._worker_scores[worker_id] = 0.0
        self._worker_reset_ema[worker_id] = 0.0
        self._worker_step_ema[worker_id] = 0.0
        self._put_available(worker_id)
        logger.info("Discarded and replaced ALFWorld worker %d", worker_id)

    async def close(self) -> None:
        """Shutdown all workers."""
        logger.info("Closing AlfWorldEnvPool...")
        self._closed = True
        close_refs = [w.close.remote() for w in self._workers]
        await asyncio.to_thread(ray.get, close_refs)
        for w in self._workers:
            ray.kill(w)
        self._workers.clear()
        AlfWorldEnvPool._instance = None
        logger.info("AlfWorldEnvPool closed")
