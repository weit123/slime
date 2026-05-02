"""Async pool of AlfWorldWorker Ray actors.

Singleton pool managing multiple ALFWorld Ray actors for parallel
environment execution. Workers are created once (expensive THOR
initialization) and reused across training iterations.
"""

from __future__ import annotations

import asyncio
import logging
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

        resources = config.get("resources_per_worker", {"num_cpus": 2, "num_gpus": 0.25})

        logger.info("Initializing AlfWorldEnvPool with %d workers...", num_workers)

        self._config_file = config_file
        self._max_steps = max_steps
        self._image_size = image_size
        self._legacy_build_path = legacy_build_path
        self._xvfb_display_base = xvfb_display_base
        self._RemoteWorker = ray.remote(**resources)(AlfWorldWorker)

        self._workers: list[ray.ObjectRef] = []
        for i in range(num_workers):
            self._workers.append(self._create_worker(i))

        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(num_workers):
            self._available.put_nowait(i)

        self._num_workers = num_workers
        logger.info("AlfWorldEnvPool initialized with %d workers", num_workers)

    def _create_worker(self, worker_id: int):
        return self._RemoteWorker.options(scheduling_strategy="SPREAD").remote(
            worker_id=worker_id,
            config_file=self._config_file,
            max_steps=self._max_steps,
            image_size=self._image_size,
            legacy_build_path=self._legacy_build_path,
            xvfb_display_base=self._xvfb_display_base,
        )

    async def acquire(self) -> tuple[ray.ObjectRef, int]:
        """Acquire an available worker. Blocks (async) if none available."""
        worker_id = await self._available.get()
        return self._workers[worker_id], worker_id

    def release(self, worker_id: int) -> None:
        """Release a worker back to the pool."""
        self._available.put_nowait(worker_id)

    async def restart(self, worker_id: int):
        """Replace a broken worker actor and return the new actor handle."""
        old_worker = self._workers[worker_id]
        try:
            ray.kill(old_worker, no_restart=True)
        except Exception:
            logger.warning("Failed to kill ALFWorld worker %d during restart", worker_id, exc_info=True)
        worker = self._create_worker(worker_id)
        self._workers[worker_id] = worker
        logger.info("Restarted ALFWorld worker %d", worker_id)
        return worker

    async def close(self) -> None:
        """Shutdown all workers."""
        logger.info("Closing AlfWorldEnvPool...")
        close_refs = [w.close.remote() for w in self._workers]
        await asyncio.to_thread(ray.get, close_refs)
        for w in self._workers:
            ray.kill(w)
        self._workers.clear()
        AlfWorldEnvPool._instance = None
        logger.info("AlfWorldEnvPool closed")
