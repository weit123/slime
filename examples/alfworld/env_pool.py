"""Async pool of AlfWorldWorker Ray actors.

This module provides AlfWorldEnvPool, a singleton async pool that manages
multiple AlfWorldWorker Ray actors for parallel environment execution.

Reference: examples/android_world/env_pool.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from examples.alfworld.env_worker import AlfWorldWorker

logger = logging.getLogger(__name__)


class AlfWorldEnvPool:
    """Singleton async pool of AlfWorldWorker Ray actors.

    The pool is a singleton: all concurrent generate() calls share the same set
    of workers. Workers are created once (expensive THOR initialization) and
    reused across training iterations. The acquire/release API is async-safe
    via asyncio.Queue.
    """

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
        resources_per_worker = config.get("resources_per_worker", {"num_cpus": 2, "num_gpus": 0.25})

        logger.info("Initializing AlfWorldEnvPool with %d workers...", num_workers)

        # Create Ray remote actor class with resources
        RemoteWorker = ray.remote(**resources_per_worker)(AlfWorldWorker)

        # Launch all workers with SPREAD scheduling
        self._workers: list[ray.ObjectRef] = []
        for i in range(num_workers):
            worker = RemoteWorker.options(scheduling_strategy="SPREAD").remote(
                worker_id=i,
                config_file=config_file,
                max_steps=max_steps,
                image_size=image_size,
            )
            self._workers.append(worker)

        logger.info("Waiting for %d ALFWorld workers to initialize...", num_workers)

        # Set up the available queue
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(num_workers):
            self._available.put_nowait(i)

        self._num_workers = num_workers
        logger.info("AlfWorldEnvPool initialized with %d workers", num_workers)

    async def acquire(self) -> tuple[ray.ObjectRef, int]:
        """Acquire an available worker. Blocks (async) if none available."""
        worker_id = await self._available.get()
        return self._workers[worker_id], worker_id

    def release(self, worker_id: int) -> None:
        """Release a worker back to the pool."""
        self._available.put_nowait(worker_id)

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
