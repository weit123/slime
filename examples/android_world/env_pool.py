"""Async pool of persistent AndroidWorldWorker Ray actors for slime.

The pool is a singleton: all concurrent generate() calls share the same set
of emulator workers. Workers are created once (expensive) and reused across
training iterations. The acquire/release API is async-safe via asyncio.Queue.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ray

from examples.android_world.env_worker import AndroidWorldWorker, generate_env_configs

logger = logging.getLogger(__name__)


class AndroidWorldEnvPool:
    """Singleton async pool of persistent AndroidWorldWorker Ray actors."""

    _instance: AndroidWorldEnvPool | None = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: dict[str, Any]) -> AndroidWorldEnvPool:
        """Return the singleton pool, creating it on first call."""
        # Fast path (no lock)
        if cls._instance is not None:
            return cls._instance
        async with cls._lock:
            # Double-checked locking
            if cls._instance is not None:
                return cls._instance
            pool = cls()
            await pool._initialize(config)
            cls._instance = pool
            return pool

    async def _initialize(self, config: dict[str, Any]) -> None:
        """Create all Ray actor workers from config."""
        num_workers: int = config.get("num_workers", 16)
        avd_name: str = config.get("avd_name", "AndroidWorldAvd")
        base_avd_name_pattern: str = config.get("base_avd_name_pattern", "slime_aw_{}")
        base_console_port: int = config.get("base_console_port", 5556)
        base_grpc_port: int = config.get("base_grpc_port", 8554)
        base_adb_server_port: int = config.get("base_adb_server_port", 5037)
        android_avd_home: str = config.get("android_avd_home", "/root/android/avd/")
        android_sdk_root: str = config.get("android_sdk_root", "/root/android/")
        emulator_path: str = config.get("emulator_path", "/root/android/emulator/emulator")
        adb_path: str = config.get("adb_path", "/root/android/platform-tools/adb")
        max_steps: int = config.get("max_turns", 15)
        temp_path: str = config.get("temp_path", "/tmp/android_world_images")
        task_family: str = config.get("task_family", "android_world")
        save_images: bool = config.get("save_images", False)
        image_size = config.get("image_size", None)
        resources_per_worker: dict = config.get("resources_per_worker", {"num_cpus": 4, "memory": 8 * 1024 * 1024 * 1024})

        logger.info("Initializing AndroidWorldEnvPool with %d workers...", num_workers)

        # Generate port/AVD configurations
        cache_avd_names, console_ports, grpc_ports, adb_server_ports = generate_env_configs(
            base_avd_name_pattern=base_avd_name_pattern,
            base_console_port=base_console_port,
            base_grpc_port=base_grpc_port,
            num_envs=num_workers,
            base_adb_server_port=base_adb_server_port,
        )

        # Create Ray remote actor class
        RemoteWorker = ray.remote(**resources_per_worker)(AndroidWorldWorker)

        # Launch all workers with SPREAD scheduling
        self._workers: list[ray.ObjectRef] = []
        for i in range(num_workers):
            worker_temp_path = f"{temp_path}/{cache_avd_names[i]}"
            worker = RemoteWorker.options(scheduling_strategy="SPREAD").remote(
                worker_id=i,
                console_port=console_ports[i],
                grpc_port=grpc_ports[i],
                adb_path=adb_path,
                adb_server_port=adb_server_ports[i],
                emulator_path=emulator_path,
                android_sdk_root=android_sdk_root,
                android_avd_home=android_avd_home,
                avd_name=cache_avd_names[i],
                max_steps=max_steps,
                temp_path=worker_temp_path,
                task_family=task_family,
                save_images=save_images,
                image_size=image_size,
                base_avd_name=avd_name,
            )
            self._workers.append(worker)

        # Ray actor __init__ runs when the first task is dispatched on the actor.
        # The first acquire->reset call will block until that worker's __init__ completes.
        logger.info("Waiting for %d emulator workers to initialize...", num_workers)

        # Set up the available queue
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(num_workers):
            self._available.put_nowait(i)

        self._num_workers = num_workers
        logger.info("AndroidWorldEnvPool initialized with %d workers", num_workers)

    async def acquire(self) -> tuple[Any, int]:
        """Acquire an available worker. Blocks (async) if none available."""
        worker_id = await self._available.get()
        return self._workers[worker_id], worker_id

    def release(self, worker_id: int) -> None:
        """Release a worker back to the pool."""
        self._available.put_nowait(worker_id)

    async def close(self) -> None:
        """Shutdown all workers and kill emulators."""
        logger.info("Closing AndroidWorldEnvPool...")
        close_refs = [w.close.remote() for w in self._workers]
        await asyncio.to_thread(ray.get, close_refs)
        for w in self._workers:
            ray.kill(w)
        self._workers.clear()
        AndroidWorldEnvPool._instance = None
        logger.info("AndroidWorldEnvPool closed")
