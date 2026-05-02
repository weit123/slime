"""ALFWorld Ray remote worker.

Wraps a single AlfEnv instance as a Ray-compatible actor.
The @ray.remote decorator is applied in env_pool.py to allow
dynamic resource configuration.
"""

from __future__ import annotations

import logging
import os
import subprocess
import shutil
import fcntl
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_xvfb(display=":99"):
    """Start Xvfb if not already running."""
    lock_path = f"/tmp/gtr_slime_xvfb_{display.replace(':', '')}.lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            subprocess.check_output(["xdpyinfo", "-display", display],
                                    stderr=subprocess.DEVNULL, timeout=2)
        except Exception:
            if shutil.which("Xvfb") is None:
                raise RuntimeError("Xvfb not found")
            env = os.environ.copy()
            env["LIBGL_ALWAYS_SOFTWARE"] = "1"
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = ""
            subprocess.Popen(["Xvfb", display, "-screen", "0", "1024x768x24", "-ac"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
            import time
            time.sleep(2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    os.environ["DISPLAY"] = display
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"


class AlfWorldWorker:
    """Ray actor wrapping a single ALFWorld environment instance.

    Args:
        worker_id: Unique identifier for this worker.
        config_file: Path to ALFWorld YAML configuration file.
        max_steps: Maximum steps per episode.
        image_size: Optional (width, height) for observation images.
    """

    def __init__(
        self,
        worker_id: int,
        config_file: str = "base_config.yaml",
        max_steps: int = 50,
        image_size: tuple[int, int] | None = None,
        legacy_build_path: str | None = None,
        xvfb_display_base: int | None = None,
    ):
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.image_size = image_size

        display_base = int(xvfb_display_base or os.environ.get("GTR_SLIME_XVFB_DISPLAY_BASE", "99"))
        os.environ.pop("DISPLAY", None)
        _ensure_xvfb(f":{display_base + worker_id}")

        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

        from examples.gtr_turbo.alfworld.alf_utils import (
            AlfEnv,
            force_legacy_thor_build,
            install_thor5_compat_patches,
            load_config_file,
        )

        force_legacy_thor_build(legacy_build_path)
        install_thor5_compat_patches()
        config = load_config_file(config_file)
        env = AlfredThorEnv(config, train_eval="eval_in_distribution")
        env.init_env(batch_size=1)
        self.alf_env = AlfEnv(env, max_steps=max_steps, image_size=image_size)
        # Warm up THOR rendering — first scene load after init produces
        # different images. Do a dummy reset to stabilize the renderer.
        try:
            self.alf_env.env.reset()
        except Exception:
            logger.warning("AlfWorldWorker %d warm-up failed (non-fatal)", worker_id)
        logger.info("AlfWorldWorker %d initialized (ThorEnv)", worker_id)

    def reset(self, task_file=None) -> dict[str, Any]:
        """Reset environment and return initial observation."""
        return self.alf_env.reset(task_file=task_file)

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute action and return (observation, reward, done, info)."""
        return self.alf_env.step(action)

    def get_task_description(self) -> str:
        return self.alf_env.task_description

    def get_admissible_commands(self) -> list[str]:
        return self.alf_env.admissible_commands

    def close(self):
        self.alf_env.close()
        logger.info("AlfWorldWorker %d closed", self.worker_id)
