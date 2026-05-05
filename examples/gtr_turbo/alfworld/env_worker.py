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
import time
from typing import Any

logger = logging.getLogger(__name__)


def _patch_flask_jinja2_compat():
    """Expose Jinja2 APIs expected by Flask 1.x when running with Jinja2 3.x."""
    import jinja2
    from markupsafe import Markup, escape

    if not hasattr(jinja2, "escape"):
        jinja2.escape = escape
    if not hasattr(jinja2, "Markup"):
        jinja2.Markup = Markup


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


def _use_existing_xorg(display: str):
    """Use an existing GPU-backed X server without forcing Mesa software GL."""
    last_exc: Exception | None = None
    for _ in range(5):
        try:
            subprocess.check_output(["xdpyinfo", "-display", display],
                                    stderr=subprocess.DEVNULL, timeout=10)
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(1)
    else:
        raise RuntimeError(f"Xorg display {display} is not available") from last_exc
    os.environ["DISPLAY"] = display
    os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
    os.environ.pop("__EGL_VENDOR_LIBRARY_FILENAMES", None)


def _configure_render_flags(
    *,
    render_image: bool | None = None,
    render_depth_image: bool | None = None,
    render_class_image: bool | None = None,
    render_object_image: bool | None = None,
) -> None:
    """Patch ALFWorld THOR render flags before scene reset/restore calls."""
    from alfworld.gen import constants

    if render_image is not None:
        constants.RENDER_IMAGE = bool(render_image)
    if render_depth_image is not None:
        constants.RENDER_DEPTH_IMAGE = bool(render_depth_image)
    if render_class_image is not None:
        constants.RENDER_CLASS_IMAGE = bool(render_class_image)
    if render_object_image is not None:
        constants.RENDER_OBJECT_IMAGE = bool(render_object_image)


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
        use_gpu_xorg: bool = False,
        xorg_display_base: int | None = None,
        xorg_num_displays: int | None = None,
        render_image: bool | None = None,
        render_depth_image: bool | None = None,
        render_class_image: bool | None = None,
        render_object_image: bool | None = None,
        worker_warmup_reset: bool = False,
    ):
        self.worker_id = worker_id
        self.max_steps = max_steps
        self.image_size = image_size
        self._broken = False
        self._last_error: str | None = None

        os.environ.pop("DISPLAY", None)
        if use_gpu_xorg:
            display_base = int(xorg_display_base or os.environ.get("GTR_SLIME_XORG_DISPLAY_BASE", "210"))
            num_displays = int(xorg_num_displays or os.environ.get("GTR_SLIME_XORG_NUM_DISPLAYS", "8"))
            display = f":{display_base + (worker_id % num_displays)}"
            _use_existing_xorg(display)
        else:
            display_base = int(xvfb_display_base or os.environ.get("GTR_SLIME_XVFB_DISPLAY_BASE", "99"))
            _ensure_xvfb(f":{display_base + worker_id}")
        _patch_flask_jinja2_compat()

        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv

        from examples.gtr_turbo.alfworld.alf_utils import (
            AlfEnv,
            force_legacy_thor_build,
            install_thor5_compat_patches,
            load_config_file,
        )

        force_legacy_thor_build(legacy_build_path)
        install_thor5_compat_patches(patch_put_object=not bool(legacy_build_path))
        _configure_render_flags(
            render_image=render_image,
            render_depth_image=render_depth_image,
            render_class_image=render_class_image,
            render_object_image=render_object_image,
        )
        config = load_config_file(config_file)
        env = AlfredThorEnv(config, train_eval="eval_in_distribution")
        env.init_env(batch_size=1)
        self.alf_env = AlfEnv(env, max_steps=max_steps, image_size=image_size)
        if worker_warmup_reset:
            # Optional compatibility warm-up for jobs that need a dummy scene
            # load before the first task-specific reset.
            try:
                self.alf_env.env.reset()
            except Exception:
                logger.warning("AlfWorldWorker %d warm-up failed (non-fatal)", worker_id)
        logger.info("AlfWorldWorker %d initialized (ThorEnv)", worker_id)

    def reset(self, task_file=None) -> dict[str, Any]:
        """Reset environment and return initial observation."""
        if self._broken:
            raise RuntimeError(f"AlfWorldWorker {self.worker_id} is broken: {self._last_error}")
        try:
            return self.alf_env.reset(task_file=task_file)
        except Exception as exc:
            self._mark_broken(exc)
            raise

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        """Execute action and return (observation, reward, done, info)."""
        if self._broken:
            raise RuntimeError(f"AlfWorldWorker {self.worker_id} is broken: {self._last_error}")
        try:
            return self.alf_env.step(action)
        except Exception as exc:
            self._mark_broken(exc)
            raise

    def _mark_broken(self, exc: BaseException) -> None:
        self._broken = True
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "AlfWorldWorker %d marked broken after %s",
            self.worker_id,
            self._last_error,
        )
        try:
            self.alf_env.close()
        except Exception:
            logger.warning("Failed to close broken AlfWorldWorker %d", self.worker_id, exc_info=True)

    def ready(self) -> dict[str, Any]:
        """Return lightweight worker status after actor construction."""
        return {
            "worker_id": self.worker_id,
            "display": os.environ.get("DISPLAY"),
            "image_size": self.image_size,
            "broken": self._broken,
            "last_error": self._last_error,
        }

    def get_task_description(self) -> str:
        return self.alf_env.task_description

    def get_admissible_commands(self) -> list[str]:
        return self.alf_env.admissible_commands

    def close(self):
        self.alf_env.close()
        logger.info("AlfWorldWorker %d closed", self.worker_id)
