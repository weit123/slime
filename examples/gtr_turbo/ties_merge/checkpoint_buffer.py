"""Checkpoint buffer for GTR-Turbo training.

Manages RL checkpoints and computes merging weights using
Simple Moving Average (SMA) or Exponential Moving Average (EMA).
"""

from __future__ import annotations

import logging
from pathlib import Path

from examples.gtr_turbo.ties_merge.ties_merging import (
    copy_hf_assets,
    save_state_dict,
    ties_merge_from_paths,
)

logger = logging.getLogger(__name__)


class CheckpointBuffer:
    """Buffer that accumulates RL checkpoints and produces merged teachers via TIES.

    Args:
        base_model_path: Path to the base (pre-RL) HuggingFace model.
        weighting: Weight strategy, "sma" (equal) or "ema" (recent-biased).
        ema_alpha: EMA decay factor. Higher values weight recent checkpoints more.
    """

    def __init__(
        self,
        base_model_path: str,
        weighting: str = "ema",
        ema_alpha: float = 0.5,
    ):
        assert weighting in ("sma", "ema"), f"Unknown weighting: {weighting}"
        self.base_model_path = base_model_path
        self.weighting = weighting
        self.ema_alpha = ema_alpha
        self.checkpoint_paths: list[str] = []

    def add_checkpoint(self, path: str) -> None:
        """Register a new HF-format checkpoint."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.checkpoint_paths.append(str(p))
        logger.info(
            "Checkpoint added (%d total): %s",
            len(self.checkpoint_paths),
            path,
        )

    def compute_weights(self) -> list[float]:
        """Compute per-checkpoint weights based on the weighting strategy.

        SMA: All checkpoints weighted equally: w_i = 1/N
        EMA: Recent checkpoints weighted more: w_i = alpha * (1-alpha)^(N-1-i),
             normalized so weights sum to 1.
        """
        n = len(self.checkpoint_paths)
        if n == 0:
            return []

        if self.weighting == "sma":
            return [1.0 / n] * n

        raw = [self.ema_alpha * ((1 - self.ema_alpha) ** (n - 1 - i)) for i in range(n)]
        total = sum(raw)
        return [w / total for w in raw]

    def merge(self, density: float = 0.8, device: str = "cpu") -> dict:
        """Perform TIES merge on all buffered checkpoints.

        Returns:
            Merged state_dict.
        """
        if len(self.checkpoint_paths) < 1:
            raise ValueError("Need at least 1 checkpoint to merge")

        weights = self.compute_weights()
        logger.info(
            "Merging %d checkpoints (weighting=%s, density=%.2f)",
            len(self.checkpoint_paths),
            self.weighting,
            density,
        )
        for i, (p, w) in enumerate(zip(self.checkpoint_paths, weights)):
            logger.info("  [%d] weight=%.4f path=%s", i, w, p)

        return ties_merge_from_paths(
            self.base_model_path,
            self.checkpoint_paths,
            weights,
            density=density,
            device=device,
        )

    def save_merged(self, output_path: str, density: float = 0.8, device: str = "cpu") -> str:
        """Merge and save the result to disk.

        Returns:
            The output path.
        """
        merged_sd = self.merge(density=density, device=device)
        save_state_dict(merged_sd, output_path)
        copy_hf_assets(self.base_model_path, output_path)

        return output_path

    def __len__(self) -> int:
        return len(self.checkpoint_paths)
