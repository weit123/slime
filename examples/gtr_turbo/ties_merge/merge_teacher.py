"""CLI tool for merging RL checkpoints into a teacher model via TIES.

Usage:
    python -m examples.gtr_turbo.ties_merge.merge_teacher \
        --base-model /path/to/base \
        --checkpoints /path/to/ckpt1 /path/to/ckpt2 ... \
        --output /path/to/merged \
        --weighting ema --ema-alpha 0.5 --density 0.8
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge RL checkpoints via TIES for GTR-Turbo teacher")
    parser.add_argument("--base-model", required=True, help="Path to the base (pre-RL) HF model")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Paths to RL checkpoint HF models")
    parser.add_argument("--output", required=True, help="Output path for the merged teacher model")
    parser.add_argument("--weighting", choices=["sma", "ema"], default="ema", help="Checkpoint weighting strategy")
    parser.add_argument("--ema-alpha", type=float, default=0.5, help="EMA decay factor (only for ema weighting)")
    parser.add_argument("--density", type=float, default=0.8, help="TIES trimming density (fraction of params to keep)")
    parser.add_argument("--device", default="cpu", help="Device for tensor operations")
    return parser.parse_args()


def main():
    args = parse_args()
    from examples.gtr_turbo.ties_merge.checkpoint_buffer import CheckpointBuffer

    buffer = CheckpointBuffer(
        base_model_path=args.base_model,
        weighting=args.weighting,
        ema_alpha=args.ema_alpha,
    )

    for ckpt in args.checkpoints:
        buffer.add_checkpoint(ckpt)

    if len(buffer) < 2:
        logger.error("Need at least 2 checkpoints, got %d", len(buffer))
        sys.exit(1)

    output_path = buffer.save_merged(args.output, density=args.density, device=args.device)
    logger.info("Merged teacher saved to: %s", output_path)


if __name__ == "__main__":
    main()
