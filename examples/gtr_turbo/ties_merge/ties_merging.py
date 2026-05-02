"""TIES merging algorithm for combining RL checkpoint task vectors.

Reference: Yadav et al., "Resolving Interference When Merging Models" (NeurIPS 2023)
Used in GTR-Turbo (Wei et al., arXiv:2512.13043) for creating merged teachers
from RL training checkpoints.
"""

from __future__ import annotations

import logging
import json
import shutil
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def _trim_task_vector(task_vector: Tensor, density: float) -> Tensor:
    """Trim a task vector by zeroing out values below the density percentile by magnitude."""
    if density >= 1.0:
        return task_vector
    flat = task_vector.flatten().abs()
    k = max(1, int(flat.numel() * density))
    threshold = flat.kthvalue(flat.numel() - k + 1).values
    mask = flat.view_as(task_vector).abs() >= threshold
    return task_vector * mask


def _elect_sign(task_vectors: list[Tensor], weights: list[float]) -> Tensor:
    """Elect the sign for each parameter position via weighted majority vote."""
    weighted_sum = torch.zeros_like(task_vectors[0])
    for tv, w in zip(task_vectors, weights):
        weighted_sum += w * tv
    return torch.sign(weighted_sum)


def _selective_average(
    task_vectors: list[Tensor],
    weights: list[float],
    elected_sign: Tensor,
) -> Tensor:
    """Average only values whose sign matches the elected sign."""
    merged = torch.zeros_like(task_vectors[0])
    weight_sum = torch.zeros_like(task_vectors[0])
    for tv, w in zip(task_vectors, weights):
        sign_match = (torch.sign(tv) == elected_sign) & (tv != 0)
        merged += w * tv * sign_match
        weight_sum += w * sign_match
    weight_sum = weight_sum.clamp(min=1e-8)
    return merged / weight_sum


def ties_merge_state_dicts(
    base_state_dict: dict[str, Any],
    checkpoint_state_dicts: list[dict[str, Any]],
    weights: list[float],
    density: float = 0.8,
) -> dict[str, Any]:
    """Merge multiple checkpoint state_dicts using TIES algorithm.

    Args:
        base_state_dict: The base (pre-RL) model state_dict.
        checkpoint_state_dicts: List of RL checkpoint state_dicts.
        weights: Per-checkpoint weights (should sum to ~1.0 for balanced merging).
        density: Fraction of parameters to keep during trimming (0.0-1.0).

    Returns:
        Merged state_dict in the same format as the base.
    """
    assert len(checkpoint_state_dicts) == len(weights), "Number of checkpoints must match number of weights"
    assert len(checkpoint_state_dicts) >= 1, "Need at least one checkpoint to merge"

    merged_state_dict = {}
    keys = list(base_state_dict.keys())

    for key in keys:
        base_param = base_state_dict[key]

        if not isinstance(base_param, Tensor) or not base_param.is_floating_point():
            merged_state_dict[key] = base_param
            continue

        task_vectors = []
        for ckpt_sd in checkpoint_state_dicts:
            if key not in ckpt_sd:
                task_vectors.append(torch.zeros_like(base_param))
                continue
            task_vectors.append(ckpt_sd[key].to(base_param.dtype) - base_param)

        trimmed = [_trim_task_vector(tv, density) for tv in task_vectors]
        elected_sign = _elect_sign(trimmed, weights)
        merged_delta = _selective_average(trimmed, weights, elected_sign)
        merged_state_dict[key] = base_param + merged_delta

    logger.info("TIES merge completed: %d keys processed, density=%.2f", len(keys), density)
    return merged_state_dict


def ties_merge_from_paths(
    base_model_path: str,
    checkpoint_paths: list[str],
    weights: list[float],
    density: float = 0.8,
    device: str = "cpu",
) -> dict[str, Any]:
    """Convenience function: load state_dicts from paths and perform TIES merge.

    Uses safetensors if available, falls back to torch.load.
    """
    base_sd = _load_state_dict(base_model_path, device=device)

    ckpt_sds = []
    for path in checkpoint_paths:
        logger.info("Loading checkpoint: %s", path)
        ckpt_sds.append(_load_state_dict(path, device=device))

    return ties_merge_state_dicts(base_sd, ckpt_sds, weights, density)


def _load_state_dict(model_path: str, device: str = "cpu") -> dict[str, Any]:
    """Load a HuggingFace model state_dict from a directory or file."""
    from pathlib import Path

    path = Path(model_path)

    if path.is_dir():
        safetensor_files = sorted(path.glob("*.safetensors"))
        if safetensor_files:
            try:
                from safetensors.torch import load_file

                state_dict = {}
                for sf in safetensor_files:
                    state_dict.update(load_file(str(sf), device=device))
                return state_dict
            except ImportError:
                pass

        bin_files = sorted(path.glob("*.bin"))
        if bin_files:
            state_dict = {}
            for bf in bin_files:
                state_dict.update(torch.load(str(bf), map_location=device, weights_only=True))
            return state_dict

        raise FileNotFoundError(f"No model files found in {model_path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device=device)

    return torch.load(str(path), map_location=device, weights_only=True)


def save_state_dict(
    state_dict: dict[str, Any],
    output_path: str,
    max_shard_size: str = "5GB",
) -> None:
    """Save a state_dict using a HuggingFace-compatible sharded layout."""
    from pathlib import Path

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import split_torch_state_dict_into_shards
        from safetensors.torch import save_file

        tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, Tensor)}
        split = split_torch_state_dict_into_shards(tensor_dict, max_shard_size=max_shard_size)

        metadata = {"format": "pt"}
        for filename, tensor_names in split.filename_to_tensors.items():
            shard = {name: tensor_dict[name].contiguous() for name in tensor_names}
            save_file(shard, str(path / filename), metadata=metadata)

        if split.is_sharded:
            index = {
                "metadata": split.metadata,
                "weight_map": split.tensor_to_filename,
            }
            with open(path / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, sort_keys=True)
    except ImportError:
        torch.save(state_dict, str(path / "pytorch_model.bin"))

    logger.info("Merged model saved to %s", output_path)


def copy_hf_assets(base_model_path: str, output_path: str) -> None:
    """Copy tokenizer, processor, config, and other non-weight HF assets."""
    from pathlib import Path

    src = Path(base_model_path)
    dst = Path(output_path)
    dst.mkdir(parents=True, exist_ok=True)

    weight_suffixes = {
        ".bin",
        ".safetensors",
    }
    weight_index_names = {
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    }

    for item in src.iterdir():
        if item.name in weight_index_names or item.suffix in weight_suffixes:
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)
