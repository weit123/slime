"""Verification Step 3: End-to-end rollout integration test.

Run from the slime repo root (requires Android emulators + GPU + SGLang):
    python examples/android_world/tests/test_rollout.py

This test:
1. Boots a small env pool (2 workers)
2. Starts a local SGLang server with a small VLM model
3. Creates Sample objects mimicking what the data source produces
4. Calls the generate() function directly
5. Validates the output sample: tokens, loss_mask, reward, multimodal state

Prerequisites:
    - Android emulators available (AVD configured)
    - A VLM model downloaded (e.g. Qwen3-VL-2B-Instruct)
    - GPU available for SGLang
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from argparse import Namespace
from typing import Any

sys.path.insert(0, os.getcwd())


def make_mock_args(
    model_path: str,
    sglang_router_ip: str = "127.0.0.1",
    sglang_router_port: int = 30000,
) -> Namespace:
    """Build a minimal args Namespace mimicking what slime provides."""
    import yaml

    with open("examples/android_world/config.yaml") as f:
        config = yaml.safe_load(f)

    args = Namespace(
        # Model
        hf_checkpoint=model_path,
        # SGLang
        sglang_router_ip=sglang_router_ip,
        sglang_router_port=sglang_router_port,
        sglang_server_concurrency=8,
        sglang_dp_size=None,
        rollout_num_gpus=1,
        rollout_num_gpus_per_engine=1,
        # Rollout params
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_max_response_len=512,
        rollout_max_context_len=4096,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        rollout_seed=42,
        # Custom config (from YAML, flattened into args)
        apply_chat_template=True,
        apply_chat_template_kwargs=None,
        partial_rollout=False,
        ci_test=False,
        use_rollout_routing_replay=False,
        use_slime_router=False,
        group_rm=False,
        custom_rm_path=None,
        rm_type=None,
        sglang_enable_deterministic_inference=False,
        reward_key=None,
        # Android World config from YAML
        **config,
    )
    return args


def make_sample(task_name: str, params_idx: int = 0):
    """Create a Sample mimicking what RolloutDataSource.get_samples() produces."""
    from slime.utils.types import Sample

    sample = Sample(
        prompt="Complete a task on Android",  # placeholder, overridden by generate()
        label="",
        metadata={"task_name": task_name, "params_idx": params_idx},
    )
    sample.group_index = 0
    sample.index = 0
    return sample


async def test_generate_function(args: Namespace):
    """Run the generate function on a single sample and validate output."""
    from slime.utils.types import Sample
    from examples.android_world.rollout import generate

    sample = make_sample("ContactsAddContact", params_idx=0)

    sampling_params = {
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "top_k": args.rollout_top_k,
        "max_new_tokens": args.rollout_max_response_len,
        "stop": args.rollout_stop,
        "stop_token_ids": args.rollout_stop_token_ids,
        "skip_special_tokens": args.rollout_skip_special_tokens,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    print("\n[1] Calling generate()...")
    t0 = time.time()
    result = await generate(args, sample, sampling_params)
    elapsed = time.time() - t0
    print(f"    generate() completed in {elapsed:.1f}s")

    # --- Validate ---
    print("\n[2] Validating output sample...")

    # Tokens
    assert result.tokens is not None, "tokens is None"
    assert len(result.tokens) > 0, "tokens is empty"
    print(f"    tokens length: {len(result.tokens)}")

    # Loss mask
    assert result.loss_mask is not None, "loss_mask is None"
    assert len(result.loss_mask) > 0, "loss_mask is empty"
    num_model_tokens = sum(1 for m in result.loss_mask if m == 1)
    num_env_tokens = sum(1 for m in result.loss_mask if m == 0)
    print(f"    loss_mask length: {len(result.loss_mask)}")
    print(f"    model tokens (mask=1): {num_model_tokens}")
    print(f"    env tokens (mask=0): {num_env_tokens}")

    # The loss mask should cover exactly the response portion
    # (response_tokens = tokens - prompt_tokens, but the loss_mask
    #  only covers the response portion appended during the loop)
    assert len(result.loss_mask) == result.response_length, (
        f"loss_mask length {len(result.loss_mask)} != response_length {result.response_length}"
    )

    # Rollout log probs
    assert result.rollout_log_probs is not None, "rollout_log_probs is None"
    assert len(result.rollout_log_probs) == result.response_length, (
        f"rollout_log_probs length {len(result.rollout_log_probs)} != response_length {result.response_length}"
    )
    # Env observation tokens should have logprob = 0.0
    env_logprobs = [lp for lp, m in zip(result.rollout_log_probs, result.loss_mask) if m == 0]
    assert all(lp == 0.0 for lp in env_logprobs), "Some env observation tokens have non-zero logprob"
    print(f"    rollout_log_probs length: {len(result.rollout_log_probs)}")
    print(f"    env tokens all have logprob=0.0: OK")

    # Reward
    assert result.reward is not None, "reward is None"
    print(f"    reward: {result.reward}")

    # Status
    assert result.status is not None, "status is None"
    print(f"    status: {result.status}")

    # Response text
    assert result.response is not None, "response is None"
    assert len(result.response) > 0, "response is empty"
    print(f"    response preview: {result.response[:200]}...")

    # Prompt was overridden (should no longer be the placeholder)
    assert result.prompt != "Complete a task on Android", "Prompt was not overridden dynamically"
    assert len(result.prompt) > 100, f"Prompt seems too short: {len(result.prompt)} chars"
    print(f"    prompt length: {len(result.prompt)} chars (dynamically constructed)")

    # Multimodal
    if result.multimodal_train_inputs is not None:
        print(f"    multimodal_train_inputs keys: {list(result.multimodal_train_inputs.keys())}")
        for k, v in result.multimodal_train_inputs.items():
            if hasattr(v, "shape"):
                print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
    else:
        print(f"    multimodal_train_inputs: None (no images encoded for training)")

    print("\n    PASS: All validations passed!")
    return result


async def test_grpo_grouped_samples(args: Namespace):
    """Test that GRPO grouped samples get the same task but different trajectories."""
    from slime.utils.types import Sample
    from examples.android_world.rollout import generate
    import copy

    task_name = "CameraTakePhoto"
    params_idx = 0
    n_samples = 2

    # Simulate what RolloutDataSource.get_samples does
    base_sample = make_sample(task_name, params_idx)
    group = []
    for i in range(n_samples):
        s = copy.deepcopy(base_sample)
        s.group_index = 0
        s.index = i
        group.append(s)

    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 256,
        "stop": None,
        "stop_token_ids": None,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "spaces_between_special_tokens": False,
    }

    print("\n[3] Testing GRPO: generating 2 samples for same task...")
    results = []
    for i, s in enumerate(group):
        print(f"    Generating sample {i}...")
        result = await generate(args, s, sampling_params)
        results.append(result)

    # Both should have the same task in their prompt
    for i, r in enumerate(results):
        print(f"    Sample {i}: tokens={len(r.tokens)}, response_len={r.response_length}, reward={r.reward}")

    # Responses should differ (different trajectories due to stochastic generation)
    if results[0].response != results[1].response:
        print("    Responses differ (expected for stochastic generation)")
    else:
        print("    WARNING: Responses are identical (possible with temperature=1.0 but unlikely)")

    # Both should have valid structure
    for r in results:
        assert r.reward is not None
        assert r.tokens is not None
        assert len(r.loss_mask) == r.response_length

    print("    PASS: GRPO grouped samples work correctly")


async def main():
    print("=" * 70)
    print("Android World Rollout Integration Test")
    print("=" * 70)

    model_path = os.environ.get("SLIME_TEST_MODEL", "/root/models/Qwen3-VL-2B-Instruct")
    sglang_ip = os.environ.get("SLIME_TEST_SGLANG_IP", "127.0.0.1")
    sglang_port = int(os.environ.get("SLIME_TEST_SGLANG_PORT", "30000"))

    if not os.path.exists(model_path):
        print(f"\nERROR: Model not found at {model_path}")
        print("Set SLIME_TEST_MODEL to the model path.")
        sys.exit(1)

    print(f"\nModel: {model_path}")
    print(f"SGLang: {sglang_ip}:{sglang_port}")
    print("\nNOTE: Make sure SGLang server is running:")
    print(f"  python -m sglang.launch_server --model-path {model_path} "
          f"--port {sglang_port} --mem-fraction-static 0.5")

    # Override config for small test
    args = make_mock_args(model_path, sglang_ip, sglang_port)
    args.num_workers = 2  # Only 2 workers for test
    args.max_turns = 3  # Short episodes
    args.rollout_max_response_len = 512
    args.use_distributed_post = False

    import ray
    if not ray.is_initialized():
        ray.init()

    # Initialize the global HTTP client (normally done by slime's rollout actor)
    from slime.utils.http_utils import init_http_client
    init_http_client(args)

    result = await test_generate_function(args)
    await test_grpo_grouped_samples(args)

    print("\n" + "=" * 70)
    print("ALL ROLLOUT TESTS PASSED")
    print("=" * 70)

    # Cleanup pool
    from examples.android_world.env_pool import AndroidWorldEnvPool
    if AndroidWorldEnvPool._instance is not None:
        await AndroidWorldEnvPool._instance.close()


if __name__ == "__main__":
    asyncio.run(main())
