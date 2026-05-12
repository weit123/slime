"""Verification Step 1: Import sanity check.

Run from the slime repo root:
    python examples/android_world/tests/test_imports.py

This test verifies that all import paths resolve correctly.
No Android emulators or GPUs are needed.
"""

import os
import sys
import traceback

# Ensure the slime repo root is on sys.path so `examples.*` imports work.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

TESTS = []
def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


@test("slime core types")
def _():
    from slime.utils.types import Sample
    s = Sample()
    assert hasattr(s, "tokens")
    assert hasattr(s, "loss_mask")
    assert hasattr(s, "reward")
    assert hasattr(s, "metadata")
    print(f"  Sample fields: tokens, loss_mask, reward, metadata, status, etc.")


@test("slime rollout GenerateState")
def _():
    from slime.rollout.sglang_rollout import GenerateState
    print(f"  GenerateState imported OK")


@test("slime processing utils")
def _():
    from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
    print(f"  encode_image_for_rollout_engine, load_processor, load_tokenizer imported OK")


@test("slime http utils")
def _():
    from slime.utils.http_utils import post, get
    print(f"  post, get imported OK")


@test("geo3k_vlm_multi_turn base_env")
def _():
    from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
    assert hasattr(BaseInteractionEnv, "reset")
    assert hasattr(BaseInteractionEnv, "step")
    assert hasattr(BaseInteractionEnv, "format_observation")
    assert hasattr(BaseInteractionEnv, "close")
    print(f"  BaseInteractionEnv methods: reset, step, format_observation, close")


@test("geo3k_vlm_multi_turn rollout helpers")
def _():
    from examples.geo3k_vlm_multi_turn.rollout import (
        _append_to_sample,
        _encode_observation_for_generation,
        _finalize_sample,
        _merge_multimodal_train_inputs,
        _run_inference_step,
        _should_stop_on_finish,
        _update_budget,
        _update_multimodal_state,
    )
    print(f"  All 8 helper functions imported OK")


@test("android_world prompts")
def _():
    from examples.android_world.prompts import (
        ANDROID_WORLD_SYSTEM_PROMPT,
        ANDROID_WORLD_TEMPLATE,
        ANDROID_WORLD_TEMPLATE_NO_HIS,
    )
    assert "mobile_use" in ANDROID_WORLD_SYSTEM_PROMPT
    assert "{task_description}" in ANDROID_WORLD_TEMPLATE
    assert "{task_description}" in ANDROID_WORLD_TEMPLATE_NO_HIS
    assert "<image>" in ANDROID_WORLD_TEMPLATE
    print(f"  3 prompt templates OK, mobile_use tool defined, placeholders present")


@test("android_world env_worker (no android_world SDK)")
def _():
    # Test that the module-level helpers that don't depend on android_world SDK parse OK
    import ast
    with open("examples/android_world/env_worker.py") as f:
        tree = ast.parse(f.read())
    # Count classes and functions
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    print(f"  Classes: {classes}")
    print(f"  Top-level functions ({len(functions)}): {[f for f in functions if not f.startswith('_')]}")


@test("android_world env_pool (no Ray)")
def _():
    import ast
    with open("examples/android_world/env_pool.py") as f:
        tree = ast.parse(f.read())
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "AndroidWorldEnvPool" in classes
    print(f"  AndroidWorldEnvPool class found")


@test("android_world env_android_world")
def _():
    import ast
    with open("examples/android_world/env_android_world.py") as f:
        tree = ast.parse(f.read())
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "AndroidWorldEnv" in classes
    print(f"  AndroidWorldEnv class found")


@test("android_world rollout")
def _():
    import ast
    with open("examples/android_world/rollout.py") as f:
        tree = ast.parse(f.read())
    functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert "generate" in functions
    assert "_build_initial_prompt" in functions
    print(f"  Functions: {functions}")


@test("android_world config.yaml")
def _():
    import yaml
    with open("examples/android_world/config.yaml") as f:
        config = yaml.safe_load(f)
    required_keys = ["max_turns", "num_workers", "avd_name", "base_avd_name_pattern",
                     "base_console_port", "base_grpc_port"]
    for k in required_keys:
        assert k in config, f"Missing key: {k}"
    print(f"  Config keys: {list(config.keys())}")
    print(f"  max_turns={config['max_turns']}, num_workers={config['num_workers']}")


@test("android_world tasks.jsonl")
def _():
    import json
    with open("examples/android_world/data/tasks.jsonl") as f:
        lines = f.readlines()
    assert len(lines) > 0
    first = json.loads(lines[0])
    assert "prompt" in first
    assert "metadata" in first
    assert "task_name" in first["metadata"]
    assert "params_idx" in first["metadata"]
    task_names = set()
    for line in lines:
        d = json.loads(line)
        task_names.add(d["metadata"]["task_name"])
    print(f"  {len(lines)} entries, {len(task_names)} unique tasks: {sorted(task_names)}")


@test("android_world SDK availability")
def _():
    try:
        from android_world.env import env_launcher, json_action
        from android_world import registry, suite_utils
        print(f"  android_world SDK imported OK")
    except ImportError as e:
        print(f"  WARNING: android_world SDK not available: {e}")
        print(f"  This is expected if running outside the Docker container.")
        print(f"  env_worker.py will fail at import time without the SDK.")


@test("ray availability")
def _():
    try:
        import ray
        print(f"  Ray version: {ray.__version__}")
    except ImportError:
        print(f"  WARNING: Ray not installed")


@test("qwen_vl_utils availability")
def _():
    try:
        from qwen_vl_utils import process_vision_info
        print(f"  qwen_vl_utils.process_vision_info OK")
    except ImportError as e:
        print(f"  WARNING: qwen_vl_utils not available: {e}")
        print(f"  Needed for VLM image encoding in rollout.")


# ---------- Runner ----------

if __name__ == "__main__":
    passed, failed, warned = 0, 0, 0
    print("=" * 70)
    print("Android World Import Verification")
    print("=" * 70)

    for name, fn in TESTS:
        print(f"\n[TEST] {name}")
        try:
            fn()
            passed += 1
            print(f"  -> PASS")
        except Exception as e:
            if "WARNING" in str(e):
                warned += 1
                print(f"  -> WARN")
            else:
                failed += 1
                print(f"  -> FAIL: {e}")
                traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed, {warned} warnings")
    if failed == 0:
        print("All critical imports OK!")
    else:
        print("Some imports FAILED — see above for details.")
        sys.exit(1)
    print("=" * 70)
