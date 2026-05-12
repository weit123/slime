"""Verification Step 2: Env pool acquire/release and worker basic ops.

Run from the slime repo root (requires Android emulators + Ray):
    python examples/android_world/tests/test_env_pool.py

This test:
1. Creates a small env pool (2 workers)
2. Tests acquire/release cycle
3. Resets a task on an acquired worker
4. Takes a few steps with fixed actions
5. Verifies observations contain screenshots
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

import numpy as np

# Ensure slime is importable
sys.path.insert(0, os.getcwd())


async def test_pool_acquire_release():
    """Test that pool acquire/release works without leaking workers."""
    from examples.android_world.env_pool import AndroidWorldEnvPool

    config = {
        "num_workers": 2,
        "avd_name": os.environ.get("AW_AVD_NAME", "AndroidWorldAvd"),
        "base_avd_name_pattern": "slime_test_aw_{}",
        "base_console_port": 5556,
        "base_grpc_port": 8554,
        "base_adb_server_port": 5037,
        "android_avd_home": os.environ.get("ANDROID_AVD_HOME", "/root/android/avd/"),
        "android_sdk_root": os.environ.get("ANDROID_SDK_ROOT", "/root/android/"),
        "emulator_path": os.environ.get("ANDROID_EMULATOR", "/root/android/emulator/emulator"),
        "adb_path": os.environ.get("ANDROID_ADB", "/root/android/platform-tools/adb"),
        "max_turns": 5,
        "temp_path": "/tmp/android_world_test",
        "task_family": "android_world",
        "save_images": False,
        "resources_per_worker": {"num_cpus": 3, "memory": 8 * 1024 * 1024 * 1024},
    }

    print("\n[1] Creating pool with 2 workers (this boots emulators, may be slow)...")
    t0 = time.time()
    pool = await AndroidWorldEnvPool.get_instance(config)
    print(f"    Pool created in {time.time() - t0:.1f}s")
    assert pool._num_workers == 2
    assert pool._available.qsize() == 2
    print("    PASS: Pool has 2 available workers")

    print("\n[2] Acquire worker 1...")
    w1_ref, w1_id = await pool.acquire()
    assert pool._available.qsize() == 1
    print(f"    Acquired worker {w1_id}, {pool._available.qsize()} remaining")

    print("\n[3] Acquire worker 2...")
    w2_ref, w2_id = await pool.acquire()
    assert pool._available.qsize() == 0
    print(f"    Acquired worker {w2_id}, {pool._available.qsize()} remaining")

    print("\n[4] Release worker 1...")
    pool.release(w1_id)
    assert pool._available.qsize() == 1
    print(f"    Released worker {w1_id}, {pool._available.qsize()} available")

    print("\n[5] Re-acquire (should get worker 1 back)...")
    w3_ref, w3_id = await pool.acquire()
    assert w3_id == w1_id, f"Expected {w1_id}, got {w3_id}"
    assert pool._available.qsize() == 0
    print(f"    Re-acquired worker {w3_id}, correct!")

    # Release all
    pool.release(w2_id)
    pool.release(w3_id)
    assert pool._available.qsize() == 2
    print("\n    PASS: Acquire/release cycle works correctly, no leaks")

    return pool


async def test_worker_reset_and_step(pool):
    """Test that a worker can reset a task and execute steps."""
    import ray

    print("\n[6] Acquiring worker for task test...")
    w_ref, w_id = await pool.acquire()

    task_name = "ContactsAddContact"
    params_idx = 0

    print(f"\n[7] Resetting worker {w_id} with task '{task_name}'...")
    t0 = time.time()
    obs, info = ray.get(w_ref.reset.remote(task_name, params_idx))
    print(f"    Reset completed in {time.time() - t0:.1f}s")

    assert obs is not None, "Observation is None after reset"
    assert "image" in obs, "No 'image' in observation"
    assert "task" in obs, "No 'task' in observation"
    assert isinstance(obs["image"], np.ndarray), f"Image is {type(obs['image'])}, expected ndarray"
    print(f"    Task: {info.get('task', '')[:80]}...")
    print(f"    Max steps: {info.get('max_steps')}")
    print(f"    Screenshot shape: {obs['image'].shape}, dtype: {obs['image'].dtype}")
    print("    PASS: Reset returns valid observation with screenshot")

    # Fixed actions for testing
    test_actions = [
        '<thinking>Waiting to observe.</thinking><tool_call>{"name": "mobile_use", "arguments": {"action": "wait", "time": 1.0}}</tool_call><conclusion>Waited.</conclusion>',
        '<thinking>Going home.</thinking><tool_call>{"name": "mobile_use", "arguments": {"action": "system_button", "button": "Home"}}</tool_call><conclusion>Pressed home.</conclusion>',
        '<thinking>Terminating.</thinking><tool_call>{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}</tool_call><conclusion>Done.</conclusion>',
    ]

    for i, action in enumerate(test_actions):
        print(f"\n[{8 + i}] Step {i + 1}: {action[60:120]}...")
        t0 = time.time()
        step_obs, reward, done, step_info = ray.get(w_ref.step.remote(action))
        print(f"    Step completed in {time.time() - t0:.1f}s")
        print(f"    reward={reward:.3f}, done={done}, info={step_info}")

        if done:
            print(f"    Episode ended. won={step_info.get('won', False)}")
            break
        else:
            assert step_obs is not None, "Non-terminal step returned None obs"
            assert isinstance(step_obs["image"], np.ndarray)
            print(f"    Screenshot shape: {step_obs['image'].shape}")

    print("    PASS: Worker reset/step cycle works")

    pool.release(w_id)


async def test_env_wrapper(pool):
    """Test the AndroidWorldEnv wrapper."""
    from examples.android_world.env_android_world import AndroidWorldEnv

    print("\n[11] Testing AndroidWorldEnv wrapper...")
    w_ref, w_id = await pool.acquire()

    env = AndroidWorldEnv(
        worker_ref=w_ref,
        worker_id=w_id,
        pool=pool,
        task_name="CameraTakePhoto",
        params_idx=0,
        max_turns=5,
    )

    print("\n[12] Resetting via env wrapper...")
    obs, info = env.reset()
    assert obs is not None
    print(f"    Task: {info.get('task', '')[:80]}...")

    print("\n[13] format_observation()...")
    msg = env.format_observation(obs)
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    has_image = any(c.get("type") == "image" for c in msg["content"])
    has_text = any(c.get("type") == "text" for c in msg["content"])
    assert has_image, "No image in formatted observation"
    assert has_text, "No text in formatted observation"
    text_content = [c for c in msg["content"] if c.get("type") == "text"][0]["text"]
    assert "user query" in text_content.lower() or "task_description" in text_content.lower() or info["task"] in text_content
    print(f"    Message has image={has_image}, text={has_text}")
    print(f"    Text preview: {text_content[:100]}...")
    print("    PASS: format_observation produces valid VLM message")

    print("\n[14] Step via env wrapper...")
    action = '<thinking>Waiting.</thinking><tool_call>{"name": "mobile_use", "arguments": {"action": "wait", "time": 1.0}}</tool_call><conclusion>Waited.</conclusion>'
    step_obs, done, step_info = env.step(action)
    print(f"    done={done}, final_reward={env.final_reward}")
    assert not done, "Single wait action should not end episode"

    print("\n[15] Close env wrapper (release back to pool)...")
    env.close()
    assert pool._available.qsize() >= 1, "Worker not released back to pool"
    print("    PASS: env.close() releases worker back to pool")


async def main():
    print("=" * 70)
    print("Android World Env Pool & Worker Test")
    print("=" * 70)

    import ray
    if not ray.is_initialized():
        ray.init()

    pool = await test_pool_acquire_release()
    await test_worker_reset_and_step(pool)
    await test_env_wrapper(pool)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)

    # Cleanup
    print("\nCleaning up pool...")
    await pool.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
