"""Unit tests for ALFWorld eval helpers."""

from __future__ import annotations

from argparse import Namespace

from slime.utils.types import Sample


def _eval_sample(category: str, success: bool, ret: float) -> Sample:
    return Sample(
        reward=1.0 if success else 0.0,
        tokens=[1, 2, 3],
        response_length=1,
        loss_mask=[1],
        metadata={
            "category": category,
            "eval_success": success,
            "eval_return": ret,
            "eval_goal_condition_success_rate": 1.0 if success else 0.25,
            "eval_steps": 3,
            "eval_illegal_actions": 1 if not success else 0,
            "eval_total_actions": 3,
        },
    )


def test_proportional_quotas_sum_to_total():
    from examples.gtr_turbo.alfworld.data.gen_eval_valid_seen import _proportional_quotas

    quotas = _proportional_quotas(
        {
            "Pick & Place": [{}] * 10,
            "Pick Two & Place": [{}] * 10,
            "Clean & Place": [{}] * 5,
            "Heat & Place": [{}] * 5,
            "Cool & Place": [{}] * 5,
            "Examine in Light": [{}] * 5,
        },
        64,
    )

    assert sum(quotas.values()) == 64
    assert quotas["Pick & Place"] == quotas["Pick Two & Place"]


def test_parallel_worker_display_offsets(monkeypatch):
    import types

    from examples.gtr_turbo.alfworld.data import gen_eval_valid_seen

    seen = {}

    monkeypatch.setattr(
        "examples.gtr_turbo.alfworld.alf_utils.load_config_file",
        lambda path: {
            "xvfb_display_base": 190,
            "legacy_build_path": None,
            "render_image": True,
            "render_depth_image": False,
            "render_class_image": False,
            "render_object_image": True,
            "alfworld_config_file": "thor.yaml",
            "max_turns": 40,
        }
        if path == "config.yaml"
        else {},
    )
    monkeypatch.setattr("examples.gtr_turbo.alfworld.alf_utils.force_legacy_thor_build", lambda path: None)
    monkeypatch.setattr("examples.gtr_turbo.alfworld.alf_utils.install_alfworld_compat_patches", lambda: None)
    monkeypatch.setattr("examples.gtr_turbo.alfworld.env_worker._patch_flask_jinja2_compat", lambda: None)
    monkeypatch.setattr("examples.gtr_turbo.alfworld.env_worker._configure_render_flags", lambda **kwargs: None)
    monkeypatch.setattr(
        "examples.gtr_turbo.alfworld.env_worker._ensure_xvfb",
        lambda display: seen.setdefault("display", display),
    )

    class FakeEnv:
        def init_env(self, batch_size):
            seen["batch_size"] = batch_size

    fake_module = types.SimpleNamespace(
        agents=types.SimpleNamespace(
            environment=types.SimpleNamespace(
                alfred_thor_env=types.SimpleNamespace(AlfredThorEnv=lambda config, train_eval: FakeEnv())
            )
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "alfworld", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "alfworld.agents", fake_module.agents)
    monkeypatch.setitem(__import__("sys").modules, "alfworld.agents.environment", fake_module.agents.environment)
    monkeypatch.setitem(
        __import__("sys").modules,
        "alfworld.agents.environment.alfred_thor_env",
        fake_module.agents.environment.alfred_thor_env,
    )

    env, max_steps = gen_eval_valid_seen._build_raw_alf_env("config.yaml", display=None, worker_id=5)

    assert isinstance(env, FakeEnv)
    assert max_steps == 40
    assert seen["display"] == ":195"
    assert seen["batch_size"] == 1


def test_log_eval_rollout_records_success_rate_and_return(monkeypatch):
    from examples.gtr_turbo.alfworld.metrics import log_eval_rollout

    logged = {}

    def fake_log(args, log_dict, step_key=None):
        logged.update(log_dict)

    monkeypatch.setattr("slime.utils.logging_utils.log", fake_log)

    args = Namespace(
        wandb_always_use_train_step=False,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=1,
    )
    data = {
        "alfworld_valid_seen": {
            "samples": [
                _eval_sample("Pick & Place", True, 51.0),
                _eval_sample("Pick & Place", False, -1.0),
                _eval_sample("Clean & Place", True, 50.5),
            ]
        }
    }

    handled = log_eval_rollout(rollout_id=0, args=args, data=data, extra_metrics={})

    assert handled is True
    assert logged["eval/alfworld_valid_seen/success_rate"] == 2 / 3
    assert logged["eval/alfworld_valid_seen/mean_return"] == (51.0 - 1.0 + 50.5) / 3
    assert logged["eval/alfworld_valid_seen/pick_place/success_rate"] == 0.5
    assert logged["eval/alfworld_valid_seen/clean_place/success_rate"] == 1.0
