"""ALFWorld environment integration for slime RL training.

This module provides a Ray actor-based environment wrapper for ALFWorld,
a text-based embodied AI benchmark built on AI2-THOR.

Key components:
- alf_utils.py: Adapted from GTR-Turbo (AlfEnv wrapper, action processing)
- env_worker.py: Ray actor wrapping AlfEnv
- env_pool.py: Singleton async pool of Ray actors
- env_alfworld.py: Per-sample environment for slime rollout
- rollout.py: Incremental multi-turn rollout
- rollout_history.py: History-based multi-turn rollout
- prompts.py: Prompt templates (matching GTR-Turbo format)
"""
