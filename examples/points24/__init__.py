"""Points24 (24 Game) environment integration for slime RL training.

This module provides a lightweight environment wrapper for the classic 24 Game,
where the agent must use four playing cards with operators +, -, *, / and
parentheses to create an expression that equals 24.

Key components:
- env_worker.py: Lightweight environment wrapper (no Ray actor needed)
- env_points24.py: Per-sample environment for slime rollout
- rollout.py: Incremental multi-turn rollout
- rollout_history.py: History-based multi-turn rollout
- prompts.py: Prompt templates (matching GTR-Turbo format)
"""
