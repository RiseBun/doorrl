"""Stage 1 — imagination / latent rollout utilities.

Everything in this package is used *only* for Stage 1 (imagination RL);
Stage 0 and the Table 3 pipeline are unaffected.
"""
from doorrl.imagination.task_reward import DEFAULT_REWARD_CFG, TaskRewardCfg, task_reward
from doorrl.imagination.imagination import ImaginedTrajectory, imagine_trajectory

__all__ = [
    "DEFAULT_REWARD_CFG",
    "ImaginedTrajectory",
    "TaskRewardCfg",
    "imagine_trajectory",
    "task_reward",
]
