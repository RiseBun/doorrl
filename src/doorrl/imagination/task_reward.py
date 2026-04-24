"""Task-shaped reward used during imagination rollouts.

Stage 0 has ``reward ≡ 0`` on nuScenes (no RL supervision signal). To
train actor-critic in imagination we need *something* to reward, so we
synthesise a simple, interpretable shaped reward directly from the
world model's predictions. Keeping this file tiny and explicit is
intentional: the paper claim is about the **representation**, not the
reward engineering, so the reward must be boring enough that the
reader doesn't ask "did you shape your way to the result?".

reward_t  =  w_prog * v_forward_ego(next_tokens_t)       # progress
            - w_coll * sigmoid(collision_pred_t)         # safety
            - w_act  * ||action_t||^2                    # comfort
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


# Token dim convention (see src/doorrl/adapters/base.py::_fill_token):
# [x, y, vx, vy, length, width, risk, visibility, ttc, ...]
_EGO_VX_DIM = 2


@dataclass(frozen=True)
class TaskRewardCfg:
    w_prog: float = 1.0
    w_coll: float = 5.0
    w_act: float = 0.01
    # Final safety clamp on per-step reward. Without this, a single
    # badly-sampled action (e.g. magnitude 170 right after
    # initialisation, before the tanh-bound fix) would produce
    # ``comfort = -0.01 * 170^2 = -289`` and cascade through the critic
    # into the world-model gradients. Even with the tanh bound on
    # action_mean plus log_std clamping we keep this belt-and-suspenders
    # clamp so one pathological sample can never poison the whole epoch.
    reward_clip: float = 5.0


DEFAULT_REWARD_CFG = TaskRewardCfg()


def task_reward(
    predicted_next_tokens: torch.Tensor,   # [B, K, raw_dim]
    predicted_collision: torch.Tensor,     # [B] (logit)
    action: torch.Tensor,                  # [B, action_dim]
    ego_slot_index: torch.Tensor | None = None,  # [B], which slot is ego
    cfg: TaskRewardCfg = DEFAULT_REWARD_CFG,
) -> torch.Tensor:
    """Return per-sample shaped reward [B].

    ``ego_slot_index`` tells us which of the K predicted slots
    corresponds to ego. For top-k variants with ``force_ego=True`` the
    ego lives at slot 0 (from Stage 0 abstraction). For set-prediction
    variants (holistic_16slot) it's ambiguous; we fall back to slot 0,
    which is accurate enough for a shaped-reward regime where the
    exact per-agent identity of the "progress" signal matters less
    than its *direction*.
    """
    if ego_slot_index is None:
        vx_next = predicted_next_tokens[:, 0, _EGO_VX_DIM]
    else:
        B = predicted_next_tokens.size(0)
        ar = torch.arange(B, device=predicted_next_tokens.device)
        vx_next = predicted_next_tokens[ar, ego_slot_index, _EGO_VX_DIM]

    progress = cfg.w_prog * vx_next
    safety = -cfg.w_coll * torch.sigmoid(predicted_collision)
    comfort = -cfg.w_act * (action ** 2).sum(dim=-1)

    reward = progress + safety + comfort
    if cfg.reward_clip is not None:
        reward = reward.clamp(-cfg.reward_clip, cfg.reward_clip)
    return reward
