from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from doorrl.config import ModelConfig


@dataclass
class PolicyOutput:
    action_mean: torch.Tensor
    action_log_std: torch.Tensor
    value: torch.Tensor


class ActorCriticHead(nn.Module):
    """Actor-critic head for DOOR-RL.

    The policy is a diagonal Gaussian over a 2-D action (for nuScenes:
    linear velocity and yaw rate; for nuPlan: the same schema after
    preprocessing). Two numerical-stability guards are applied at the
    output:

    1. ``action_mean`` is passed through a scaled ``tanh`` so it cannot
       diverge beyond a physically reasonable range. Without this the
       un-regularised actor gradient from Stage 1 imagination RL can
       push the mean to tens or hundreds in a handful of steps, which
       then blows up the (action^2)-weighted comfort reward and ripples
       through the critic and world model (observed in pilot run:
       ``R ~= -285`` after epoch 1, NaN by epoch 3).
    2. ``action_log_std`` is clamped. Diagonal-Gaussian entropy and
       reparameterised sampling both use ``exp(log_std)`` and ``log_std``
       directly, so letting the parameter drift to large magnitudes
       makes sampling variance infinite or zero -- both numerically
       catastrophic under BF16.

    ``ACTION_MEAN_BOUND``, ``LOG_STD_MIN``, and ``LOG_STD_MAX`` are set
    conservatively so the Stage 0 evaluation path (which only reads
    ``action_mean``) behaves identically to before the fix for any
    reasonable latent, while Stage 1 rollouts get bounded samples.
    """

    ACTION_MEAN_BOUND: float = 3.0      # tanh-scaled ceiling on |action_mean|
    LOG_STD_MIN: float = -2.0           # std_min = exp(-2) ~= 0.135
    LOG_STD_MAX: float = 0.5            # std_max = exp(0.5) ~= 1.65

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        # Initialise log_std at 0 (std=1) but immediately clamp on forward;
        # see class docstring for rationale.
        self.action_log_std = nn.Parameter(torch.zeros(config.action_dim))

    def forward(self, latent: torch.Tensor) -> PolicyOutput:
        raw_mean = self.policy(latent)
        action_mean = self.ACTION_MEAN_BOUND * torch.tanh(
            raw_mean / self.ACTION_MEAN_BOUND
        )
        log_std = self.action_log_std.clamp(
            self.LOG_STD_MIN, self.LOG_STD_MAX
        ).unsqueeze(0).expand_as(action_mean)
        value = self.value(latent).squeeze(-1)
        return PolicyOutput(action_mean=action_mean, action_log_std=log_std, value=value)
