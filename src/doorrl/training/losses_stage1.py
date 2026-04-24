"""Stage 1 imagination-RL losses.

    L = L_actor + alpha_c * L_critic + alpha_s * L_sanity

- L_actor  : -E[ log pi(a_t|s_t) * stop_grad(adv_t) ] - beta * H(pi)
- L_critic : 0.5 * E[ (V(s_t) - stop_grad(ret_t))^2 ]
- L_sanity : Stage 0 losses on the real t=0 batch (obs/reward/continue/
             collision), keeps the world model grounded while we train
             actor/critic through imagination.

Advantage/return computed with GAE-lambda and per-step continue
probability as the discount mask.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from doorrl.imagination.imagination import ImaginedTrajectory
from doorrl.schema import SceneBatch
from doorrl.training.losses import compute_losses as compute_stage0_losses
from doorrl.config import TrainingConfig


@dataclass(frozen=True)
class Stage1LossCfg:
    gamma: float = 0.97
    lam: float = 0.95
    entropy_beta: float = 0.01
    critic_weight: float = 0.5
    sanity_weight: float = 1.0
    # Huber delta for critic loss. MSE on values exploded at epoch-9 in
    # the pilot (bootstrap feedback pushing |return| beyond 100; squared
    # target amplified gradients catastrophically). Huber is linear
    # beyond |delta|, so runaway targets produce bounded gradient per
    # sample — value still learns, but a single bad batch can no longer
    # hammer the encoder. Delta=10 covers the bulk of well-behaved
    # returns (reward clip is ±5 over K=5 steps → ~±25 max, values
    # typically in [0, 30]) and engages linear regime only for the
    # pathological tail.
    critic_huber_delta: float = 10.0


def _gae_lambda(
    rewards: torch.Tensor,     # [B, K]
    values: torch.Tensor,      # [B, K+1]
    continues: torch.Tensor,   # [B, K]  in [0,1]
    gamma: float,
    lam: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (advantages, returns), each [B, K]."""
    B, K = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(K)):
        not_done = continues[:, t]
        delta = rewards[:, t] + gamma * not_done * values[:, t + 1] - values[:, t]
        gae = delta + gamma * lam * not_done * gae
        advantages[:, t] = gae
    returns = advantages + values[:, :-1]
    return advantages, returns


def stage1_losses(
    traj: ImaginedTrajectory,
    real_batch: SceneBatch,
    stage0_cfg: TrainingConfig,
    cfg: Stage1LossCfg = Stage1LossCfg(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total Stage 1 loss and diagnostic scalars."""
    advantages, returns = _gae_lambda(
        traj.rewards, traj.values, traj.continues, cfg.gamma, cfg.lam,
    )
    # Normalise advantages for stable policy-gradient scale.
    adv_detached = advantages.detach()
    if adv_detached.numel() > 1:
        adv_detached = (adv_detached - adv_detached.mean()) / (adv_detached.std() + 1e-6)

    # --- Actor loss ----------------------------------------------------
    policy_grad = -(traj.log_probs * adv_detached).mean()
    # Diagonal-Gaussian entropy = 0.5 * sum(log(2*pi*e) + 2*log_std).
    log2pi_e = float(torch.log(torch.tensor(2.0 * torch.pi * 2.718281828))) / 2.0
    # (constant; only the log_std term matters for gradient.)
    entropy = (log2pi_e + traj.action_log_stds).sum(dim=-1).mean()
    actor_loss = policy_grad - cfg.entropy_beta * entropy

    # --- Critic loss (Huber) ------------------------------------------
    # smooth_l1_loss with beta=delta: 0.5*(x/beta)^2*beta if |x|<beta,
    # else |x|-0.5*beta. Equivalent to Huber(delta=beta). Bounds
    # gradient magnitude to delta per sample even when returns drift
    # out of distribution.
    value_pred = traj.values[:, :-1]  # [B, K]
    critic_loss = torch.nn.functional.smooth_l1_loss(
        value_pred, returns.detach(), beta=cfg.critic_huber_delta, reduction="mean",
    )

    # --- Sanity loss (Stage 0 losses on the real t=0 batch) -----------
    sanity_total, sanity_stats = compute_stage0_losses(
        real_batch, traj.first_output, stage0_cfg,
    )

    total = actor_loss + cfg.critic_weight * critic_loss + cfg.sanity_weight * sanity_total

    stats = {
        "total": float(total.detach().item()),
        "actor": float(actor_loss.detach().item()),
        "policy_grad": float(policy_grad.detach().item()),
        "entropy": float(entropy.detach().item()),
        "critic": float(critic_loss.detach().item()),
        "sanity": float(sanity_total.detach().item()),
        "adv_mean": float(advantages.detach().mean().item()),
        "adv_std": float(advantages.detach().std().item()),
        "ret_mean": float(returns.detach().mean().item()),
        "value_mean": float(traj.values.detach().mean().item()),
        "reward_mean": float(traj.rewards.detach().mean().item()),
        "collision_max": float(traj.collisions.detach().max(dim=1).values.mean().item()),
        # Diagnostic signals for the pilot-NaN post-mortem: if action
        # magnitude or log_std drift back towards the pre-fix disaster
        # zone, these will surface it long before critic loss explodes.
        "action_abs_max": float(traj.actions.detach().abs().max().item()),
        "action_abs_mean": float(traj.actions.detach().abs().mean().item()),
        "log_std_mean": float(traj.action_log_stds.detach().mean().item()),
    }
    # Keep sanity breakdown under a namespace so keys don't collide.
    for k, v in sanity_stats.items():
        stats[f"sanity_{k}"] = v
    return total, stats


def bc_loss(
    output,
    batch: SceneBatch,
    stage0_cfg: TrainingConfig,
    bc_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BC baseline: regress sampled-action mean towards GT actions,
    plus the Stage 0 losses for WM stability. No imagination, no value.
    """
    action_mean = output.policy.action_mean
    bc_mse = ((action_mean - batch.actions) ** 2).mean()
    sanity_total, sanity_stats = compute_stage0_losses(batch, output, stage0_cfg)
    total = bc_weight * bc_mse + sanity_total
    stats = {
        "total": float(total.detach().item()),
        "bc_mse": float(bc_mse.detach().item()),
        "sanity": float(sanity_total.detach().item()),
    }
    for k, v in sanity_stats.items():
        stats[f"sanity_{k}"] = v
    return total, stats


def ac1_loss(
    traj: ImaginedTrajectory,
    real_batch: SceneBatch,
    stage0_cfg: TrainingConfig,
    cfg: Stage1LossCfg = Stage1LossCfg(),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Model-free-ish baseline: 1-step TD AC using the *imagined*
    reward at t=0 plus bootstrapped V(s_1). Equivalent to
    ``stage1_losses`` with horizon=1 but kept as a separate entry
    point for clarity; the caller already restricts the trajectory to
    K=1 when building this baseline.
    """
    return stage1_losses(traj, real_batch, stage0_cfg, cfg)
