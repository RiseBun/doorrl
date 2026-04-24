"""Stage 1 evaluation: latent return, imagined collision, rollout stability.

All three metrics are computed by rolling the (trained) model forward
``horizon`` steps on each held-out val batch under the deterministic
policy and aggregating. None of them require a simulator or any
Stage-0 ground truth beyond the initial encoding state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch

from doorrl.imagination.imagination import imagine_trajectory
from doorrl.imagination.task_reward import DEFAULT_REWARD_CFG, TaskRewardCfg


@dataclass
class Stage1Metrics:
    latent_return_mean: float
    latent_return_std: float
    imagined_collision_rate: float   # fraction of rollouts with max_t > 0.5
    collision_mean: float            # mean of max-over-time sigmoid(coll)
    # Variant-agnostic stability: relative change of the *ego slot*
    # latent over the imagination horizon. Ego is always slot-0 for
    # force_ego top-k variants, so this is directly comparable across
    # object_only / decoupled / (+vis) runs — unlike ``rollout_stability``
    # below which depends on how each variant constructs global_latent.
    rollout_stability: float         # primary metric: ego-slot relative drift
    rollout_stability_global: float  # legacy: global_latent relative drift
    horizon: int
    n_samples: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "latent_return_mean": self.latent_return_mean,
            "latent_return_std": self.latent_return_std,
            "imagined_collision_rate": self.imagined_collision_rate,
            "collision_mean": self.collision_mean,
            "rollout_stability": self.rollout_stability,
            "rollout_stability_global": self.rollout_stability_global,
            "horizon": self.horizon,
            "n_samples": self.n_samples,
        }


def _stability_score(latents: torch.Tensor) -> torch.Tensor:
    """Per-sample mean step-over-step cosine distance of a latent signal.

    Returns [B]. Applied to whatever [B, K+1, D] trajectory the caller
    considers the canonical "state" — currently ``ego_latents`` for the
    primary metric and ``global_latents`` for the legacy metric.

    Why cosine and not relative L2:
      * The decoupled(+vis) variant multiplies the dyn latent by
        ``visibility ∈ [0, 1]`` before abstraction. When the WM-imagined
        next visibility drifts towards 0 (very possible in BF16), the
        post-scale ego-slot vector collapses to ~0, and the relative
        change ``||Δ||/||e||`` diverges (denominator floor → tiny,
        numerator → finite), giving nonsensical stability numbers like
        2e6 even though the underlying *direction* of the ego
        representation is perfectly stable.
      * Cosine distance 1 - cos(e_t, e_{t+1}) is scale-invariant, so
        visibility weighting (a non-negative scalar multiplier) doesn't
        affect it. It is also bounded in [0, 2] which makes cross-run
        aggregation well-behaved.
    """
    a = latents[:, :-1, :]  # [B, K, D]
    b = latents[:, 1:, :]   # [B, K, D]
    # eps=1e-3 handles the degenerate all-zero case (e.g. visibility==0
    # for ego) gracefully: cosine returns 0 rather than NaN/inf.
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1, eps=1e-3)
    return (1.0 - cos).mean(dim=1)                     # [B], ∈ [0, 2]


@torch.no_grad()
def evaluate_stage1(
    model,
    val_loader,
    device: torch.device,
    horizon: int = 5,
    reward_cfg: TaskRewardCfg = DEFAULT_REWARD_CFG,
) -> Stage1Metrics:
    model.eval()
    returns: List[torch.Tensor] = []
    coll_max: List[torch.Tensor] = []
    stab_ego: List[torch.Tensor] = []
    stab_global: List[torch.Tensor] = []
    n = 0

    for batch in val_loader:
        batch = batch.to(device)
        traj = imagine_trajectory(
            model, batch,
            horizon=horizon, deterministic=True,
            reward_cfg=reward_cfg,
            detach_world_model=True,
        )
        # Sum of rewards over horizon (undiscounted, as per §4).
        returns.append(traj.rewards.sum(dim=1))           # [B]
        coll_max.append(traj.collisions.max(dim=1).values)  # [B]
        stab_ego.append(_stability_score(traj.ego_latents))       # [B]
        stab_global.append(_stability_score(traj.global_latents)) # [B]
        n += batch.tokens.size(0)

    if not returns:
        return Stage1Metrics(
            latent_return_mean=float("nan"),
            latent_return_std=float("nan"),
            imagined_collision_rate=float("nan"),
            collision_mean=float("nan"),
            rollout_stability=float("nan"),
            rollout_stability_global=float("nan"),
            horizon=horizon,
            n_samples=0,
        )

    R = torch.cat(returns)
    C = torch.cat(coll_max)
    S_ego = torch.cat(stab_ego)
    S_glob = torch.cat(stab_global)
    return Stage1Metrics(
        latent_return_mean=float(R.mean().item()),
        latent_return_std=float(R.std().item()),
        imagined_collision_rate=float((C > 0.5).float().mean().item()),
        collision_mean=float(C.mean().item()),
        rollout_stability=float(S_ego.mean().item()),
        rollout_stability_global=float(S_glob.mean().item()),
        horizon=horizon,
        n_samples=n,
    )
