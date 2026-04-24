"""Multi-step imagination rollout for Stage 1.

Runs the model forward for ``horizon`` steps under a sampled (or
deterministic) policy. At each step:
    1. encode current tokens  ->  per-token latents
    2. abstract               ->  K selected slots
    3. sample action from actor-critic(global_latent)
    4. world_model(selected_tokens, action) -> predicted_next_tokens
    5. **Re-encode**: scatter predicted_next_tokens back into the
       97-slot token tensor at the *selected_indices* positions and
       step forward. Non-selected slots are kept as context (frozen).

Caveats documented inline:
    * For set-prediction variants (holistic_16slot) the selected_indices
      is a placeholder (all zeros), so we cannot scatter predictions
      back to the original token layout. For those variants we keep the
      tokens fixed across imagination steps ("frozen-world approximation");
      the actor-critic still learns because different sampled actions
      produce different WM predictions and therefore different rewards.
      This is a known limitation of non-object-centric representations
      under a token-space rollout; it is part of the fair comparison.
    * Gradients flow through the whole loop by default so actor/critic
      losses can shape the world model. Use ``deterministic=True`` and
      ``torch.no_grad()`` from the caller for evaluation-time rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from doorrl.imagination.task_reward import TaskRewardCfg, task_reward, DEFAULT_REWARD_CFG
from doorrl.schema import SceneBatch


@dataclass
class ImaginedTrajectory:
    """Container for a K-step imagination rollout.

    All tensors are stacked along the time dimension.

    Attributes
    ----------
    actions         [B, K, action_dim]    sampled (or deterministic) actions
    action_means    [B, K, action_dim]    policy means (for logging / det eval)
    action_log_stds [B, K, action_dim]
    log_probs       [B, K]                log π(action_t | s_t)
    values          [B, K+1]              V(s_0), V(s_1), ..., V(s_K)
    rewards         [B, K]                shaped reward (task + wm reward head)
    collisions      [B, K]                sigmoid(collision_pred_t) in [0, 1]
    continues       [B, K]                sigmoid(continue_pred_t)  in [0, 1]
    global_latents  [B, K+1, D]           per-step abstract global latent
                                           (variant-dependent definition —
                                           NOT used for stability; kept for
                                           logging / backward compat).
    ego_latents     [B, K+1, D]           per-step ego-slot latent
                                           (selected_tokens[:, 0, :]). This
                                           is the variant-agnostic signal
                                           used by the rollout-stability
                                           metric: slot 0 is ego for every
                                           ``force_ego=True`` variant
                                           (object_only, decoupled, ...),
                                           so the quantity has the same
                                           physical meaning across models.
    first_output                           the t=0 DoorRLOutput; caller uses
                                           it to compute the sanity loss on
                                           the real transition without a
                                           second forward pass.
    """
    actions: torch.Tensor
    action_means: torch.Tensor
    action_log_stds: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    collisions: torch.Tensor
    continues: torch.Tensor
    global_latents: torch.Tensor
    ego_latents: torch.Tensor
    first_output: object
    horizon: int


# Default hard cap on any sampled action value. With tanh-bounded mean
# in [-3, 3] and log_std <= 0.5 (std ~= 1.65), a 3-sigma sample is ~8; we
# clip slightly above that to keep reasonable samples intact while
# ruling out pathological tails from the reparameterised normal. The
# value is overridable per-call so follow-up ablations can tighten it
# (e.g. 5.0) to stop an over-exploring actor from saturating the clip.
_ACTION_SAMPLE_CLIP: float = 8.0


def _sample_action(
    mean: torch.Tensor,
    log_std: torch.Tensor,
    deterministic: bool,
    sample_clip: float = _ACTION_SAMPLE_CLIP,
):
    """Diagonal gaussian sample with reparameterization. Returns (action, log_prob)."""
    if deterministic:
        action = mean
    else:
        eps = torch.randn_like(mean)
        action = mean + torch.exp(log_std) * eps
    action = action.clamp(-sample_clip, sample_clip)
    var = torch.exp(2.0 * log_std).clamp_min(1e-8)
    # log N(a | mu, sigma^2) summed over action dims.
    log_prob = -0.5 * (
        ((action - mean) ** 2) / var + 2.0 * log_std + torch.log(torch.tensor(
            2.0 * torch.pi, device=action.device, dtype=action.dtype,
        ))
    ).sum(dim=-1)
    return action, log_prob


def _scatter_back_predictions(
    tokens: torch.Tensor,            # [B, S, raw_dim]
    selected_indices: torch.Tensor,  # [B, K]
    selected_mask: torch.Tensor,     # [B, K]
    predicted_next: torch.Tensor,    # [B, K, raw_dim]
) -> torch.Tensor:
    """Write predicted_next[b, k] into tokens[b, selected_indices[b, k], :]
    for every k where selected_mask[b, k] is True. Non-selected positions
    are left unchanged.
    """
    B, S, D = tokens.shape
    K = selected_indices.size(1)
    out = tokens.clone()

    # For masked-out slots, point them at slot 0 and later overwrite with
    # the original slot-0 value -> no-op. This is the simplest correct
    # scatter under variable-length masks and avoids Python loops.
    safe_idx = torch.where(selected_mask, selected_indices, torch.zeros_like(selected_indices))
    expanded_idx = safe_idx.unsqueeze(-1).expand(-1, -1, D)

    # Predictions for masked slots must not overwrite anything -> replace
    # them with the value currently in tokens[b, 0] so scatter is a no-op.
    original_slot0 = out[:, 0:1, :].expand(-1, K, -1)
    preds = torch.where(
        selected_mask.unsqueeze(-1), predicted_next, original_slot0,
    )
    out.scatter_(1, expanded_idx, preds)
    return out


def imagine_trajectory(
    model,
    batch: SceneBatch,
    horizon: int = 5,
    deterministic: bool = False,
    reward_cfg: TaskRewardCfg = DEFAULT_REWARD_CFG,
    detach_world_model: bool = False,
    action_sample_clip: float = _ACTION_SAMPLE_CLIP,
) -> ImaginedTrajectory:
    """Rollout the model ``horizon`` steps in imagination.

    Parameters
    ----------
    model : DoorRLModelVariant
        Must expose ``encoder``, ``abstraction``, ``world_model``,
        ``policy`` (or be a variant whose ``forward`` returns a
        :class:`DoorRLOutput`). For the first step we call
        ``model(batch)`` to reuse all the variant-specific logic
        (holistic / object_only / decoupled, etc.); for subsequent
        steps we construct a minimal SceneBatch with the imagined
        tokens and call ``model(...)`` again.

    detach_world_model : bool
        If True, the world model is treated as a fixed simulator — no
        gradients flow into its parameters from the AC loss. Used by
        the ``ac1`` baseline (model-free-ish).
    """
    B, S, D = batch.tokens.shape
    device = batch.tokens.device

    # Step 0: first full forward with the provided batch. The WM inside
    # uses batch.actions which we throw away; we'll re-call world_model
    # with our *sampled* action below.
    first_output = model(batch)
    first_abstr = first_output.abstraction
    first_policy = first_output.policy

    action_0, logp_0 = _sample_action(
        first_policy.action_mean, first_policy.action_log_std, deterministic,
        sample_clip=action_sample_clip,
    )

    actions_list: List[torch.Tensor] = [action_0]
    means_list = [first_policy.action_mean]
    log_stds_list = [first_policy.action_log_std]
    log_probs_list = [logp_0]
    values_list = [first_policy.value]
    rewards_list: List[torch.Tensor] = []
    collisions_list: List[torch.Tensor] = []
    continues_list: List[torch.Tensor] = []
    global_latents_list = [first_abstr.global_latent]
    # Ego slot latent (slot 0) — variant-agnostic rollout-stability signal.
    # For holistic set-prediction variants slot 0 is a learned query rather
    # than ego specifically; the metric is still well-defined per-variant,
    # just with a slightly different physical interpretation, which is a
    # known limitation of cross-variant comparison for that one class.
    ego_latents_list = [first_abstr.selected_tokens[:, 0, :]]

    # Rolling state.
    cur_tokens = batch.tokens
    cur_types = batch.token_types
    cur_mask = batch.token_mask
    cur_abstr = first_abstr

    for t in range(horizon):
        action_t = actions_list[t]

        # WM call. If detach_world_model, cut gradients into WM params.
        wm_ctx = torch.no_grad() if detach_world_model else torch.enable_grad()
        with wm_ctx:
            wm_out = model.world_model(
                selected_tokens=cur_abstr.selected_tokens,
                selected_mask=cur_abstr.selected_mask,
                actions=action_t,
            )

        # Reward & safety signals from this step.
        r_shaped = task_reward(
            predicted_next_tokens=wm_out.predicted_next_tokens,
            predicted_collision=wm_out.predicted_collision,
            action=action_t,
            ego_slot_index=None,  # slot 0 = ego for top-k variants with force_ego
            cfg=reward_cfg,
        )
        r_wm = wm_out.predicted_reward  # Stage-0 untrained head ~ 0, harmless
        rewards_list.append(r_shaped + r_wm)
        collisions_list.append(torch.sigmoid(wm_out.predicted_collision))
        continues_list.append(torch.sigmoid(wm_out.predicted_continue))

        if t == horizon - 1:
            # Bootstrap value from current state's policy.value already
            # captured at t (values_list[t] is V(s_t)). We still need
            # V(s_{horizon}) for GAE, which is computed below after the
            # final re-encode+abstract.
            pass

        # ---- Advance to t+1 -----------------------------------------
        # Scatter predicted_next into tokens (skip for set-prediction).
        if cur_abstr.is_set_prediction:
            # Frozen-world approximation: keep tokens as-is. See header.
            next_tokens = cur_tokens
        else:
            next_tokens = _scatter_back_predictions(
                cur_tokens,
                cur_abstr.selected_indices,
                cur_abstr.selected_mask,
                wm_out.predicted_next_tokens,
            )

        # Re-encode + re-abstract + policy at t+1.
        # We build a minimal SceneBatch the variant's forward can consume.
        dummy_action = torch.zeros_like(batch.actions)  # ignored; we re-call WM
        next_batch = SceneBatch(
            tokens=next_tokens,
            token_mask=cur_mask,
            token_types=cur_types,
            actions=dummy_action,
            next_tokens=next_tokens,
            rewards=torch.zeros(B, device=device),
            continues=torch.ones(B, device=device),
        )
        next_output = model(next_batch)
        cur_abstr = next_output.abstraction
        cur_tokens = next_tokens

        global_latents_list.append(cur_abstr.global_latent)
        ego_latents_list.append(cur_abstr.selected_tokens[:, 0, :])
        values_list.append(next_output.policy.value)

        if t < horizon - 1:
            action_next, logp_next = _sample_action(
                next_output.policy.action_mean,
                next_output.policy.action_log_std,
                deterministic,
                sample_clip=action_sample_clip,
            )
            actions_list.append(action_next)
            means_list.append(next_output.policy.action_mean)
            log_stds_list.append(next_output.policy.action_log_std)
            log_probs_list.append(logp_next)

    traj = ImaginedTrajectory(
        actions=torch.stack(actions_list, dim=1),             # [B, K, A]
        action_means=torch.stack(means_list, dim=1),
        action_log_stds=torch.stack(log_stds_list, dim=1),
        log_probs=torch.stack(log_probs_list, dim=1),          # [B, K]
        values=torch.stack(values_list, dim=1),                # [B, K+1]
        rewards=torch.stack(rewards_list, dim=1),              # [B, K]
        collisions=torch.stack(collisions_list, dim=1),
        continues=torch.stack(continues_list, dim=1),
        global_latents=torch.stack(global_latents_list, dim=1),  # [B, K+1, D]
        ego_latents=torch.stack(ego_latents_list, dim=1),         # [B, K+1, D]
        first_output=first_output,
        horizon=horizon,
    )
    return traj
