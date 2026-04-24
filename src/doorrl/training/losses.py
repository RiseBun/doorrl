from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from doorrl.config import TrainingConfig
from doorrl.models.doorrl import DoorRLOutput
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import batched_index_select


# Position in the raw token vector that holds TTC (time-to-collision) for
# relation tokens. Must stay in sync with NormalizedSceneConverter._fill_token.
_TTC_FEATURE_INDEX = 8
# Threshold (seconds) matching the Table 3 collision evaluator so training
# and evaluation share the same positive-label definition.
_COLLISION_TTC_THRESHOLD_SEC = 3.0
# Token types whose next-state (x,y,vx,vy) we supervise under set-prediction
# matching; map / signal / relation tokens are excluded because their
# "next state" has no per-frame physical meaning.
_DYNAMIC_TYPES = (
    int(TokenType.EGO),
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
)
# Only (x, y, vx, vy) are semantically comparable across token types and
# match the Table 3 evaluator's metric definition.
_POS_VEL_DIMS = 4
# Indices of the per-relation semantic features supervised by Fix #2:
#   8  -> ttc, 9 -> lane_conflict, 10 -> priority
# These are the only dims of the raw token vector whose "next-frame" value
# carries decision-relevant information for RELATION tokens (their
# next-frame x/y/vx/vy are not physical positions of any object and
# regressing them couples relation slots to a meaningless target).
_RELATION_FEATURE_DIMS = (8, 9, 10)
# Relative weight of the relation-feature MSE inside the combined obs loss.
# 1.0 means each ground-truth element (a single dim of one token) contributes
# equally regardless of token type.
_RELATION_OBS_WEIGHT = 1.0


def _set_prediction_obs_loss(
    batch: SceneBatch, output: DoorRLOutput,
) -> torch.Tensor:
    """Set-prediction (DETR-style) observation loss.

    Used by variants whose K predicted slots are *learned compressed slots*
    (e.g. ``HOLISTIC_16SLOT``) and therefore do not correspond 1:1 to
    original token positions. The previous code path indexed
    ``batch.next_tokens`` by the variant's placeholder ``selected_indices``
    (which are all 0 -> ego), causing all 16 slots to be supervised toward
    ego's next state - a trivial target since ego is at (0,0) in
    ego-relative coordinates. The model would drive ``train_obs`` to ~0
    while learning nothing about other agents, breaking the fair
    comparison with top-k variants.

    Fix: for each ground-truth dynamic agent, take the nearest predicted
    slot by (x, y) distance (matching is non-differentiable; argmin is
    detached) and apply MSE on the matched pairs over (x, y, vx, vy).
    Mirrors the evaluator in ``table3_metrics.py`` so training and
    evaluation share the same protocol.
    """
    pred_next = output.world_model.predicted_next_tokens  # [B, K, raw_dim]
    pred_mask = output.abstraction.selected_mask.to(torch.bool)  # [B, K]
    next_tokens = batch.next_tokens  # [B, S, raw_dim]
    token_mask = batch.token_mask  # [B, S]
    token_types = batch.token_types  # [B, S]

    dyn_gt_mask = torch.zeros_like(token_mask, dtype=torch.bool)
    for t in _DYNAMIC_TYPES:
        dyn_gt_mask |= (token_types == t)
    dyn_gt_mask &= token_mask

    batch_size = pred_next.size(0)
    total_se = pred_next.new_zeros(())
    total_count = 0

    for i in range(batch_size):
        gt_idx = torch.nonzero(dyn_gt_mask[i], as_tuple=False).flatten()
        if gt_idx.numel() == 0:
            continue
        gt_xyv = next_tokens[i, gt_idx, :_POS_VEL_DIMS]  # [N_i, 4]

        valid_pred_idx = torch.nonzero(pred_mask[i], as_tuple=False).flatten()
        if valid_pred_idx.numel() == 0:
            continue
        pred_xyv_valid = pred_next[i, valid_pred_idx, :_POS_VEL_DIMS]  # [K_v, 4]

        # Non-differentiable nearest-neighbour matching by (x, y).
        with torch.no_grad():
            diff = (
                gt_xyv[:, :2].unsqueeze(1) - pred_xyv_valid[:, :2].unsqueeze(0)
            )  # [N_i, K_v, 2]
            dist_sq = (diff * diff).sum(dim=-1)
            min_local = dist_sq.argmin(dim=1)  # [N_i]

        matched_pred = pred_xyv_valid[min_local]  # [N_i, 4]
        se = ((matched_pred - gt_xyv) ** 2).sum()
        total_se = total_se + se
        total_count += int(gt_xyv.numel())

    if total_count == 0:
        return pred_next.new_zeros(())
    return total_se / float(total_count)


def _typed_obs_loss(
    batch: SceneBatch, output: DoorRLOutput,
) -> Tuple[torch.Tensor, float, float]:
    """Type-aware observation loss for top-k variants (Fix #2).

    Replaces the previous "MSE over all 40 raw dims, equally for every
    selected token type". That formulation forced every selected slot,
    including ``RELATION`` slots, to regress the next-frame value of dims
    such as (x, y, vx, vy, length, width, ...). For relation tokens those
    dims are *not* physical agent positions; supervising them dragged
    relation-token slots toward a target that has no consistent meaning,
    competed with object slots inside the 16-slot bottleneck, and
    explained the empirical observation that ``object_relation`` lost to
    ``object_only`` under tight context budgets.

    The new policy disentangles supervision by token type:

      * Dynamic slots (EGO / VEHICLE / PEDESTRIAN / CYCLIST):
          MSE on (x, y, vx, vy) only -- exactly matching the evaluator.
      * Relation slots (RELATION):
          MSE on (TTC, lane_conflict, priority) only -- the semantic
          edge attributes a relation token is meant to predict.
      * Map / Signal / PAD:
          No contribution. Their "next state" is essentially constant
          and supervising it is noise.

    Returns the combined loss plus the two per-element-mean components
    (for diagnostics in the training log).
    """
    pred_next = output.world_model.predicted_next_tokens  # [B, K, raw_dim]
    sel_idx = output.abstraction.selected_indices  # [B, K]
    sel_mask = output.abstraction.selected_mask  # [B, K]

    target_next = batched_index_select(batch.next_tokens, sel_idx)  # [B, K, raw_dim]
    sel_types = torch.gather(batch.token_types, 1, sel_idx)  # [B, K]

    dyn_mask = torch.zeros_like(sel_types, dtype=torch.bool)
    for t in _DYNAMIC_TYPES:
        dyn_mask |= (sel_types == t)
    dyn_mask &= sel_mask.bool()

    rel_mask = (sel_types == int(TokenType.RELATION)) & sel_mask.bool()

    pred_dyn = pred_next[..., :_POS_VEL_DIMS]
    tgt_dyn = target_next[..., :_POS_VEL_DIMS]
    dyn_se_per_slot = ((pred_dyn - tgt_dyn) ** 2).sum(dim=-1)  # [B, K]
    dyn_count = dyn_mask.float().sum().clamp_min(1.0)
    dyn_loss = (dyn_se_per_slot * dyn_mask.float()).sum() / (
        dyn_count * float(_POS_VEL_DIMS)
    )

    rel_dims = list(_RELATION_FEATURE_DIMS)
    pred_rel = pred_next[..., rel_dims]
    tgt_rel = target_next[..., rel_dims]
    rel_se_per_slot = ((pred_rel - tgt_rel) ** 2).sum(dim=-1)  # [B, K]
    rel_count = rel_mask.float().sum().clamp_min(1.0)
    rel_loss = (rel_se_per_slot * rel_mask.float()).sum() / (
        rel_count * float(len(rel_dims))
    )
    if int(rel_mask.sum().item()) == 0:
        rel_loss = pred_next.new_zeros(())

    total = dyn_loss + _RELATION_OBS_WEIGHT * rel_loss
    return total, float(dyn_loss.detach().item()), float(rel_loss.detach().item())


def _derive_collision_targets(batch: SceneBatch) -> torch.Tensor:
    """Per-sample binary collision label = 1 iff any valid relation token has
    TTC < threshold. Matches the Table 3 evaluator definition.

    Previously this was implicitly ``1 - batch.continues``, but ``continues``
    defaults to 1.0 everywhere in the nuScenes adapter, making the target a
    constant 0 and effectively disabling collision-head training.
    """
    relation_mask = (
        (batch.token_types == int(TokenType.RELATION)) & batch.token_mask
    )
    ttc = batch.tokens[..., _TTC_FEATURE_INDEX]
    # Fill invalid (non-relation / padded) slots with +inf so they cannot
    # trigger a positive label regardless of their stored payload.
    sentinel = torch.full_like(ttc, float('inf'))
    ttc_valid = torch.where(relation_mask, ttc, sentinel)
    return (ttc_valid < _COLLISION_TTC_THRESHOLD_SEC).any(dim=-1).float()


def compute_losses(
    batch: SceneBatch,
    output: DoorRLOutput,
    config: TrainingConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    obs_dyn_val = 0.0
    obs_rel_val = 0.0
    if getattr(output.abstraction, "is_set_prediction", False):
        # Variants whose K slots are learned compressed slots, not 1:1
        # to original tokens. Must use nearest-assignment matching to
        # avoid the trivial-target failure mode (see _set_prediction_obs_loss
        # docstring) and stay aligned with the Table 3 evaluator.
        obs_loss = _set_prediction_obs_loss(batch, output)
        obs_dyn_val = float(obs_loss.detach().item())
    else:
        # Fix #2: type-aware obs loss. Dynamic slots regress (x,y,vx,vy);
        # relation slots regress (TTC, lane_conflict, priority); map /
        # signal slots are not supervised. See _typed_obs_loss for the
        # rationale.
        obs_loss, obs_dyn_val, obs_rel_val = _typed_obs_loss(batch, output)

    if torch.isnan(obs_loss).any() or torch.isinf(obs_loss).any():
        print(f"Warning: obs_loss is NaN/Inf, setting to 0")
        obs_loss = torch.tensor(
            0.0,
            device=output.world_model.predicted_next_tokens.device,
            requires_grad=True,
        )
    
    # Reward loss - 仅在权重>0时计算
    if config.reward_weight > 0:
        reward_loss = F.mse_loss(output.world_model.predicted_reward, batch.rewards)
    else:
        reward_loss = torch.tensor(0.0, device=batch.rewards.device)
    
    # Continue loss - 仅在权重>0时计算
    if config.continue_weight > 0:
        continue_logits = output.world_model.predicted_continue.clamp(-10, 10)
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits,
            batch.continues,
        )
    else:
        continue_loss = torch.tensor(0.0, device=batch.continues.device)
    
    # Collision loss - 仅在权重>0时计算
    if config.collision_weight > 0:
        collision_logits = output.world_model.predicted_collision.clamp(-10, 10)
        collision_targets = _derive_collision_targets(batch)
        collision_loss = F.binary_cross_entropy_with_logits(
            collision_logits,
            collision_targets,
        )
    else:
        collision_loss = torch.tensor(0.0, device=batch.continues.device)
    
    # BC loss - 仅在权重>0时计算
    if config.bc_weight > 0:
        bc_loss = F.mse_loss(output.policy.action_mean, batch.actions)
    else:
        bc_loss = torch.tensor(0.0, device=batch.actions.device)
    
    # 检查所有loss
    for name, loss_val in [("obs", obs_loss), ("reward", reward_loss), 
                            ("continue", continue_loss), ("collision", collision_loss),
                            ("bc", bc_loss)]:
        if torch.isnan(loss_val).any() or torch.isinf(loss_val).any():
            print(f"Warning: {name}_loss is NaN/Inf, setting to 0")

    total = (
        config.obs_weight * obs_loss
        + config.reward_weight * reward_loss
        + config.continue_weight * continue_loss
        + config.collision_weight * collision_loss
        + config.bc_weight * bc_loss
    )
    
    # 最终检查
    if torch.isnan(total).any() or torch.isinf(total).any():
        print(f"Warning: Total loss is NaN/Inf! Clipping...")
        total = torch.tensor(10.0, device=batch.tokens.device, requires_grad=True)
    
    stats = {
        "total": float(total.detach().item()),
        "obs": float(obs_loss.detach().item()),
        "obs_dyn": obs_dyn_val,
        "obs_rel": obs_rel_val,
        "reward": float(reward_loss.detach().item()),
        "continue": float(continue_loss.detach().item()),
        "collision": float(collision_loss.detach().item()),
        "bc": float(bc_loss.detach().item()),
    }
    return total, stats
