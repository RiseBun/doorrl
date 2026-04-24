"""
Stage 0 (fair): Decision-Oriented Representation Analysis metrics.

Goal
----
把原来的 "Representation Sufficiency" 表从 "信息量比较" 改成真正的
"表示质量比较". 关键改动:

1.  所有 variant 都用 ``top_k`` (默认 16) 个 slot 作 world-model context.
    Holistic variant 也必须走 learned-query bottleneck 压到 16 slot
    (``HOLISTIC_16SLOT``). ``HOLISTIC`` (全 97 token) 只作 upper-bound ref.

2.  评估用 **nearest-assignment (set-prediction 风格)** 统一对齐:
    对每个 GT dynamic agent, 从 predicted_next_tokens 的 K 个 slot 里
    找 (x,y) 距离最近的 slot 作为该 agent 的预测. 这样 holistic_16slot
    (slot 不对应原始 token 位置) 和 top-k variants (slot 对应选中位置)
    被放在同一评估协议下.

3.  指标:
      * Dyn Rollout MSE    (x,y,vx,vy) ↓
      * Action MSE         ↓  (policy head 对 GT teacher action)
      * Collision F1       ↑
      * Rare ADE           ↓  (所有 rare agent 的 nearest-assignment 欧氏距离)
      * Interaction Recall@1m ↑
          (只看 rare 且当前帧距 ego < 20 m 的 agent, nearest-dist < 1 m 计命中)

Reward Error 因 target 几乎恒为 0 从主表移除 (见 losses.py).
Rare Recall@5m 已失区分度 -> 改为 Interaction Recall@1m.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from doorrl.schema import SceneBatch, TokenType
from doorrl.models.doorrl import DoorRLOutput


# Only (x, y, vx, vy) are semantically comparable across token types.
_POS_VEL_DIMS = 4
# Dynamic agent types considered in rollout/ADE.
_DYNAMIC_TYPES = (
    int(TokenType.EGO),
    int(TokenType.VEHICLE),
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
)
# Rare agent types (safety-critical tail).
_RARE_TYPES = (
    int(TokenType.PEDESTRIAN),
    int(TokenType.CYCLIST),
)
# "Interactive" simplification: rare agent is relevant to ego planning iff
# currently within this radius (metres, ego-relative). Avoids needing the
# relation-table TTC linkback which would require reworking the adapter.
_INTERACTIVE_EGO_RADIUS_M = 20.0
# Tight threshold for Recall (replaces the saturated Rare Recall @ 5 m).
_INTERACTION_RECALL_DIST_M = 1.0
# Collision risk label: TTC < this -> positive.
_COLLISION_TTC_THRESHOLD = 3.0
_TTC_FEATURE_INDEX = 8


def _nearest_match(
    pred_xyv: torch.Tensor,  # [K, >=4]
    pred_mask: torch.Tensor,  # [K] bool
    gt_xy: torch.Tensor,  # [N, 2]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """For each of N GT positions, find the nearest valid slot index and
    return (min_dist [N], matched_slot_idx [N]).

    If no valid slot exists returns (inf, 0) so callers can filter.
    """
    if pred_mask.numel() == 0 or pred_mask.sum().item() == 0:
        inf = torch.full(
            (gt_xy.size(0),), float("inf"),
            device=gt_xy.device, dtype=pred_xyv.dtype,
        )
        zero = torch.zeros(gt_xy.size(0), device=gt_xy.device, dtype=torch.long)
        return inf, zero

    valid_idx = torch.nonzero(pred_mask, as_tuple=False).flatten()  # [K_v]
    valid_xy = pred_xyv[valid_idx, :2]  # [K_v, 2]
    # Pairwise distances [N, K_v]
    diff = gt_xy.unsqueeze(1) - valid_xy.unsqueeze(0)
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)
    min_dist, min_local = dist.min(dim=1)
    matched = valid_idx[min_local]
    return min_dist, matched


@dataclass
class Table3Metrics:
    """Fair-Stage-0 Table 3 accumulator.

    All accumulators are weighted *per scalar element* so that per-batch
    means do not dominate multi-batch averages. E.g. ``dyn_rollout_se_sum``
    is the sum of squared errors over every (agent, dim) pair seen so far.
    """
    dyn_rollout_se_sum: float = 0.0
    dyn_rollout_elem_count: int = 0

    action_se_sum: float = 0.0
    action_elem_count: int = 0

    collision_tp: int = 0
    collision_fp: int = 0
    collision_fn: int = 0
    collision_tn: int = 0

    rare_ade_sum: float = 0.0
    rare_ade_count: int = 0

    interaction_total: int = 0
    interaction_hit: int = 0

    # Per-batch rollout means (used only for the "± std" cosmetic column).
    _rollout_batch_means: List[float] = field(default_factory=list)
    _action_batch_means: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------
    def update_batch(self, batch: SceneBatch, output: DoorRLOutput):
        """Accumulate metrics from one batch of model output."""
        self._update_rollout_and_ade(batch, output)
        self._update_action_mse(batch, output)
        self._update_collision_f1(batch, output)

    # ------------------------------------------------------------------
    def _update_rollout_and_ade(
        self, batch: SceneBatch, output: DoorRLOutput,
    ) -> None:
        """Dyn Rollout MSE + Rare ADE + Interaction Recall@1m, all via
        nearest-assignment between GT dynamic agents and predicted slots.

        Only **dynamic-type** predicted slots are considered as match
        candidates. Relation slots are excluded because Fix #2 supervises
        them on (TTC, lane_conflict, priority) only -- their (x, y) outputs
        are *not* trained, so including them in the candidate pool would
        let arbitrary noise become the "nearest" match and inflate
        Dyn Rollout / Rare ADE / collapse Interaction Recall.

        This change is symmetric across variants:
          * Object-only / Holistic / Holistic-full: their selected slots
            are already dynamic-type tokens, so the filter is a no-op.
          * Object+Relation / +Visibility: the filter restricts matching
            to the dynamic subset of selected slots, mirroring how the
            losses now treat them.
          * HOLISTIC_16SLOT (set-prediction): all 16 learned slots are
            globally trained on dynamic agents, so we treat every valid
            slot as a dynamic candidate (consistent with how
            ``_set_prediction_obs_loss`` supervises them).
        """
        pred_next = output.world_model.predicted_next_tokens  # [B, K, D]
        pred_mask_all = output.abstraction.selected_mask.to(torch.bool)  # [B, K]
        sel_indices = output.abstraction.selected_indices  # [B, K]
        is_set_pred = bool(getattr(output.abstraction, "is_set_prediction", False))
        tokens = batch.tokens  # [B, S, D]
        next_tokens = batch.next_tokens  # [B, S, D]
        token_mask = batch.token_mask  # [B, S]
        token_types = batch.token_types  # [B, S]

        batch_size = tokens.size(0)
        batch_se = 0.0
        batch_elems = 0

        for i in range(batch_size):
            tt_i = token_types[i]
            tm_i = token_mask[i]

            dyn_mask = torch.zeros_like(tm_i, dtype=torch.bool)
            for t in _DYNAMIC_TYPES:
                dyn_mask |= (tt_i == t)
            dyn_mask &= tm_i

            if dyn_mask.sum() == 0:
                continue

            gt_xyv = next_tokens[i][dyn_mask][:, :_POS_VEL_DIMS]  # [N, 4]
            cur_xy = tokens[i][dyn_mask][:, :2]  # [N, 2]
            types_i = tt_i[dyn_mask]  # [N]

            pred_xyv_i = pred_next[i][:, :_POS_VEL_DIMS]  # [K, 4]
            pred_mask_i = pred_mask_all[i]  # [K]

            # Restrict candidates to dynamic-type slots (see docstring).
            if is_set_pred:
                slot_is_dyn = pred_mask_i  # all set-pred slots are dyn-trained
            else:
                slot_types = tt_i.gather(0, sel_indices[i])  # [K]
                slot_is_dyn = torch.zeros_like(slot_types, dtype=torch.bool)
                for t in _DYNAMIC_TYPES:
                    slot_is_dyn |= (slot_types == t)
            match_mask = pred_mask_i & slot_is_dyn

            min_dist, matched_idx = _nearest_match(
                pred_xyv_i, match_mask, gt_xyv[:, :2]
            )
            # If no valid slot, skip this sample entirely (should not happen
            # in practice since selected_mask has at least one True).
            if torch.isinf(min_dist).all():
                continue

            matched_pred_xyv = pred_xyv_i[matched_idx]  # [N, 4]
            se = ((matched_pred_xyv - gt_xyv) ** 2).sum().item()
            n_elems = gt_xyv.numel()
            batch_se += se
            batch_elems += n_elems

            # Rare ADE + Interaction Recall@1m
            rare_type_mask = torch.zeros_like(types_i, dtype=torch.bool)
            for t in _RARE_TYPES:
                rare_type_mask |= (types_i == t)
            if rare_type_mask.any():
                rare_dists = min_dist[rare_type_mask]
                rare_cur_xy = cur_xy[rare_type_mask]
                # Rare ADE: all rare agents.
                self.rare_ade_sum += rare_dists.sum().item()
                self.rare_ade_count += int(rare_type_mask.sum().item())
                # Interaction filter: current dist to ego < 20 m.
                cur_ego_dist = torch.sqrt(
                    (rare_cur_xy * rare_cur_xy).sum(dim=1) + 1e-12
                )
                interactive_mask = cur_ego_dist < _INTERACTIVE_EGO_RADIUS_M
                n_inter = int(interactive_mask.sum().item())
                if n_inter > 0:
                    inter_dists = rare_dists[interactive_mask]
                    n_hit = int((inter_dists < _INTERACTION_RECALL_DIST_M).sum().item())
                    self.interaction_total += n_inter
                    self.interaction_hit += n_hit

        self.dyn_rollout_se_sum += batch_se
        self.dyn_rollout_elem_count += batch_elems
        if batch_elems > 0:
            self._rollout_batch_means.append(batch_se / batch_elems)

    # ------------------------------------------------------------------
    def _update_action_mse(
        self, batch: SceneBatch, output: DoorRLOutput,
    ) -> None:
        """Action MSE: validation BC-style error of the policy head."""
        pred = output.policy.action_mean
        gt = batch.actions
        if pred.shape != gt.shape:
            return
        se = ((pred - gt) ** 2).sum().item()
        n = pred.numel()
        self.action_se_sum += se
        self.action_elem_count += n
        if n > 0:
            self._action_batch_means.append(se / n)

    # ------------------------------------------------------------------
    def _update_collision_f1(
        self, batch: SceneBatch, output: DoorRLOutput,
    ) -> None:
        """Collision TP/FP/FN/TN using TTC<3s derived labels."""
        pred_prob = output.world_model.predicted_collision.detach()
        pred_bin = (pred_prob > 0.5).long().cpu().tolist()

        tokens = batch.tokens
        token_types = batch.token_types
        token_mask = batch.token_mask
        batch_size = tokens.size(0)

        for i in range(batch_size):
            rel_mask = (
                (token_types[i] == int(TokenType.RELATION)) & token_mask[i]
            )
            label = 0
            if rel_mask.any():
                ttc = tokens[i, rel_mask, _TTC_FEATURE_INDEX]
                if (ttc < _COLLISION_TTC_THRESHOLD).any().item():
                    label = 1

            p = pred_bin[i]
            if p == 1 and label == 1:
                self.collision_tp += 1
            elif p == 1 and label == 0:
                self.collision_fp += 1
            elif p == 0 and label == 1:
                self.collision_fn += 1
            else:
                self.collision_tn += 1

    # ------------------------------------------------------------------
    def compute_table3(self) -> Dict[str, float]:
        dyn_rollout_mse = (
            self.dyn_rollout_se_sum / self.dyn_rollout_elem_count
            if self.dyn_rollout_elem_count > 0 else 0.0
        )
        dyn_rollout_std = (
            float(np.std(self._rollout_batch_means))
            if len(self._rollout_batch_means) > 1 else 0.0
        )
        action_mse = (
            self.action_se_sum / self.action_elem_count
            if self.action_elem_count > 0 else 0.0
        )
        action_std = (
            float(np.std(self._action_batch_means))
            if len(self._action_batch_means) > 1 else 0.0
        )

        tp, fp, fn, tn = (
            self.collision_tp, self.collision_fp,
            self.collision_fn, self.collision_tn,
        )
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        coll_acc = (
            (tp + tn) / (tp + fp + fn + tn)
            if (tp + fp + fn + tn) > 0 else 0.0
        )
        base_rate_pos = (
            (tp + fn) / (tp + fp + fn + tn)
            if (tp + fp + fn + tn) > 0 else 0.0
        )

        rare_ade = (
            self.rare_ade_sum / self.rare_ade_count
            if self.rare_ade_count > 0 else 0.0
        )
        interaction_recall = (
            self.interaction_hit / self.interaction_total
            if self.interaction_total > 0 else 0.0
        )

        return {
            "dyn_rollout_mse": dyn_rollout_mse,
            "dyn_rollout_std": dyn_rollout_std,
            "action_mse": action_mse,
            "action_mse_std": action_std,
            "collision_f1": f1,
            "collision_precision": precision,
            "collision_recall": recall,
            "collision_accuracy": coll_acc,
            "collision_base_rate_pos": base_rate_pos,
            "rare_ade": rare_ade,
            "rare_ade_count": self.rare_ade_count,
            "interaction_recall_at_1m": interaction_recall,
            "interaction_total": self.interaction_total,
            "interaction_hit": self.interaction_hit,
        }

    def print_table3_row(self, variant_name: str) -> str:
        m = self.compute_table3()
        return (
            f"{variant_name:<25} | "
            f"DynRollout={m['dyn_rollout_mse']:.4f} | "
            f"ActMSE={m['action_mse']:.4f} | "
            f"CollF1={m['collision_f1']:.4f} | "
            f"RareADE={m['rare_ade']:.4f} | "
            f"IntRec@1m={m['interaction_recall_at_1m']:.4f}"
        )


def evaluate_stage0(
    model,
    data_loader,
    variant_name: str,
    device: torch.device,
    verbose: bool = True,
) -> Table3Metrics:
    """Evaluate one variant and return populated ``Table3Metrics``."""
    model.eval()
    metrics = Table3Metrics()

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            metrics.update_batch(batch, output)

    if verbose:
        print(f"\nEvaluating: {variant_name}")
        print(metrics.print_table3_row(variant_name))

    return metrics
