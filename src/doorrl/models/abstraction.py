from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from doorrl.config import ModelConfig
from doorrl.utils import batched_index_select, masked_mean


@dataclass
class AbstractionOutput:
    selected_tokens: torch.Tensor
    selected_mask: torch.Tensor
    selected_indices: torch.Tensor
    importance: torch.Tensor
    global_latent: torch.Tensor
    # True when the K selected slots are *learned compressed slots* (e.g.
    # holistic_16slot's queries) and ``selected_indices`` is therefore a
    # placeholder that does NOT map slot k to original token position k.
    # Loss/eval code must then use set-prediction (nearest-assignment)
    # matching instead of indexing GT by ``selected_indices``.
    is_set_prediction: bool = False


class DecisionSufficientAbstraction(nn.Module):
    """Top-k attention-pooled abstraction.

    Parameters
    ----------
    config:
        Default model config; ``top_k`` is read from it unless overridden.
    top_k_override:
        If given, this k is used instead of ``config.top_k``. Lets the
        decoupled (Route C) variants instantiate two abstractions with
        per-type budgets (``top_k_dyn`` and ``top_k_rel``) without
        mutating the shared config.
    force_ego:
        Force token-0 (ego) to always be selected. The original DOOR-RL
        abstraction does this so the world model never loses ego context.
        For *typed* abstractions that operate over a strict subset of
        token types (e.g. the relation-only path of the decoupled
        variant) this must be ``False``, otherwise an out-of-set token
        (ego) would always occupy slot 0 and the path's effective budget
        would shrink by one.
    """

    def __init__(
        self,
        config: ModelConfig,
        top_k_override: int | None = None,
        force_ego: bool = True,
    ) -> None:
        super().__init__()
        self.top_k = top_k_override if top_k_override is not None else config.top_k
        self.force_ego = force_ego
        self.query_proj = nn.Linear(config.model_dim, config.model_dim)
        self.key_proj = nn.Linear(config.model_dim, config.model_dim)
        self.score_proj = nn.Linear(config.model_dim, 1)

    def forward(self, latent: torch.Tensor, token_mask: torch.Tensor) -> AbstractionOutput:
        ego = latent[:, :1, :]
        query = self.query_proj(ego)
        keys = self.key_proj(latent)
        similarity = (query * keys).sum(dim=-1) / math.sqrt(latent.size(-1))
        saliency = self.score_proj(latent).squeeze(-1)
        scores = similarity + saliency
        scores = scores.masked_fill(~token_mask, float("-inf"))
        if self.force_ego:
            scores[:, 0] = float("inf")

        k = min(self.top_k, latent.size(1))
        selected_scores, selected_indices = torch.topk(scores, k=k, dim=1)
        selected_tokens = batched_index_select(latent, selected_indices)
        selected_mask = batched_index_select(
            token_mask.unsqueeze(-1).float(), selected_indices
        ).squeeze(-1) > 0.5
        global_latent = masked_mean(latent, token_mask, dim=1)
        importance = torch.softmax(selected_scores.masked_fill(~selected_mask, -1e9), dim=1)
        return AbstractionOutput(
            selected_tokens=selected_tokens,
            selected_mask=selected_mask,
            selected_indices=selected_indices,
            importance=importance,
            global_latent=global_latent,
        )
