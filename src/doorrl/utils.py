from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.float()
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weighted = values * weights
    denom = weights.sum(dim=dim).clamp_min(1.0)
    return weighted.sum(dim=dim) / denom


def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must have shape [B, S, D]")
    gather_index = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
    return values.gather(1, gather_index)
