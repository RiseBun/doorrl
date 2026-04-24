from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Iterable, List

import torch


class TokenType(IntEnum):
    EGO = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    MAP = 4
    SIGNAL = 5
    RELATION = 6
    PAD = 7


@dataclass
class SceneBatch:
    tokens: torch.Tensor
    token_mask: torch.Tensor
    token_types: torch.Tensor
    actions: torch.Tensor
    next_tokens: torch.Tensor
    rewards: torch.Tensor
    continues: torch.Tensor

    def validate(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [B, S, D]")
        if self.token_mask.shape != self.tokens.shape[:2]:
            raise ValueError("token_mask must have shape [B, S]")
        if self.token_types.shape != self.tokens.shape[:2]:
            raise ValueError("token_types must have shape [B, S]")
        if self.next_tokens.shape != self.tokens.shape:
            raise ValueError("next_tokens must match tokens shape")
        if self.actions.ndim != 2 or self.actions.shape[0] != self.tokens.shape[0]:
            raise ValueError("actions must have shape [B, A]")
        if self.rewards.shape[0] != self.tokens.shape[0]:
            raise ValueError("rewards must have shape [B]")
        if self.continues.shape[0] != self.tokens.shape[0]:
            raise ValueError("continues must have shape [B]")

    def to(self, device: torch.device | str, non_blocking: bool = True) -> "SceneBatch":
        kw = {"non_blocking": non_blocking}
        return SceneBatch(
            tokens=self.tokens.to(device, **kw),
            token_mask=self.token_mask.to(device, **kw),
            token_types=self.token_types.to(device, **kw),
            actions=self.actions.to(device, **kw),
            next_tokens=self.next_tokens.to(device, **kw),
            rewards=self.rewards.to(device, **kw),
            continues=self.continues.to(device, **kw),
        )

    @staticmethod
    def collate(items: Iterable[Dict[str, torch.Tensor]]) -> "SceneBatch":
        batch: Dict[str, List[torch.Tensor]] = {
            "tokens": [],
            "token_mask": [],
            "token_types": [],
            "actions": [],
            "next_tokens": [],
            "rewards": [],
            "continues": [],
        }
        for item in items:
            for key in batch:
                batch[key].append(item[key])
        collated = SceneBatch(
            tokens=torch.stack(batch["tokens"], dim=0),
            token_mask=torch.stack(batch["token_mask"], dim=0),
            token_types=torch.stack(batch["token_types"], dim=0),
            actions=torch.stack(batch["actions"], dim=0),
            next_tokens=torch.stack(batch["next_tokens"], dim=0),
            rewards=torch.stack(batch["rewards"], dim=0),
            continues=torch.stack(batch["continues"], dim=0),
        )
        collated.validate()
        return collated
