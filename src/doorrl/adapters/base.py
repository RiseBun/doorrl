from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Sequence

import torch

from doorrl.schema import TokenType


class BenchmarkMode(str, Enum):
    OFFLINE_DATASET = "offline_dataset"
    CLOSED_LOOP_REACTIVE = "closed_loop_reactive"
    CLOSED_LOOP_NON_REACTIVE = "closed_loop_non_reactive"
    EXTERNAL_EVALUATION = "external_evaluation"


@dataclass
class TokenizationSpec:
    raw_dim: int
    max_tokens: int
    max_dynamic_objects: int
    max_map_tokens: int
    max_relation_tokens: int
    action_dim: int = 2


@dataclass
class AdapterDescription:
    name: str
    mode: BenchmarkMode
    purpose: str
    expected_inputs: Sequence[str]
    outputs: Sequence[str]


class NormalizedSceneConverter:
    def __init__(self, spec: TokenizationSpec) -> None:
        self.spec = spec
        self.type_map = {
            "ego": TokenType.EGO,
            "vehicle": TokenType.VEHICLE,
            "pedestrian": TokenType.PEDESTRIAN,
            "cyclist": TokenType.CYCLIST,
            "map": TokenType.MAP,
            "signal": TokenType.SIGNAL,
            "relation": TokenType.RELATION,
        }

    def build_scene_item(self, record: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        raw_dim = self.spec.raw_dim
        max_tokens = self.spec.max_tokens
        tokens = torch.zeros(max_tokens, raw_dim, dtype=torch.float32)
        next_tokens = torch.zeros_like(tokens)
        token_mask = torch.zeros(max_tokens, dtype=torch.bool)
        token_types = torch.full((max_tokens,), int(TokenType.PAD), dtype=torch.long)

        ego = dict(record.get("ego", {}))
        next_ego = dict(record.get("next_ego", ego))
        self._fill_token(tokens, 0, ego, "ego")
        self._fill_token(next_tokens, 0, next_ego, "ego")
        token_mask[0] = True
        token_types[0] = int(TokenType.EGO)

        cursor = 1
        for obj in list(record.get("objects", []))[: self.spec.max_dynamic_objects]:
            if cursor >= max_tokens:
                break
            kind = str(obj.get("token_type", "vehicle"))
            self._fill_token(tokens, cursor, obj, kind)
            self._fill_token(next_tokens, cursor, obj, kind)
            token_mask[cursor] = True
            token_types[cursor] = int(self.type_map.get(kind, TokenType.VEHICLE))
            cursor += 1

        for node in list(record.get("map_elements", []))[: self.spec.max_map_tokens]:
            if cursor >= max_tokens:
                break
            self._fill_token(tokens, cursor, node, "map")
            self._fill_token(next_tokens, cursor, node, "map")
            token_mask[cursor] = True
            token_types[cursor] = int(TokenType.MAP)
            cursor += 1

        for rel in list(record.get("relations", []))[: self.spec.max_relation_tokens]:
            if cursor >= max_tokens:
                break
            self._fill_token(tokens, cursor, rel, "relation")
            self._fill_token(next_tokens, cursor, rel, "relation")
            token_mask[cursor] = True
            token_types[cursor] = int(TokenType.RELATION)
            cursor += 1

        actions = torch.tensor(
            record.get("action", [0.0] * self.spec.action_dim),
            dtype=torch.float32,
        )
        if actions.numel() != self.spec.action_dim:
            raise ValueError(
                f"Expected action_dim={self.spec.action_dim}, got {actions.numel()}"
            )

        reward = torch.tensor(float(record.get("reward", 0.0)), dtype=torch.float32)
        continues = torch.tensor(float(record.get("continue", 1.0)), dtype=torch.float32)

        next_tokens[~token_mask] = 0.0

        # Final NaN/Inf safety net: upstream extractors (especially nuScenes
        # box_velocity / CAN pose) can yield NaN on edge samples. Letting a
        # single NaN slip through here poisons every loss via the forward
        # pass, so sanitize all float tensors defensively.
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0)
        next_tokens = torch.nan_to_num(
            next_tokens, nan=0.0, posinf=0.0, neginf=0.0
        )
        actions = torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        continues = torch.nan_to_num(
            continues, nan=0.0, posinf=0.0, neginf=0.0
        )

        return {
            "tokens": tokens,
            "token_mask": token_mask,
            "token_types": token_types,
            "actions": actions,
            "next_tokens": next_tokens,
            "rewards": reward,
            "continues": continues,
        }

    def _fill_token(
        self,
        target: torch.Tensor,
        index: int,
        source: Mapping[str, Any],
        token_type: str,
    ) -> None:
        target[index, 0] = float(source.get("x", 0.0))
        target[index, 1] = float(source.get("y", 0.0))
        target[index, 2] = float(source.get("vx", source.get("speed", 0.0)))
        target[index, 3] = float(source.get("vy", 0.0))
        target[index, 4] = float(source.get("length", 0.0))
        target[index, 5] = float(source.get("width", 0.0))
        target[index, 6] = float(source.get("risk", 0.0))
        target[index, 7] = float(source.get("visibility", 1.0))
        target[index, 8] = float(source.get("ttc", 0.0))
        target[index, 9] = float(source.get("lane_conflict", 0.0))
        target[index, 10] = float(source.get("priority", 0.0))
        target[index, 11] = float(source.get("distance", 0.0))
        target[index, 12] = float(source.get("heading", 0.0))
        target[index, 13] = float(source.get("is_interactive", 0.0))
        target[index, 14] = float(self.type_map.get(token_type, TokenType.PAD))
