from __future__ import annotations

import math
from typing import Dict

import torch
from torch.utils.data import Dataset

from doorrl.config import DoorRLConfig
from doorrl.schema import TokenType


class SyntheticDrivingDataset(Dataset):
    def __init__(self, config: DoorRLConfig, size: int, seed: int) -> None:
        self.config = config
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        del index
        model_cfg = self.config.model
        data_cfg = self.config.data
        raw_dim = model_cfg.raw_dim
        max_tokens = model_cfg.max_tokens

        tokens = torch.zeros(max_tokens, raw_dim)
        next_tokens = torch.zeros_like(tokens)
        token_mask = torch.zeros(max_tokens, dtype=torch.bool)
        token_types = torch.full((max_tokens,), int(TokenType.PAD), dtype=torch.long)

        action = torch.empty(model_cfg.action_dim).uniform_(
            -1.0, 1.0, generator=self.generator
        )

        ego_speed = torch.empty(1).uniform_(3.0, 8.0, generator=self.generator).item()
        tokens[0, 0] = 0.0
        tokens[0, 1] = 0.0
        tokens[0, 2] = ego_speed
        tokens[0, 3] = 0.0
        tokens[0, 4] = 4.5
        tokens[0, 5] = 1.8
        tokens[0, 6] = 0.0
        tokens[0, 7] = 1.0
        token_mask[0] = True
        token_types[0] = int(TokenType.EGO)

        cursor = 1
        num_dynamic = int(
            torch.randint(
                low=4,
                high=data_cfg.max_dynamic_objects + 1,
                size=(1,),
                generator=self.generator,
            ).item()
        )
        min_distance = 1e9
        interaction_risk = 0.0

        for _ in range(num_dynamic):
            if cursor >= max_tokens:
                break
            same_lane = bool(torch.rand(1, generator=self.generator).item() > 0.35)
            dx = torch.empty(1).uniform_(4.0, 30.0, generator=self.generator).item()
            dy = (
                torch.empty(1).uniform_(-0.8, 0.8, generator=self.generator).item()
                if same_lane
                else torch.empty(1).uniform_(-3.5, 3.5, generator=self.generator).item()
            )
            vx = torch.empty(1).uniform_(1.0, 8.0, generator=self.generator).item()
            vy = torch.empty(1).uniform_(-0.4, 0.4, generator=self.generator).item()
            visibility = 1.0 if torch.rand(1, generator=self.generator).item() > 0.2 else 0.4
            rel_speed = ego_speed - vx
            distance = math.sqrt(dx * dx + dy * dy)
            ttc = dx / max(rel_speed, 0.5) if rel_speed > 0.0 else 20.0
            risk = max(0.0, 1.0 / max(distance, 1.0))

            tokens[cursor, 0] = dx
            tokens[cursor, 1] = dy
            tokens[cursor, 2] = vx
            tokens[cursor, 3] = vy
            tokens[cursor, 4] = 4.2
            tokens[cursor, 5] = 1.8
            tokens[cursor, 6] = risk
            tokens[cursor, 7] = visibility
            tokens[cursor, 8] = ttc / 20.0
            token_mask[cursor] = True
            token_types[cursor] = int(TokenType.VEHICLE)

            ego_dx = ego_speed + 0.5 * action[0].item()
            response = -0.35 * action[0].item() if same_lane and dx < 12.0 else 0.0
            next_tokens[cursor] = tokens[cursor]
            next_tokens[cursor, 0] = dx + (vx + response) - ego_dx
            next_tokens[cursor, 2] = vx + response

            min_distance = min(min_distance, distance)
            interaction_risk = max(interaction_risk, risk)
            cursor += 1

        num_map = min(data_cfg.max_map_tokens, max_tokens - cursor - data_cfg.max_relation_tokens)
        for _ in range(num_map):
            if cursor >= max_tokens:
                break
            lane_x = torch.empty(1).uniform_(0.0, 40.0, generator=self.generator).item()
            lane_y = torch.empty(1).uniform_(-4.0, 4.0, generator=self.generator).item()
            tokens[cursor, 0] = lane_x
            tokens[cursor, 1] = lane_y
            tokens[cursor, 4] = 1.0
            tokens[cursor, 7] = 1.0
            token_mask[cursor] = True
            token_types[cursor] = int(TokenType.MAP)
            next_tokens[cursor] = tokens[cursor]
            cursor += 1

        relation_slots = min(data_cfg.max_relation_tokens, num_dynamic, max_tokens - cursor)
        for rel_idx in range(relation_slots):
            source_index = 1 + rel_idx
            dx = tokens[source_index, 0].item()
            dy = tokens[source_index, 1].item()
            distance = math.sqrt(dx * dx + dy * dy)
            rel_speed = tokens[0, 2].item() - tokens[source_index, 2].item()
            tokens[cursor, 0] = dx
            tokens[cursor, 1] = dy
            tokens[cursor, 2] = rel_speed
            tokens[cursor, 3] = distance
            tokens[cursor, 6] = 1.0 / max(distance, 1.0)
            tokens[cursor, 7] = tokens[source_index, 7]
            tokens[cursor, 8] = 1.0 if abs(dy) < 1.0 else 0.0
            token_mask[cursor] = True
            token_types[cursor] = int(TokenType.RELATION)
            next_tokens[cursor] = tokens[cursor]
            cursor += 1

        next_tokens[0] = tokens[0]
        next_tokens[0, 0] = tokens[0, 0] + ego_speed + 0.5 * action[0]
        next_tokens[0, 1] = tokens[0, 1] + 0.2 * action[1]
        next_tokens[0, 2] = tokens[0, 2] + 0.5 * action[0]
        noise = torch.randn(next_tokens.shape, generator=self.generator)
        next_tokens += self.config.data.noise_std * noise
        next_tokens[~token_mask] = 0.0

        collision = 1.0 if min_distance < 2.0 else 0.0
        progress_reward = next_tokens[0, 0].item() / 10.0
        reward = progress_reward - 2.0 * collision - 0.5 * interaction_risk
        cont = 0.0 if collision > 0.0 else 1.0

        return {
            "tokens": tokens.float(),
            "token_mask": token_mask,
            "token_types": token_types,
            "actions": action.float(),
            "next_tokens": next_tokens.float(),
            "rewards": torch.tensor(reward, dtype=torch.float32),
            "continues": torch.tensor(cont, dtype=torch.float32),
        }
