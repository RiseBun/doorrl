from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from doorrl.config import ModelConfig
from doorrl.utils import masked_mean


@dataclass
class WorldModelOutput:
    predicted_next_tokens: torch.Tensor
    predicted_reward: torch.Tensor
    predicted_continue: torch.Tensor
    predicted_collision: torch.Tensor
    pooled_latent: torch.Tensor


class ReactiveObjectRelationalWorldModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.action_proj = nn.Linear(config.action_dim, config.model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.next_token_head = nn.Linear(config.model_dim, config.raw_dim)
        self.reward_head = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.collision_head = nn.Sequential(
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        selected_tokens: torch.Tensor,
        selected_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> WorldModelOutput:
        action_token = self.action_proj(actions).unsqueeze(1)
        sequence = torch.cat([action_token, selected_tokens], dim=1)

        action_mask = torch.ones(
            actions.size(0), 1, dtype=torch.bool, device=actions.device
        )
        valid_mask = torch.cat([action_mask, selected_mask], dim=1)
        transformed = self.transformer(
            sequence,
            src_key_padding_mask=~valid_mask,
        )
        token_latent = transformed[:, 1:, :]
        pooled_latent = masked_mean(token_latent, selected_mask, dim=1)

        return WorldModelOutput(
            predicted_next_tokens=self.next_token_head(token_latent),
            predicted_reward=self.reward_head(pooled_latent).squeeze(-1),
            predicted_continue=self.continue_head(pooled_latent).squeeze(-1),
            predicted_collision=self.collision_head(pooled_latent).squeeze(-1),
            pooled_latent=pooled_latent,
        )
