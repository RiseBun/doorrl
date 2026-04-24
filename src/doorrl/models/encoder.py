from __future__ import annotations

import torch
from torch import nn

from doorrl.config import ModelConfig


class TokenEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(config.raw_dim, config.model_dim)
        self.type_embed = nn.Embedding(config.num_token_types, config.model_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.model_dim),
            nn.Linear(config.model_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.model_dim),
        )
        self.out_norm = nn.LayerNorm(config.model_dim)

    def forward(self, tokens: torch.Tensor, token_types: torch.Tensor) -> torch.Tensor:
        latent = self.input_proj(tokens) + self.type_embed(token_types)
        latent = latent + self.mlp(latent)
        return self.out_norm(latent)
