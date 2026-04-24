from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from doorrl.config import ModelConfig
from doorrl.models.abstraction import AbstractionOutput, DecisionSufficientAbstraction
from doorrl.models.encoder import TokenEncoder
from doorrl.models.policy import ActorCriticHead, PolicyOutput
from doorrl.models.world_model import ReactiveObjectRelationalWorldModel, WorldModelOutput
from doorrl.schema import SceneBatch


@dataclass
class DoorRLOutput:
    abstraction: AbstractionOutput
    world_model: WorldModelOutput
    policy: PolicyOutput


class DoorRLModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.encoder = TokenEncoder(config)
        self.abstraction = DecisionSufficientAbstraction(config)
        self.world_model = ReactiveObjectRelationalWorldModel(config)
        self.policy = ActorCriticHead(config)

    def forward(self, batch: SceneBatch) -> DoorRLOutput:
        latent = self.encoder(batch.tokens, batch.token_types)
        abstraction = self.abstraction(latent, batch.token_mask)
        world_model = self.world_model(
            selected_tokens=abstraction.selected_tokens,
            selected_mask=abstraction.selected_mask,
            actions=batch.actions,
        )
        policy = self.policy(abstraction.global_latent)
        return DoorRLOutput(
            abstraction=abstraction,
            world_model=world_model,
            policy=policy,
        )
