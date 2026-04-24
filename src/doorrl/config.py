from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    raw_dim: int = 40
    model_dim: int = 128
    hidden_dim: int = 256
    action_dim: int = 2
    max_tokens: int = 97
    num_token_types: int = 8
    top_k: int = 16
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    # Decoupled (Route C) abstraction budgets. Used by the
    # OBJECT_RELATION_DECOUPLED[_VISIBILITY] variants. The two budgets are
    # *typed* slots, not hand-picked numbers: the design point is that
    # dynamic agents and relation context get separate, non-competing
    # representational capacity. They are sized so that
    #   top_k_dyn + top_k_rel == top_k
    # to keep the world-model context budget identical to the other
    # 16-slot variants in the fair Stage 0 comparison.
    top_k_dyn: int = 12
    top_k_rel: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    obs_weight: float = 1.0
    reward_weight: float = 0.5
    continue_weight: float = 0.25
    collision_weight: float = 0.25
    bc_weight: float = 0.1
    log_every: int = 10


@dataclass
class DataConfig:
    train_size: int = 64
    val_size: int = 16
    max_dynamic_objects: int = 12
    max_map_tokens: int = 32
    max_relation_tokens: int = 12
    noise_std: float = 0.05


@dataclass
class BenchmarkConfig:
    offline_dataset: str = "nuscenes"
    closed_loop_benchmark: str = "nuplan"
    external_evaluation: str = "navsim"
    nuplan_mode: str = "closed_loop_reactive"
    nuscenes_root: str = ""
    nuplan_root: str = ""
    navsim_root: str = ""


@dataclass
class DoorRLConfig:
    seed: int = 7
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "DoorRLConfig":
        payload = json.loads(Path(path).read_text())
        return cls(
            seed=payload.get("seed", 7),
            model=ModelConfig(**payload.get("model", {})),
            training=TrainingConfig(**payload.get("training", {})),
            data=DataConfig(**payload.get("data", {})),
            benchmark=BenchmarkConfig(**payload.get("benchmark", {})),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
