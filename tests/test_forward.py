from __future__ import annotations

import sys
from pathlib import Path

from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data import SyntheticDrivingDataset
from doorrl.models import DoorRLModel
from doorrl.schema import SceneBatch


def test_doorrl_forward_shapes() -> None:
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    dataset = SyntheticDrivingDataset(config=config, size=4, seed=config.seed)
    loader = DataLoader(dataset, batch_size=2, collate_fn=SceneBatch.collate)
    batch = next(iter(loader))

    model = DoorRLModel(config.model)
    output = model(batch)

    assert output.abstraction.selected_tokens.shape[:2] == (2, config.model.top_k)
    assert output.world_model.predicted_next_tokens.shape == (
        2,
        config.model.top_k,
        config.model.raw_dim,
    )
    assert output.policy.action_mean.shape == (2, config.model.action_dim)
