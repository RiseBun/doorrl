from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data import SyntheticDrivingDataset
from doorrl.models import DoorRLModel
from doorrl.schema import SceneBatch
from doorrl.training import DoorRLTrainer
from doorrl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a debug DOOR-RL training job.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "debug_mvp.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DoorRLConfig.from_json(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = SyntheticDrivingDataset(
        config=config,
        size=config.data.train_size,
        seed=config.seed,
    )
    val_dataset = SyntheticDrivingDataset(
        config=config,
        size=config.data.val_size,
        seed=config.seed + 1,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=SceneBatch.collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
    )

    model = DoorRLModel(config.model)
    trainer = DoorRLTrainer(model=model, config=config.training, device=device)
    trainer.fit(train_loader, val_loader=val_loader)


if __name__ == "__main__":
    main()
