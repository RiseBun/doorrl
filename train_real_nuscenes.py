"""使用真实nuScenes数据训练DOOR-RL"""
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
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models import DoorRLModel
from doorrl.schema import SceneBatch
from doorrl.training import DoorRLTrainer
from doorrl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DOOR-RL training with real nuScenes data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "debug_mvp.json"),
        help="Path to a JSON config file.",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default="/mnt/datasets/e2e-nuscenes/20260302",
        help="Path to nuScenes dataset root.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="List of scene names to use (default: first 10 scenes).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional override for number of training epochs.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=5,
        help="Number of scenes to use if --scenes not specified.",
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

    # 创建真实数据集
    print(f"\nLoading nuScenes data from {args.nuscenes_root}...")
    
    full_dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        scenes=args.scenes,
        version='v1.0-trainval',
    )
    
    # 划分训练集和验证集 (80/20)
    from torch.utils.data import random_split
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Total: {dataset_size} samples")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=SceneBatch.collate,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=2,
    )

    # 创建模型
    model = DoorRLModel(config.model)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    trainer = DoorRLTrainer(model=model, config=config.training, device=device)
    trainer.fit(train_loader, val_loader=val_loader)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
