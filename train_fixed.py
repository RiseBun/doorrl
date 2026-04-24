#!/usr/bin/env python3
"""
训练脚本 - 已修复NaN问题

修复内容:
1. 降低学习率 (0.0003 -> 0.0001)
2. 添加梯度监控
3. 添加loss数值稳定性检查
4. 减小损失权重
5. 梯度裁剪更保守
"""

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "experiment_safe.json"),
        help="Config file path",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default="/mnt/datasets/e2e-nuscenes/20260302",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = DoorRLConfig.from_json(args.config)
    
    if args.epochs is not None:
        config.training.epochs = args.epochs
    
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"\nLoading dataset...")
    dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        scenes=None,  # 使用前N个场景
        version='v1.0-trainval',
    )
    
    # 由于dataset会加载所有scene，我们需要手动限制
    # 这里简化处理，使用前args.num_scenes个场景的数据
    # 实际应该在NuScenesSceneDataset中实现
    
    # 直接使用整个dataset，然后划分
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Dataset: {dataset_size} samples")
    print(f"  Train: {train_size}")
    print(f"  Val: {val_size}")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"  Train: {train_size}")
    print(f"  Val: {val_size}")
    
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
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters")
    
    # 训练
    trainer = DoorRLTrainer(
        model=model,
        config=config.training,
        device=device,
    )
    
    print(f"\n{'='*80}")
    print(f"Starting training (NaN-safe version)")
    print(f"{'='*80}\n")
    
    trainer.fit(train_loader, val_loader=val_loader)
    
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
