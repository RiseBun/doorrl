"""消融实验训练脚本 - 对比不同表示方式"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant, create_model_variant
from doorrl.schema import SceneBatch
from doorrl.training import DoorRLTrainer
from doorrl.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation study for DOOR-RL representations.")
    
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
        "--variant",
        type=str,
        choices=["holistic", "object_only", "object_relation", "object_relation_visibility"],
        default="object_relation",
        help="Model variant to train.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=20,
        help="Number of scenes to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "experiments" / "ablation"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    
    return parser.parse_args()


def run_ablation_experiment(args: argparse.Namespace) -> None:
    """运行单个消融实验"""
    
    # 1. 加载配置
    config = DoorRLConfig.from_json(args.config)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.seed = args.seed
    
    # 2. 设置随机种子
    set_seed(config.seed)
    
    # 3. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DOOR-RL Ablation Study: {args.variant}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Scenes: {args.num-scenes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # 4. 创建数据集
    print(f"\nLoading nuScenes data from {args.nuscenes_root}...")
    
    full_dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        scenes=None,  # 使用默认场景
        version='v1.0-trainval',
    )
    
    # 限制场景数量
    if len(full_dataset) > args.num-scenes * 40:  # 假设每场景约40帧
        subset_indices = list(range(args.num-scenes * 40))
        full_dataset = torch.utils.data.Subset(full_dataset, subset_indices)
    
    # 划分训练集和验证集 (80/20)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=SceneBatch.collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=4,
        pin_memory=True,
    )
    
    # 5. 创建模型变体
    variant = ModelVariant(args.variant)
    model = create_model_variant(config.model, variant)
    
    # 6. 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{args.variant}_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = exp_dir / "config.json"
    config_dict = {
        "variant": args.variant,
        "num_scenes": args.num-scenes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
    }
    config_path.write_text(json.dumps(config_dict, indent=2))
    print(f"\nExperiment directory: {exp_dir}")
    
    # 7. 训练
    trainer = DoorRLTrainer(
        model=model,
        config=config.training,
        device=device,
    )
    
    print(f"\nStarting training...")
    print(f"{'='*60}")
    
    # 记录训练历史
    train_history = []
    val_history = []
    
    # 修改trainer以记录历史
    original_run_epoch = trainer.run_epoch
    
    def run_epoch_with_logging(loader, training):
        stats = original_run_epoch(loader, training)
        history = train_history if training else val_history
        history.append(stats)
        return stats
    
    trainer.run_epoch = run_epoch_with_logging
    trainer.fit(train_loader, val_loader=val_loader)
    
    # 8. 保存结果
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    
    # 保存训练历史
    history_path = exp_dir / "history.json"
    history_data = {
        "train": train_history,
        "val": val_history,
    }
    history_path.write_text(json.dumps(history_data, indent=2))
    
    # 保存模型权重
    model_path = exp_dir / "model.pt"
    torch.save({
        'variant': args.variant,
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'final_train_stats': train_history[-1] if train_history else None,
        'final_val_stats': val_history[-1] if val_history else None,
    }, model_path)
    
    print(f"Results saved to: {exp_dir}")
    print(f"  - config.json")
    print(f"  - history.json")
    print(f"  - model.pt")
    
    # 打印最终结果
    if val_history:
        final_val = val_history[-1]
        print(f"\nFinal validation metrics:")
        print(f"  Total loss:    {final_val['total']:.4f}")
        print(f"  Obs loss:      {final_val['obs']:.4f}")
        print(f"  Reward loss:   {final_val['reward']:.4f}")
        print(f"  Collision loss: {final_val['collision']:.4f}")


def main() -> None:
    args = parse_args()
    run_ablation_experiment(args)


if __name__ == "__main__":
    main()
