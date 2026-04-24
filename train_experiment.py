"""增强版训练脚本 - 带日志、检查点和实验管理"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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
    parser = argparse.ArgumentParser(description="Run DOOR-RL experiment with real nuScenes data.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "experiment_baseline.json"),
        help="Path to experiment config file.",
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
        help="List of scene names to use.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=10,
        help="Number of scenes to use if --scenes not specified.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/runs",
        help="Output directory for experiment artifacts.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from.",
    )
    return parser.parse_args()


def create_experiment_dir(output_dir: str, config_path: str) -> Path:
    """创建实验输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = Path(config_path).stem
    exp_name = f"{config_name}_{timestamp}"
    
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    (exp_dir / "checkpoints").mkdir()
    (exp_dir / "logs").mkdir()
    
    return exp_dir


def save_experiment_config(exp_dir: Path, args: argparse.Namespace, config: DoorRLConfig) -> None:
    """保存实验配置"""
    # 保存命令行参数
    args_dict = vars(args)
    args_dict['timestamp'] = datetime.now().isoformat()
    
    with open(exp_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # 保存完整配置
    with open(exp_dir / "config.json", "w") as f:
        json.dump({
            "seed": config.seed,
            "model": {k: v for k, v in vars(config.model).items()},
            "training": {k: v for k, v in vars(config.training).items()},
            "data": {k: v for k, v in vars(config.data).items()},
        }, f, indent=2)
    
    print(f"Experiment config saved to: {exp_dir}")


class EnhancedDoorRLTrainer(DoorRLTrainer):
    """增强版训练器 - 支持检查点保存"""
    
    def __init__(self, model, config, device, exp_dir: Path = None):
        super().__init__(model, config, device)
        self.exp_dir = exp_dir
        self.best_val_loss = float('inf')
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        if self.exp_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        # 保存最新检查点
        latest_path = self.exp_dir / "checkpoints" / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.exp_dir / "checkpoints" / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best checkpoint (val_loss={self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def fit(self, train_loader, val_loader=None) -> None:
        """增强版训练循环"""
        log_file = None
        if self.exp_dir:
            log_file = open(self.exp_dir / "logs" / "training.log", "w")
        
        for epoch in range(self.config.epochs):
            train_stats = self.run_epoch(train_loader, training=True)
            
            message = (
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train: total={train_stats['total']:.4f} obs={train_stats['obs']:.4f}"
            )
            
            is_best = False
            if val_loader is not None:
                val_stats = self.run_epoch(val_loader, training=False)
                message += f" | Val: total={val_stats['total']:.4f} obs={val_stats['obs']:.4f}"
                
                # 检查是否最佳
                if val_stats['total'] < self.best_val_loss:
                    self.best_val_loss = val_stats['total']
                    is_best = True
            
            print(message)
            
            # 写入日志文件
            if log_file:
                log_file.write(message + "\n")
                log_file.flush()
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0 or (epoch + 1) == self.config.epochs:
                self.save_checkpoint(epoch + 1, is_best)
        
        if log_file:
            log_file.close()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if self.exp_dir:
            print(f"Experiment artifacts saved to: {self.exp_dir}")


def main() -> None:
    args = parse_args()
    
    # 加载配置
    config = DoorRLConfig.from_json(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    
    # 创建实验目录
    exp_dir = create_experiment_dir(args.output_dir, args.config)
    save_experiment_config(exp_dir, args, config)
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    
    # 加载数据集
    print(f"\nLoading nuScenes data from {args.nuscenes_root}...")
    
    # 确定使用的场景
    scenes = args.scenes
    if scenes is None:
        print(f"Using first {args.num_scenes} scenes")
    else:
        print(f"Using specified scenes: {scenes}")
    
    # 创建数据集
    full_dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        scenes=scenes,
        version='v1.0-trainval',
    )
    
    # 简单划分训练集/验证集 (80/20)
    dataset_size = len(full_dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
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
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=2,
        pin_memory=True,
    )
    
    # 创建模型
    model = DoorRLModel(config.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    print(f"Model size: {num_params * 4 / 1e6:.1f} MB (float32)")
    
    # 创建训练器
    trainer = EnhancedDoorRLTrainer(
        model=model,
        config=config.training,
        device=device,
        exp_dir=exp_dir,
    )
    
    # 可选：从检查点恢复
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # 开始训练
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")
    
    trainer.fit(train_loader, val_loader=val_loader)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed!")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
