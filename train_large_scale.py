"""大规模训练脚本 - 支持真实nuScenes数据的大规模实验"""
from __future__ import annotations

import argparse
import json
import time
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

import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Large-scale DOOR-RL training")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "experiment_large_scale.json"))
    parser.add_argument("--nuscenes-root", type=str, default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument("--num-scenes", type=int, default=100, help="Number of scenes to use")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/large_runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--variant", type=str, default="object_relation", 
                        choices=["holistic", "object_only", "object_relation", "full"],
                        help="Model variant to use")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def create_experiment_dir(output_dir: str, config_name: str) -> Path:
    """创建实验输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{config_name}_{timestamp}"
    
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "metrics").mkdir(exist_ok=True)
    
    return exp_dir


def save_config(exp_dir: Path, args: argparse.Namespace, config: DoorRLConfig) -> None:
    """保存实验配置"""
    args_dict = vars(args)
    args_dict['timestamp'] = datetime.now().isoformat()
    
    with open(exp_dir / "args.json", "w") as f:
        json.dump(args_dict, f, indent=2)
    
    config_dict = {
        "seed": config.seed,
        "model": {k: v for k, v in vars(config.model).items()},
        "training": {k: v for k, v in vars(config.training).items()},
        "data": {k: v for k, v in vars(config.data).items()},
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


class LargeScaleTrainer(DoorRLTrainer):
    """大规模训练器 - 支持断点续训和详细日志"""
    
    def __init__(self, model, config, device, exp_dir: Path = None, save_every: int = 10):
        super().__init__(model, config, device)
        self.exp_dir = exp_dir
        self.save_every = save_every
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.train_history = []
        self.val_history = []
        
        # 创建日志文件
        if self.exp_dir:
            self.log_file = open(exp_dir / "logs" / "training.log", "w")
            self.metrics_file = open(exp_dir / "metrics" / "metrics.jsonl", "w")
        else:
            self.log_file = None
            self.metrics_file = None
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """保存检查点"""
        if self.exp_dir is None:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.exp_dir / "checkpoints" / "latest.pth")
        
        # 保存最佳检查点
        if is_best:
            torch.save(checkpoint, self.exp_dir / "checkpoints" / "best.pth")
            print(f"  ✓ Best checkpoint saved (val_loss={self.best_val_loss:.4f})")
        
        # 保存epoch检查点
        if epoch % self.save_every == 0:
            torch.save(checkpoint, self.exp_dir / "checkpoints" / f"epoch_{epoch}.pth")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_epoch = checkpoint['epoch']
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        print(f"Loaded checkpoint from epoch {self.start_epoch}")
        return self.start_epoch
    
    def fit(self, train_loader, val_loader=None) -> None:
        """大规模训练循环"""
        total_epochs = self.config.epochs
        
        for epoch in range(self.start_epoch, total_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_stats = self.run_epoch(train_loader, training=True)
            self.train_history.append(train_stats.copy())
            
            # 验证阶段
            message = f"Epoch {epoch + 1}/{total_epochs}"
            message += f" | Train: total={train_stats['total']:.4f}"
            
            is_best = False
            if val_loader is not None:
                val_stats = self.run_epoch(val_loader, training=False)
                self.val_history.append(val_stats.copy())
                
                message += f" | Val: total={val_stats['total']:.4f}"
                
                if val_stats['total'] < self.best_val_loss:
                    self.best_val_loss = val_stats['total']
                    is_best = True
            
            epoch_time = time.time() - epoch_start
            message += f" | Time: {epoch_time:.1f}s"
            
            print(message)
            
            # 记录到日志文件
            if self.log_file:
                self.log_file.write(message + "\n")
                self.log_file.flush()
            
            # 记录到metrics文件
            if self.metrics_file:
                metrics_record = {
                    'epoch': epoch + 1,
                    'train': train_stats,
                    'val': self.val_history[-1] if val_loader else None,
                    'epoch_time': epoch_time,
                }
                self.metrics_file.write(json.dumps(metrics_record) + "\n")
                self.metrics_file.flush()
            
            # 保存检查点
            self.save_checkpoint(epoch + 1, is_best)
        
        # 关闭日志文件
        if self.log_file:
            self.log_file.close()
        if self.metrics_file:
            self.metrics_file.close()
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        if self.exp_dir:
            print(f"Experiment artifacts saved to: {self.exp_dir}")


def get_available_scenes(nusc) -> list:
    """获取所有可用场景"""
    return [s['name'] for s in nusc.scene]


def main() -> None:
    args = parse_args()
    
    # 加载配置
    config = DoorRLConfig.from_json(args.config)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # 创建实验目录
    config_name = Path(args.config).stem
    exp_dir = create_experiment_dir(args.output_dir, config_name)
    save_config(exp_dir, args, config)
    
    print(f"=" * 80)
    print(f"Large-Scale DOOR-RL Training")
    print(f"=" * 80)
    print(f"Experiment dir: {exp_dir}")
    print(f"Config: {args.config}")
    print(f"Scenes: {args.num_scenes}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Variant: {args.variant}")
    print(f"=" * 80)
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 检测设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        mem_free = mem_total - torch.cuda.memory_allocated() / 1e9
        print(f"GPU Memory: {mem_free:.1f}GB / {mem_total:.1f}GB")
    
    # 加载数据集
    print(f"\nLoading nuScenes data from {args.nuscenes_root}...")
    
    full_dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        version='v1.0-trainval',
    )
    
    # 选择指定数量的场景
    available_scenes = full_dataset.scenes
    selected_scenes = available_scenes[:min(args.num_scenes, len(available_scenes))]
    
    print(f"Available scenes: {len(available_scenes)}")
    print(f"Selected scenes: {len(selected_scenes)}")
    
    # 重新构建数据集
    dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        scenes=selected_scenes,
        version='v1.0-trainval',
    )
    
    # 划分训练集/验证集 (90/10)
    total_size = len(dataset)
    val_size = int(0.1 * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Total: {total_size} samples")
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=SceneBatch.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"  Batches per epoch (train): {len(train_loader)}")
    print(f"  Batches per epoch (val): {len(val_loader)}")
    
    # 创建模型
    model = DoorRLModel(config.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {num_params:,} parameters ({num_params * 4 / 1e6:.1f} MB)")
    
    # 创建训练器
    trainer = LargeScaleTrainer(
        model=model,
        config=config.training,
        device=device,
        exp_dir=exp_dir,
        save_every=config.training.get('save_every', 10),
    )
    
    # 可选：从检查点恢复
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch}")
    
    # 开始训练
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")
    
    train_start = time.time()
    trainer.fit(train_loader, val_loader=val_loader)
    total_time = time.time() - train_start
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"Training completed!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Results saved to: {exp_dir}")
    print(f"{'='*80}")
    
    # 保存最终结果
    results = {
        "best_val_loss": trainer.best_val_loss,
        "total_epochs": config.training.epochs,
        "total_time_minutes": total_time / 60,
        "train_size": train_size,
        "val_size": val_size,
        "num_params": num_params,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()