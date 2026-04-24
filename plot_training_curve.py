#!/usr/bin/env python3
"""可视化训练曲线"""
import matplotlib.pyplot as plt
import re
from pathlib import Path

def parse_training_log(log_file):
    """解析训练日志"""
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 匹配: epoch=1 train_total=829.8699 train_obs=829.8699 val_total=992.4824 val_obs=992.4824
            match = re.search(r'epoch=(\d+).*?train_total=([\d.]+).*?val_total=([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses

def plot_training_curve(log_file, output_file='training_curve.png'):
    """绘制训练曲线"""
    epochs, train_losses, val_losses = parse_training_log(log_file)
    
    if not epochs:
        print(f"No training data found in {log_file}")
        return
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=6)
    
    # 标注最佳点
    best_val_idx = val_losses.index(min(val_losses))
    plt.annotate(
        f'Best Val: {val_losses[best_val_idx]:.2f}\n@ Epoch {epochs[best_val_idx]}',
        xy=(epochs[best_val_idx], val_losses[best_val_idx]),
        xytext=(epochs[best_val_idx] + 2, val_losses[best_val_idx] + 50),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red'
    )
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Training curve saved to: {output_file}")
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"Training Statistics:")
    print(f"{'='*60}")
    print(f"Total epochs: {len(epochs)}")
    print(f"Initial train loss: {train_losses[0]:.2f}")
    print(f"Final train loss: {train_losses[-1]:.2f}")
    print(f"Best val loss: {val_losses[best_val_idx]:.2f} @ epoch {epochs[best_val_idx]}")
    print(f"Final val loss: {val_losses[-1]:.2f}")
    print(f"Train loss reduction: {(1 - train_losses[-1]/train_losses[0])*100:.1f}%")
    print(f"Val loss reduction: {(1 - val_losses[-1]/val_losses[0])*100:.1f}%")

if __name__ == '__main__':
    import sys
    
    # 查找最新的训练日志
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        # 自动查找最新的日志
        log_dir = Path('experiments/runs')
        if log_dir.exists():
            log_files = list(log_dir.glob('*/logs/training.log'))
            if log_files:
                log_file = str(sorted(log_files)[-1])
                print(f"Using latest log: {log_file}")
            else:
                print("No training logs found!")
                sys.exit(1)
        else:
            print("No experiments directory found!")
            sys.exit(1)
    
    output_file = str(Path(log_file).parent / 'training_curve.png')
    plot_training_curve(log_file, output_file)
