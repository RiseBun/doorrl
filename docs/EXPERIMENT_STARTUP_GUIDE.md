# 🚀 实验启动指南

## 📊 当前状态检查

### ✅ 已就绪

| 项目 | 状态 | 说明 |
|------|------|------|
| **环境配置** | ✅ 完成 | conda环境 `find_physics_zone`，所有依赖已安装 |
| **数据集** | ✅ 就绪 | nuScenes (850场景), nuPlan (完整) |
| **代码框架** | ✅ 完成 | Adapter、Dataset、Model、Trainer全部实现 |
| **测试验证** | ✅ 通过 | 所有测试通过，真实数据pipeline验证成功 |
| **文档** | ✅ 完善 | 使用指南、API文档、项目说明 |

### ⚠️ 缺失的实验基础设施

| 项目 | 状态 | 优先级 | 说明 |
|------|------|--------|------|
| **实验配置** | ⚠️ 仅调试配置 | 🔴 高 | 缺少真实训练的实验配置 |
| **日志系统** | ❌ 缺失 | 🔴 高 | 无TensorBoard/WandB日志 |
| **检查点保存** | ❌ 缺失 | 🔴 高 | 无法保存和恢复训练 |
| **实验管理** | ❌ 缺失 | 🟡 中 | 无实验追踪和对比工具 |
| **评估指标** | ❌ 缺失 | 🟡 中 | 无定量评估指标 |
| **可视化** | ❌ 缺失 | 🟢 低 | 无训练曲线和结果可视化 |

---

## 🎯 立即开始实验（3个方案）

### 方案1：快速验证实验（现在就可以跑）⭐⭐⭐

**目标**: 验证真实数据pipeline可以训练

**所需时间**: 5-10分钟

**步骤**:
```bash
cd /mnt/cpfs/prediction/lipeinan/code
conda activate find_physics_zone

# 使用现有配置，前3个场景，2个epoch
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --scenes scene-0001 scene-0002 scene-0003 \
    --epochs 2
```

**预期输出**:
```
Using device: cuda
Loading nuScenes data from ...
NuScenesSceneDataset: 120 samples from 3 scenes
Dataset size: 120 samples

Model parameters: 1,234,567

epoch=1 train_total=123.4567 train_obs=120.1234
epoch=2 train_total=98.7654 train_obs=95.4321

Training completed!
```

**优点**: 
- ✅ 零准备，立即运行
- ✅ 验证代码和数据pipeline
- ✅ 快速发现问题

**缺点**:
- ❌ 无日志记录
- ❌ 无模型保存
- ❌ 无法复现实验

---

### 方案2：完整实验设置（推荐）⭐⭐⭐⭐⭐

**目标**: 建立可复现、可追踪的实验流程

**所需时间**: 30分钟设置 + 数小时训练

#### Step 1: 创建实验配置

```bash
# 创建实验配置目录
mkdir -p configs/experiments
```

创建文件 `configs/experiments/nuscenes_wm_pretrain.json`:

```json
{
  "seed": 42,
  "model": {
    "raw_dim": 40,
    "model_dim": 256,
    "hidden_dim": 512,
    "action_dim": 2,
    "max_tokens": 128,
    "num_token_types": 8,
    "top_k": 32,
    "num_heads": 8,
    "num_layers": 4,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0003,
    "weight_decay": 1e-05,
    "obs_weight": 1.0,
    "reward_weight": 0.5,
    "continue_weight": 0.25,
    "collision_weight": 0.25,
    "bc_weight": 0.1,
    "log_every": 5,
    "save_every": 10,
    "eval_every": 5
  },
  "data": {
    "max_dynamic_objects": 20,
    "max_map_tokens": 48,
    "max_relation_tokens": 20,
    "noise_std": 0.0
  },
  "experiment": {
    "name": "nuscenes_wm_pretrain_v1",
    "num_scenes": 50,
    "scenes_file": "configs/experiments/scenes_50.txt"
  }
}
```

#### Step 2: 创建场景列表

```bash
# 创建场景列表文件
cat > configs/experiments/scenes_50.txt << 'EOF'
scene-0001
scene-0002
scene-0003
scene-0004
scene-0005
scene-0006
scene-0007
scene-0008
scene-0009
scene-0010
scene-0011
scene-0012
scene-0013
scene-0014
scene-0015
scene-0016
scene-0017
scene-0018
scene-0019
scene-0020
scene-0021
scene-0022
scene-0023
scene-0024
scene-0025
scene-0026
scene-0027
scene-0028
scene-0029
scene-0030
scene-0031
scene-0032
scene-0033
scene-0034
scene-0035
scene-0036
scene-0037
scene-0038
scene-0039
scene-0040
scene-0041
scene-0042
scene-0043
scene-0044
scene-0045
scene-0046
scene-0047
scene-0048
scene-0049
scene-0050
EOF
```

#### Step 3: 创建目录结构

```bash
# 创建实验输出目录
mkdir -p experiments
mkdir -p experiments/runs
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
```

#### Step 4: 运行实验

```bash
python3 train_real_nuscenes.py \
    --config configs/experiments/nuscenes_wm_pretrain.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302
```

---

### 方案3：生产级实验系统（长期）⭐⭐⭐⭐

**目标**: 完整的实验管理和追踪系统

**需要添加的功能**:

#### 1. 日志系统 (TensorBoard/WandB)

安装:
```bash
pip install tensorboard wandb
```

在训练脚本中添加:
```python
from torch.utils.tensorboard import SummaryWriter

# 初始化
writer = SummaryWriter(f'experiments/logs/{experiment_name}')

# 记录损失
writer.add_scalar('Loss/train_total', stats['total'], epoch)
writer.add_scalar('Loss/train_obs', stats['obs'], epoch)

# 记录模型参数
for name, param in model.named_parameters():
    writer.add_histogram(f'Params/{name}', param, epoch)
```

#### 2. 检查点保存

在trainer中添加:
```python
def save_checkpoint(self, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'config': self.config,
    }, filepath)

def load_checkpoint(self, filepath):
    checkpoint = torch.load(filepath)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']
```

#### 3. 实验管理脚本

创建 `run_experiment.py`:
```python
#!/usr/bin/env python3
"""实验运行脚本 - 自动管理配置、日志、检查点"""

import argparse
import json
from datetime import datetime
from pathlib import Path

def run_experiment(config_path, output_dir):
    # 生成实验名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = json.load(open(config_path))
    exp_name = f"{config['experiment']['name']}_{timestamp}"
    
    # 创建输出目录
    exp_dir = Path(output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=2))
    
    # 运行训练
    print(f"Running experiment: {exp_name}")
    print(f"Output directory: {exp_dir}")
    
    # TODO: 调用训练脚本
    # subprocess.run(['python3', 'train_real_nuscenes.py', ...])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', default='experiments/runs')
    args = parser.parse_args()
    
    run_experiment(args.config, args.output_dir)
```

---

## 🔧 立即可做的改进

### 改进1: 增强训练脚本（10分钟）

在 `train_real_nuscenes.py` 中添加:

```python
import json
from datetime import datetime
from pathlib import Path

def main():
    args = parse_args()
    config = DoorRLConfig.from_json(args.config)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"run_{timestamp}"
    output_dir = Path("experiments/runs") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    (output_dir / 'config.json').write_text(
        json.dumps(vars(args), indent=2)
    )
    
    # ... 训练代码 ...
    
    # 保存最终模型
    checkpoint_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': config.training.epochs,
        'model_state_dict': model.state_dict(),
        'config': config,
    }, checkpoint_path)
    
    print(f"Experiment saved to: {output_dir}")
```

### 改进2: 添加验证集划分（5分钟）

```python
# 在NuScenesSceneDataset中
def __init__(self, ..., val_ratio=0.2):
    # 划分训练集和验证集
    num_val = int(len(self.sample_index) * val_ratio)
    self.train_index = self.sample_index[:-num_val]
    self.val_index = self.sample_index[-num_val:]
```

### 改进3: 添加简单评估（15分钟）

```python
def evaluate(model, val_loader, device):
    """简单评估函数"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            output = model(batch)
            loss, stats = compute_losses(batch, output, config)
            total_loss += stats['total']
            num_batches += 1
    
    return total_loss / num_batches
```

---

## 📋 实验检查清单

### 开始实验前

- [ ] ✅ 环境配置完成
- [ ] ✅ 数据下载完成
- [ ] ✅ 代码测试通过
- [ ] ⚠️ 创建实验配置
- [ ] ⚠️ 创建输出目录
- [ ] ⚠️ 选择实验场景

### 实验运行中

- [ ] 监控训练损失（打印/TensorBoard）
- [ ] 检查GPU使用情况
- [ ] 定期检查中间结果
- [ ] 保存重要检查点

### 实验完成后

- [ ] 记录最终指标
- [ ] 保存模型和配置
- [ ] 生成训练曲线
- [ ] 撰写实验笔记

---

## 🎓 典型实验流程

### 实验1: 基线训练

```bash
# 配置
Config: configs/experiments/baseline.json
Scenes: 50个
Epochs: 50
Model: model_dim=256, num_layers=4

# 运行
python3 train_real_nuscenes.py \
    --config configs/experiments/baseline.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302
```

### 实验2: 表示消融

```bash
# Variant 1: 无关系token
Config: configs/experiments/ablation_no_relation.json
修改: max_relation_tokens=0

# Variant 2: 完整关系
Config: configs/experiments/ablation_full_relation.json
修改: max_relation_tokens=20
```

### 实验3: 规模消融

```bash
# 小模型
Config: model_dim=128, num_layers=2

# 中模型
Config: model_dim=256, num_layers=4

# 大模型
Config: model_dim=512, num_layers=6
```

---

## 💡 推荐行动

### 现在立即做（5分钟）

```bash
# 1. 运行快速验证实验
cd /mnt/cpfs/prediction/lipeinan/code
python3 train_real_nuscenes.py \
    --scenes scene-0001 scene-0002 scene-0003 \
    --epochs 2

# 2. 创建实验目录
mkdir -p experiments/{runs,checkpoints,logs}
```

### 今天完成（1-2小时）

1. ✅ 创建完整的实验配置
2. ✅ 增强训练脚本（添加日志和检查点）
3. ✅ 运行第一个正式实验（50场景，20 epochs）
4. ✅ 建立实验记录文档

### 本周完成

1. 集成TensorBoard日志
2. 添加验证集评估
3. 运行消融实验
4. 分析结果

---

## 📊 实验记录模板

创建 `experiments/experiment_log.md`:

```markdown
# 实验记录

## Experiment 001: Baseline World Model Pretraining

**日期**: 2026-04-16
**配置**: configs/experiments/baseline.json
**数据**: nuScenes 50 scenes

### 超参数
- model_dim: 256
- num_layers: 4
- batch_size: 16
- learning_rate: 0.0003
- epochs: 50

### 结果
- Final train loss: XXX
- Final val loss: XXX
- Training time: X hours

### 观察
- ...

### 问题
- ...
```

---

## 🔍 故障排除

### 问题1: CUDA Out of Memory

```bash
# 减小batch size
--batch-size 8

# 或减小模型
model_dim: 128
```

### 问题2: 数据加载慢

```python
# 增加worker数量
DataLoader(..., num_workers=4, pin_memory=True)
```

### 问题3: 损失不下降

```bash
# 检查学习率
learning_rate: 0.0001  # 减小

# 或增加warmup
# TODO: 添加learning rate scheduler
```

---

## 📞 需要帮助？

如果遇到问题：

1. 查看日志输出
2. 检查GPU使用: `nvidia-smi`
3. 验证数据: `python3 test_real_data.py`
4. 查看文档: `docs/REAL_DATA_PIPELINE.md`

---

## 🎯 总结

### 你现在的状态

✅ **完全具备开始实验的条件！**

### 最小启动步骤

```bash
# 就这3步，现在就可以开始

# 1. 进入项目目录
cd /mnt/cpfs/prediction/lipeinan/code

# 2. 激活环境
conda activate find_physics_zone

# 3. 运行实验
python3 train_real_nuscenes.py \
    --scenes scene-0001 scene-0002 scene-0003 \
    --epochs 5
```

### 下一步优化

跑完上面的快速实验后，再考虑：
1. 添加日志系统
2. 添加检查点保存
3. 创建完整实验配置
4. 运行大规模实验

**不要等到所有工具都完美才开始 - 先跑起来，再迭代优化！** 🚀
