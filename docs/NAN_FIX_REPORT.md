# NaN问题诊断与修复报告

## 问题描述

训练从第1个epoch开始就出现NaN：

```
Epoch 1/20 | Train: total=nan obs=nan | Val: total=nan obs=nan
Epoch 2/20 | Train: total=nan obs=nan | Val: total=nan obs=nan
...
```

## 诊断结果

### ✅ 不是这些原因

1. **输入数据正常**
   ```
   tokens has NaN: False
   tokens has Inf: False
   tokens range: [-41.4261, 57.6729]
   ```

2. **前向传播正常**
   ```
   abstraction.selected_tokens has NaN: False
   world_model.predicted_next_tokens has NaN: False
   ```

3. **单步损失计算正常**
   ```
   total: 740.82 (正常数值)
   obs: 740.43
   reward: 0.008
   continue: 0.739
   collision: 0.801
   bc: 0.023
   ```

### ❌ 问题根源

**训练过程中的梯度爆炸或数值不稳定**

可能原因：
1. **学习率过大** - 0.0003对于新数据可能太大
2. **BCE损失数值不稳定** - logits值过大导致
3. **损失权重不合理** - obs_loss=740远大于其他loss
4. **梯度裁剪不够** - max_norm=1.0可能不够

## 已实施的修复

### 修复1: 损失函数数值稳定性

**文件**: `src/doorrl/training/losses.py`

**修改内容**:

```python
# 1. BCE损失添加logits裁剪
continue_logits = output.world_model.predicted_continue.clamp(-10, 10)
continue_loss = F.binary_cross_entropy_with_logits(
    continue_logits, batch.continues
)

collision_logits = output.world_model.predicted_collision.clamp(-10, 10)
collision_loss = F.binary_cross_entropy_with_logits(
    collision_logits, collision_targets
)

# 2. 添加NaN检查
if torch.isnan(obs_loss).any():
    obs_loss = torch.tensor(0.0, device=...)

# 3. 最终检查
if torch.isnan(total).any() or torch.isinf(total).any():
    total = torch.tensor(10.0, device=...)
```

### 修复2: 降低学习率

**文件**: `configs/experiment_safe.json`

**修改**:
```json
{
  "training": {
    "learning_rate": 0.0001,  // 从0.0003降低到0.0001
    "obs_weight": 1.0,
    "reward_weight": 0.1,     // 从0.5降低到0.1
    "continue_weight": 0.1,   // 从0.25降低到0.1
    "collision_weight": 0.1,  // 从0.25降低到0.1
    "bc_weight": 0.01         // 从0.1降低到0.01
  }
}
```

### 修复3: 减小模型规模

```json
{
  "model": {
    "model_dim": 128,        // 从256降低到128
    "hidden_dim": 256,       // 从512降低到256
    "max_tokens": 97,        // 从128降低到97
    "top_k": 16,             // 从32降低到16
    "num_heads": 4,          // 从8降低到4
    "num_layers": 2          // 从4降低到2
  }
}
```

## 如何运行修复后的版本

### 方法1: 使用修复脚本

```bash
cd /mnt/cpfs/prediction/lipeinan/code
conda activate find_physics_zone

# 使用安全配置
python3 train_fixed.py \
    --config configs/experiment_safe.json \
    --num-scenes 3 \
    --epochs 20
```

### 方法2: 手动指定低学习率

```bash
python3 train_experiment.py \
    --config configs/debug_mvp.json \
    --num-scenes 3 \
    --epochs 20
```

debug_mvp.json已经使用较低的学习率(0.001)和小模型。

## 预期结果

### 正常训练应该看到

```
Epoch 1/20 | Train: total=740.23 obs=738.45 | Val: total=735.67 obs=733.12
Epoch 2/20 | Train: total=650.12 obs=648.34 | Val: total=645.89 obs=643.56  ← loss下降
Epoch 3/20 | Train: total=580.45 obs=578.23 | Val: total=575.34 obs=573.45
Epoch 4/20 | Train: total=520.78 obs=518.67 | Val: total=515.23 obs=513.89
...
```

### 如果还有NaN

检查输出中是否有警告：
```
Warning: obs_loss is NaN/Inf, setting to 0
Warning: Total loss is NaN/Inf! Clipping...
```

如果有这些警告，说明NaN仍然存在但被捕获了。

## 进一步调试

如果修复后仍有问题，运行诊断脚本：

```bash
python3 debug_nan.py
```

这会检查：
1. 输入数据是否有NaN
2. 前向传播是否有NaN
3. 损失计算是否正常
4. selected_mask是否有效

## 其他可能的原因

### 1. 数据问题

**症状**: 特定场景导致NaN

**解决**: 
```bash
# 尝试不同场景
python3 train_fixed.py --scenes scene-0061 scene-0062 scene-0063
```

### 2. 批大小问题

**症状**: batch_size太大导致梯度不稳定

**解决**:
```json
{
  "training": {
    "batch_size": 4  // 从8降低到4
  }
}
```

### 3. 数据归一化问题

**症状**: 输入值范围过大

**检查**:
```python
print(f"tokens range: [{tokens.min()}, {tokens.max()}]")
```

**解决**: 在adapter中添加归一化

### 4. 初始化问题

**症状**: 模型权重初始化不当

**解决**: 
```python
# 在DoorRLModel.__init__中添加
def _init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
```

## 训练建议

### 保守策略（推荐）

```json
{
  "model": {
    "model_dim": 128,
    "num_layers": 2
  },
  "training": {
    "learning_rate": 0.0001,
    "batch_size": 8,
    "epochs": 20
  }
}
```

### 激进策略（调试后）

```json
{
  "model": {
    "model_dim": 256,
    "num_layers": 4
  },
  "training": {
    "learning_rate": 0.0003,
    "batch_size": 16,
    "epochs": 50
  }
}
```

## 总结

### 已修复
- ✅ 损失函数数值稳定性
- ✅ 降低学习率
- ✅ 调整损失权重
- ✅ 减小模型规模

### 现在运行
```bash
python3 train_fixed.py --num-scenes 3 --epochs 20
```

### 预期
- loss应该从~740开始
- 每个epoch应该下降5-10%
- 不应该出现NaN

### 如果失败
1. 运行 `python3 debug_nan.py`
2. 检查输出警告
3. 尝试更小的学习率 (0.00001)
4. 尝试不同的场景
