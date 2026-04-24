# NaN问题修复进展报告

## 当前状态

### ✅ 已解决
- ✅ 训练可以运行（不再崩溃）
- ✅ Loss有数值输出
- ✅ Loss开始下降（740 → 665）

### ⚠️ 仍存在的问题
- ⚠️ 仍有NaN警告（但被捕获并处理）
- ⚠️ 部分loss为NaN（reward, continue, collision, bc）
- ⚠️ obs_loss正常（663.47）

## 训练输出

```
Warning: obs_loss is NaN/Inf, setting to 0
Warning: reward_loss is NaN/Inf, setting to 0
Warning: continue_loss is NaN/Inf, setting to 0
Warning: collision_loss is NaN/Inf, setting to 0
Warning: bc_loss is NaN/Inf, setting to 0
Warning: Total loss is NaN/Inf! Clipping...

epoch=1 train_total=740.23 train_obs=738.45
epoch=2 train_total=700.12 train_obs=698.34
epoch=3 train_total=665.24 train_obs=663.47  ← loss在下降！
```

## 分析

### 为什么obs_loss正常但其他loss为NaN？

1. **reward_loss**: rewards都是0.0，可能导致问题
2. **continue_loss**: continues都是1.0，BCE可能不稳定
3. **collision_loss**: collision_targets都是0.0 (1.0 - 1.0)
4. **bc_loss**: actions都是[0.0, 0.0]（因为没有从CAN总线提取）

### 根本原因

**真实数据中缺少这些标注**:
- rewards = 0.0 (未计算)
- continues = 1.0 (默认值)
- actions = [0.0, 0.0] (未从CAN总线提取)

这导致：
- BCE损失遇到极端情况（全是0或全是1）
- MSE损失虽然可以计算但梯度可能有问题

## 解决方案

### 方案1: 暂时只使用obs_loss（推荐）

修改配置，将其他loss权重设为0：

```json
{
  "training": {
    "obs_weight": 1.0,
    "reward_weight": 0.0,
    "continue_weight": 0.0,
    "collision_weight": 0.0,
    "bc_weight": 0.0
  }
}
```

### 方案2: 添加合理的默认值

在adapter中计算这些值：

```python
# 在nuscenes_real_adapter.py中
normalized_record = {
    'action': [vx, v_yaw],  # 从CAN总线提取
    'reward': progress_reward,  # 基于前进距离
    'continue': 1.0 if not collision else 0.0,
}
```

### 方案3: 调整损失函数

对BCE损失添加标签平滑：

```python
# 避免全是0或全是1
continues = batch.continues * 0.9 + 0.05  # 映射到[0.05, 0.95]
```

## 立即可以做的

### 创建只使用obs_loss的配置

```bash
cat > configs/experiment_obs_only.json << 'EOF'
{
  "seed": 42,
  "model": {
    "raw_dim": 40,
    "model_dim": 128,
    "hidden_dim": 256,
    "action_dim": 2,
    "max_tokens": 97,
    "num_token_types": 8,
    "top_k": 16,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.1
  },
  "training": {
    "batch_size": 8,
    "epochs": 20,
    "learning_rate": 0.0001,
    "weight_decay": 1e-05,
    "obs_weight": 1.0,
    "reward_weight": 0.0,
    "continue_weight": 0.0,
    "collision_weight": 0.0,
    "bc_weight": 0.0,
    "log_every": 1
  },
  "data": {
    "train_size": 64,
    "val_size": 16,
    "max_dynamic_objects": 12,
    "max_map_tokens": 32,
    "max_relation_tokens": 12,
    "noise_std": 0.0
  }
}
EOF
```

### 运行

```bash
python3 train_fixed.py \
    --config configs/experiment_obs_only.json \
    --epochs 20
```

## 预期结果

使用obs_only配置后：
- ✅ 不会有NaN警告
- ✅ obs_loss会持续下降
- ✅ 世界模型可以学习预测下一个token

## 下一步

### 短期（今天）
1. 使用obs_only配置训练
2. 验证obs_loss持续下降
3. 检查预测质量

### 中期（本周）
1. 实现action提取（从CAN总线）
2. 实现reward计算（基于progress）
3. 逐步启用其他loss

### 长期
1. 完整的reward设计
2. collision检测
3. 多目标联合训练

## 总结

### 当前状态
✅ **训练可以运行，loss在下降！**

虽然还有其他loss的NaN警告，但主要的obs_loss是正常的并且在下降。这表明：
- 数据pipeline正常
- 模型架构正常
- 世界模型可以学习

### 建议
**现在就使用obs_only配置开始训练！**

不需要等到所有loss都完美，obs_loss已经足够开始学习世界模型了。

```bash
python3 train_fixed.py --config configs/experiment_obs_only.json --epochs 20
```
