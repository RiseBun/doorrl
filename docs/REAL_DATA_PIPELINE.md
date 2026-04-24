# 真实数据Pipeline使用指南

本文档介绍如何使用nuScenes和nuPlan真实数据训练DOOR-RL模型。

## 已实现的功能

### 1. NuScenes真实数据Adapter

**文件**: `src/doorrl/adapters/nuscenes_real_adapter.py`

**功能**:
- ✅ 从nuScenes数据库加载场景和样本
- ✅ 提取ego状态(位置、速度、朝向)
- ✅ 提取动态对象(车辆、行人、骑行者)
- ✅ 计算关系特征(相对位置、速度、TTC、碰撞风险)
- ✅ 转换为DOOR-RL token schema

**支持的数据**:
- nuScenes v1.0-trainval (850场景, 34149样本)
- 3D标注框
- CAN总线数据(可选)
- 传感器标定

### 2. NuPlan真实数据Adapter

**文件**: `src/doorrl/adapters/nuplan_real_adapter.py`

**功能**:
- ✅ 支持reactive和non-reactive模式
- ✅ 提取ego状态和动态对象
- ✅ 提取地图元素
- ✅ 计算关系特征
- ⚠️ 需要连接nuPlan数据库(TODO)

### 3. 真实数据集类

**文件**: `src/doorrl/data/real_dataset.py`

**提供的数据集**:
- `RealDrivingDataset` - 通用真实数据集
- `NuScenesSceneDataset` - nuScenes场景数据集

## 快速开始

### 测试真实数据Pipeline

```bash
cd /mnt/cpfs/prediction/lipeinan/code
python3 test_real_data.py
```

**测试内容**:
1. NuScenes Adapter初始化
2. 场景样本加载
3. 样本转换为token
4. 模型前向传播
5. 场景序列提取

### 使用真实nuScenes数据训练

```bash
# 使用前5个场景训练2个epoch
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --num-scenes 5 \
    --epochs 2

# 使用指定场景
python3 train_real_nuscenes.py \
    --config configs/debug_mvp.json \
    --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
    --scenes scene-0001 scene-0002 scene-0003 \
    --epochs 5
```

### 直接使用Adapter

```python
from doorrl.adapters.base import TokenizationSpec
from doorrl.adapters.nuscenes_real_adapter import NuScenesRealDataAdapter

# 创建spec
spec = TokenizationSpec(
    raw_dim=40,
    max_tokens=97,
    max_dynamic_objects=16,
    max_map_tokens=48,
    max_relation_tokens=16,
    action_dim=2,
)

# 初始化adapter
adapter = NuScenesRealDataAdapter(
    spec=spec,
    nuscenes_root="/mnt/datasets/e2e-nuscenes/20260302",
    version='v1.0-trainval',
    use_can_bus=True,
)

# 加载场景
samples = adapter.get_scene_samples("scene-0001")

# 转换为scene item
scene_item = adapter.convert_sample_to_scene_item(
    sample=samples[0],
    compute_relations=True,
)

# scene_item包含:
# - tokens: [97, 40]
# - token_mask: [97]
# - token_types: [97]
# - actions: [2]
# - next_tokens: [97, 40]
# - rewards: scalar
# - continues: scalar
```

## 数据流程

### NuScenes数据处理流程

```
nuScenes Database
    ↓
[1] 加载场景 (NuScenesRealDataAdapter.get_scene_samples)
    ↓
[2] 提取Ego状态 (_extract_ego_state)
    - 位置、速度、朝向
    - CAN总线数据(可选)
    ↓
[3] 提取动态对象 (_extract_objects)
    - 3D标注框转换到ego坐标系
    - 速度计算
    - 类别映射(VEHICLE/PEDESTRIAN/CYCLIST)
    ↓
[4] 计算关系特征 (_compute_relations)
    - 相对位置/速度
    - 碰撞时间(TTC)
    - 碰撞风险
    - 车道冲突
    ↓
[5] 转换为Token (NormalizedSceneConverter.build_scene_item)
    - Token编码
    - Mask生成
    - 类型标注
    ↓
DOOR-RL SceneBatch
```

### Token类型分布

从真实数据测试中看到的token类型:
- `0` (EGO) - 自车
- `1` (VEHICLE) - 车辆
- `2` (PEDESTRIAN) - 行人
- `4` (MAP) - 地图元素
- `6` (RELATION) - 关系token
- `7` (PAD) - 填充

## 配置说明

### debug_mvp.json (用于调试)

```json
{
  "model": {
    "raw_dim": 40,          // 原始token维度
    "max_tokens": 97,       // 最大token数量
    "top_k": 16             // 选择top-k重要token
  },
  "data": {
    "max_dynamic_objects": 12,   // 最大动态对象数
    "max_map_tokens": 32,        // 最大地图token数
    "max_relation_tokens": 12    // 最大关系token数
  }
}
```

### 推荐配置 (用于真实训练)

```json
{
  "model": {
    "raw_dim": 40,
    "model_dim": 256,
    "hidden_dim": 512,
    "max_tokens": 128,
    "top_k": 32,
    "num_heads": 8,
    "num_layers": 4
  },
  "data": {
    "max_dynamic_objects": 20,
    "max_map_tokens": 64,
    "max_relation_tokens": 20
  },
  "training": {
    "batch_size": 16,
    "epochs": 50,
    "learning_rate": 0.0003
  }
}
```

## 性能参考

基于测试数据:
- **数据集加载**: ~30秒 (nuScenes v1.0-trainval)
- **样本转换**: ~0.1秒/样本
- **Scene-0001**: 40帧, 平均每帧21个有效token
- **内存使用**: 约2GB (加载完整数据集)

## 下一步工作

### 已完成 ✅
- [x] NuScenes Adapter实现
- [x] 关系特征计算(TTC、风险、车道冲突)
- [x] 真实数据集类
- [x] 训练脚本
- [x] 测试验证

### 待实现 🔧
- [ ] NuPlan数据库连接
- [ ] 地图元素提取(车道、路沿)
- [ ] Action从CAN总线/轨迹提取
- [ ] Reward计算
- [ ] 数据缓存机制
- [ ] 多进程数据加载优化
- [ ] 可视化调试工具

## 常见问题

### Q: 加载nuScenes很慢怎么办?
A: 首次加载需要约30秒构建索引。可以使用预缓存:
```python
# 保存索引
import pickle
with open('nuscenes_cache.pkl', 'wb') as f:
    pickle.dump(nusc, f)
```

### Q: 如何选择场景?
A: 建议:
- 调试: 使用前5-10个场景
- 训练: 使用trainval split的所有场景
- 验证: 使用不同的场景划分

### Q: CAN总线数据加载失败?
A: CAN总线数据是可选的。设置`use_can_bus=False`即可:
```python
adapter = NuScenesRealDataAdapter(
    ...,
    use_can_bus=False,
)
```

### Q: 如何加速数据加载?
A: 使用多进程:
```python
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=SceneBatch.collate,
    num_workers=4,  # 使用4个worker
    pin_memory=True,
)
```

## 相关文件

- Adapter: `src/doorrl/adapters/nuscenes_real_adapter.py`
- Adapter: `src/doorrl/adapters/nuplan_real_adapter.py`
- Dataset: `src/doorrl/data/real_dataset.py`
- 训练: `train_real_nuscenes.py`
- 测试: `test_real_data.py`
- 探索: `explore_nuscenes.py`
