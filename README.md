# DOOR-RL: 面向决策的目标-关系强化学习

一个用于自动驾驶决策学习的科研代码库。

## 核心思想

**反应式训练很重要，目标-关系表示法很重要，高保真迁移评估很重要。**

本项目探索了一种类型化预算抽象机制，将动态智能体选择与关系边选择分离，从而实现更鲁棒的自动驾驶潜在世界强化学习。

## 项目结构

```
code/
├── src/doorrl/
│   ├── adapters/           # 数据集适配器 (nuScenes, nuPlan, NAVSIM)
│   ├── data/              # 数据加载与预处理
│   ├── models/            # 核心模型
│   │   ├── abstraction.py       # 决策充分抽象
│   │   ├── doorrl_variant.py    # 模型变体 (7种)
│   │   ├── world_model.py       # 反应式世界模型
│   │   └── policy.py            # 演员-评论家策略
│   ├── imagination/       # 潜在想象 rollout
│   ├── training/          # 训练循环
│   └── evaluation/        # 指标与评估
├── configs/               # 模型配置
├── experiments/           # 实验结果
└── docs/                  # 详细文档
```

## 论文表格

### 表3: 表示充分性消融实验 (Stage 0)

在公平的16槽位世界模型上下文预算下，基于nuScenes数据集（700个场景，28,096个样本）：

| 变体 | Ctx | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|------|:---:|:---------:|:---------:|:----------:|:-----------:|
| Holistic-16Slot | 16 | 2.11 ± 0.16 | 0.978 | 1.42 | 0.643 |
| Object-only-16 | 16 | 3.74 ± 1.01 | 0.946 | 1.10 | 0.901 |
| Object+Relation-16 (naive) | 16 | 40.28 ± 29.54 | 0.980 | 7.51 | 0.430 |
| Obj+Rel+Vis-16 | 16 | 15.80 ± 9.93 | 0.933 | 2.96 | 0.728 |
| **Obj+Rel-Decoupled (Ours)** | 16 | **2.11 ± 0.19** | 0.929 | **0.49** | **0.984** |
| **Decoupled+Visibility (Ours)** | 16 | **1.88 ± 0.23** | 0.926 | **0.52** | **0.980** |
| Holistic-full (ref) | 97 | 0.11 | 0.988 | 0.26 | 1.000 |

**核心发现**: 将关系token简单混合到共享top-k瓶颈中会导致灾难性失败。解耦抽象机制通过类型化预算解决了这一问题。

### 表4: 多步潜在想象强化学习 (Stage 1)

即将到来...

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/RiseBun/doorrl.git
cd doorrl

# 安装依赖
pip install torch numpy nuscenes-devkit

# 或使用提供的安装脚本
bash setup_doorrl_env.sh
```

### Stage 0: 表示充分性实验

```bash
# 训练所有变体 (seed 7)
python run_stage0_table3.py \
    --variant all_with_decoupled \
    --num-scenes 700 \
    --epochs 15 \
    --seed 7

# 运行3种子实验
bash scripts/run_fix2_3seeds.sh
```

### Stage 1: 潜在想象强化学习

```bash
# 运行想象强化学习实验
python run_stage1_table4.py \
    --condition pilot \
    --horizon 5 \
    --seed 7
```

## 文档

- [Stage 0 设计与结果](docs/stage0.md) - 表示充分性分析
- [Stage 1 设计](docs/stage1_design.md) - 潜在想象强化学习设计
- [Tokenization规范](docs/TOKENIZATION_SPEC.md) - 场景token schema
- [服务器交接指南](docs/SERVER_HANDOFF.md) - 开发指南

## 核心组件

### 1. Token Schema

场景被token化为固定大小的序列：

- **动态token**: ego、车辆、行人、骑行者 (~12-15个token)
- **关系token**: 交互边 (TTC、车道冲突、优先级)
- **地图token**: 车道、人行横道、停止线等
- **信号token**: 交通灯状态

### 2. 模型变体 (7种)

| 变体 | 描述 |
|------|------|
| `holistic` | 完整97-token上下文 (上界) |
| `holistic_16slot` | 学习查询，16个压缩槽位 |
| `object_only` | 仅对动态智能体做top-k |
| `object_relation` | 对dyn+rel做top-k (共享预算，**失败**) |
| `object_relation_visibility` | + 可见性加权 |
| `object_relation_decoupled` | **解耦** top-k: K_dyn=12, K_rel=4 |
| `object_relation_decoupled_visibility` | **解耦 + 可见性** |

### 3. 解耦抽象 (核心创新)

```python
# 两个独立的选择头
K_dyn = 12  # 选择top-12个动态智能体
K_rel = 4   # 选择top-4个关系边
# 总计 = 16个槽位 (与其他变体预算相同)

# 类型间无预算竞争
# 关系槽位不再抢占动态智能体槽位
```

## 引用

如果本工作对你的研究有帮助，请引用：

```bibtex
@article{doorrl2026,
  title={DOOR-RL: Decision-Oriented Object-Relational Reinforcement Learning},
  author={},
  year={2026}
}
```

## 许可证

MIT License

## 致谢

基于 [nuScenes](https://www.nuscenes.org/nuscenes)、[nuPlan](https://www.nuscenes.org/nuplan) 和 [NAVSIM](https://navsim.ethz.ch/) 构建。
