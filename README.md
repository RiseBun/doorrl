# DOOR-RL: Decision-Oriented Object-Relational Reinforcement Learning

A research codebase for autonomous driving decision learning with object-relational representation.

## Key Idea

**Reactive training matters, object-relational representation matters, and high-fidelity transfer matters.**

This project explores a typed-budget abstraction mechanism that separates dynamic agent selection from relation edge selection, enabling more robust latent-world reinforcement learning for autonomous driving.

## Project Structure

```
code/
├── src/doorrl/
│   ├── adapters/           # Dataset adapters (nuScenes, nuPlan, NAVSIM)
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Core models
│   │   ├── abstraction.py       # Decision-sufficient abstraction
│   │   ├── doorrl_variant.py    # Model variants (7 types)
│   │   ├── world_model.py       # Reactive world model
│   │   └── policy.py            # Actor-critic policy
│   ├── imagination/       # Latent imagination rollout
│   ├── training/          # Training loops
│   └── evaluation/        # Metrics and evaluation
├── configs/               # Model configurations
├── experiments/           # Experiment results
└── docs/                  # Detailed documentation
```

## Paper Tables

### Table 3: Representation Sufficiency Ablation (Stage 0)

Under a fair 16-slot world-model context budget, on nuScenes (700 scenes, 28,096 samples):

| Variant | Ctx | DynRoll ↓ | Coll F1 ↑ | Rare ADE ↓ | IntRec@1m ↑ |
|---------|:---:|:---------:|:---------:|:----------:|:-----------:|
| Holistic-16Slot | 16 | 2.11 ± 0.16 | 0.978 | 1.42 | 0.643 |
| Object-only-16 | 16 | 3.74 ± 1.01 | 0.946 | 1.10 | 0.901 |
| Object+Relation-16 (naive) | 16 | 40.28 ± 29.54 | 0.980 | 7.51 | 0.430 |
| Obj+Rel+Vis-16 | 16 | 15.80 ± 9.93 | 0.933 | 2.96 | 0.728 |
| **Obj+Rel-Decoupled (Ours)** | 16 | **2.11 ± 0.19** | 0.929 | **0.49** | **0.984** |
| **Decoupled+Visibility (Ours)** | 16 | **1.88 ± 0.23** | 0.926 | **0.52** | **0.980** |
| Holistic-full (ref) | 97 | 0.11 | 0.988 | 0.26 | 1.000 |

**Key Finding**: Naively mixing relation tokens into a shared top-k bottleneck fails catastrophically. Decoupled abstraction with typed per-role budgets resolves this.

### Table 4: Multi-step Latent Imagination RL (Stage 1)

Coming soon...

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RiseBun/doorrl.git
cd doorrl

# Install dependencies
pip install torch numpy nuscenes-devkit

# Or use the provided setup script
bash setup_doorrl_env.sh
```

### Stage 0: Representation Sufficiency Experiment

```bash
# Train all variants (seed 7)
python run_stage0_table3.py \
    --variant all_with_decoupled \
    --num-scenes 700 \
    --epochs 15 \
    --seed 7

# Run 3-seed experiment
bash scripts/run_fix2_3seeds.sh
```

### Stage 1: Latent Imagination RL

```bash
# Run imagination RL experiment
python run_stage1_table4.py \
    --condition pilot \
    --horizon 5 \
    --seed 7
```

## Documentation

- [Stage 0 Design & Results](docs/stage0.md) - Representation sufficiency analysis
- [Stage 1 Design](docs/stage1_design.md) - Latent imagination RL design
- [Tokenization Specification](docs/TOKENIZATION_SPEC.md) - Scene token schema
- [Server Handoff Guide](docs/SERVER_HANDOFF.md) - Development guide

## Core Components

### 1. Token Schema

The scene is tokenized into a fixed-size token sequence:

- **Dynamic tokens**: ego, vehicle, pedestrian, cyclist (~12-15 tokens)
- **Relation tokens**: interaction edges (TTC, lane conflict, priority)
- **Map tokens**: lane, crosswalk, stop line, etc.
- **Signal tokens**: traffic light states

### 2. Model Variants (7 types)

| Variant | Description |
|---------|-------------|
| `holistic` | Full 97-token context (upper bound) |
| `holistic_16slot` | Learned queries, 16 compressed slots |
| `object_only` | Top-k over dynamic agents only |
| `object_relation` | Top-k over dyn + rel (shared budget, **fails**) |
| `object_relation_visibility` | + Visibility weighting |
| `object_relation_decoupled` | **Decoupled** top-k: K_dyn=12, K_rel=4 |
| `object_relation_decoupled_visibility` | **Decoupled + Visibility** |

### 3. Decoupled Abstraction (Key Innovation)

```python
# Two independent selection heads
K_dyn = 12  # Select top-12 dynamic agents
K_rel = 4   # Select top-4 relation edges
# Total = 16 slots (same budget as other variants)

# No budget competition between types
# Relation slots no longer starve dynamic agent slots
```

## Citation

If you find this useful for your research, please cite:

```bibtex
@article{doorrl2026,
  title={DOOR-RL: Decision-Oriented Object-Relational Reinforcement Learning},
  author={},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

Built on top of [nuScenes](https://www.nuscenes.org/nuscenes), [nuPlan](https://www.nuscenes.org/nuplan), and [NAVSIM](https://navsim.ethz.ch/).
