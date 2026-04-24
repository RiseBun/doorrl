"""
Stage 0: Table 3 表示充分性消融实验

目标: 证明 object-relational representation matters

输出: 论文Table 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import random

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.data.nuplan_dataset import NuPlanPreprocessedDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch
from doorrl.training import DoorRLTrainer
from doorrl.utils import set_seed
from doorrl.evaluation.table3_metrics import Table3Metrics, evaluate_stage0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0: Table 3 Representation Sufficiency Ablation"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "debug_mvp.json"),
        help="Path to config file.",
    )
    parser.add_argument(
        "--nuscenes-root",
        type=str,
        default="/mnt/datasets/e2e-nuscenes/20260302",
        help="Path to nuScenes dataset.",
    )
    parser.add_argument(
        "--token-cache-dir",
        type=str,
        default="experiments/_token_cache",
        help="Disk cache dir for pre-tokenised nuScenes scenes. "
             "Skips the ~18 min devkit pass on every subsequent run.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["nuscenes", "nuplan"],
        default="nuscenes",
        help=(
            "Which data source to train/evaluate on. "
            "'nuplan' reads Diffusion-Planner preprocessed NPZ from "
            "--nuplan-root."
        ),
    )
    parser.add_argument(
        "--nuplan-root",
        type=str,
        default="/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split",
        help="Path to preprocessed nuPlan NPZ root (processed_agent64_split).",
    )
    parser.add_argument(
        "--nuplan-num-samples",
        type=int,
        default=5000,
        help=(
            "Number of NPZ files to load when --dataset nuplan. "
            "With ~1 M files available, start small (e.g. 5000) for "
            "pilots and grow for the final runs."
        ),
    )
    parser.add_argument(
        "--nuplan-index-json",
        type=str,
        default=None,
        help=(
            "Optional: path to a precomputed list of NPZ relative paths "
            "(e.g. diffusion_planner_agent64_train_paths.json); bypasses "
            "filesystem walking. Can be relative to --nuplan-root."
        ),
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=[
            "holistic",
            "holistic_16slot",
            "object_only",
            "object_relation",
            "object_relation_visibility",
            "object_relation_decoupled",
            "object_relation_decoupled_visibility",
            "all",
            "all_with_decoupled",
        ],
        default="object_relation",
        help="Model variant to evaluate, or 'all' to run all variants.",
    )
    parser.add_argument(
        "--scene-val-ratio",
        type=float,
        default=0.2,
        help="Fraction of scenes held out for validation (scene-level split).",
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
        help="Training epochs.",
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
        default=str(ROOT / "experiments" / "table3_representation_sufficiency"),
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate, skip training (requires saved model).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Override the learning rate from the config. "
            "If left None, use config's LR optionally multiplied by --lr-scale."
        ),
    )
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier applied on top of the config's learning_rate. "
            "Use this to compensate for large-batch training, e.g. "
            "--batch-size 256 --lr-scale 8 when the original was bs=32."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile(mode='reduce-overhead').",
    )

    return parser.parse_args()


def _build_loaders(args: argparse.Namespace, config: DoorRLConfig):
    """Load + tokenize + split the data once, return loaders.

    Factored out so all model variants in ``run_all_variants`` share a single
    tokenisation pass (each variant only swaps the model, not the data).

    Split policy (fair Stage 0):
      * For nuScenes: scene-level shuffle (seed), NOT sample-level. Sample-
        level splitting puts adjacent frames of the same 20 s scene into
        both train and val, which is a data leak.
      * For nuPlan preprocessed: each NPZ is a standalone frame anchor so
        sample-level shuffle is safe (no within-"scene" temporal leak).
        We still go through the same ``indices_for_scenes`` API for
        interface parity; each NPZ file is treated as its own "scene".
    """
    if args.dataset == "nuplan":
        print("Loading nuPlan preprocessed data...")
        full_dataset = NuPlanPreprocessedDataset(
            config=config,
            data_root=args.nuplan_root,
            num_samples=args.nuplan_num_samples,
            index_json=args.nuplan_index_json,
            seed=args.seed,
        )
    else:
        print("Loading nuScenes data...")
        full_dataset = NuScenesSceneDataset(
            config=config,
            nuscenes_root=args.nuscenes_root,
            num_scenes=args.num_scenes,
            version='v1.0-trainval',
            cache_dir=getattr(args, "token_cache_dir", None) or None,
        )

    # ---- Scene-level split --------------------------------------------
    scene_names_sorted = sorted(set(full_dataset.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(scene_names_sorted)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * n_total))) if n_total > 1 else 0
    n_train = n_total - n_val
    train_scenes = set(shuffled[:n_train])
    val_scenes = set(shuffled[n_train:])

    train_indices = full_dataset.indices_for_scenes(train_scenes)
    val_indices = full_dataset.indices_for_scenes(val_scenes)

    print(
        f"Scene-level split: {n_train} train scenes / {n_val} val scenes "
        f"-> {len(train_indices)} train samples / {len(val_indices)} val samples"
    )
    if n_val == 0 or len(val_indices) == 0:
        raise ValueError(
            "Scene-level split produced 0 validation samples; "
            f"got n_total={n_total}, scene_val_ratio={args.scene_val_ratio}."
        )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Data is fully in-memory cached, so num_workers>0 only helps with
    # collate + pin_memory overlap. 2 workers + persistent + prefetch is the
    # sweet spot; more doesn't help because __getitem__ is an O(1) dict
    # lookup into self._cache.
    loader_kwargs = dict(
        batch_size=config.training.batch_size,
        collate_fn=SceneBatch.collate,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def run_stage0_experiment(
    args: argparse.Namespace,
    loaders=None,
) -> Dict:
    """
    运行Stage 0实验

    Args:
        args: CLI 解析后的参数
        loaders: 可选的 ``(train_loader, val_loader)`` 元组; 传入则复用
                 (用于 ``run_all_variants`` 下 4 个 variant 共享数据), 否则
                 现场构建 (保留 single-variant 调用的向后兼容).

    Returns:
        实验结果字典
    """
    # 1. 加载配置
    config = DoorRLConfig.from_json(args.config)
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.seed = args.seed
    # Learning rate override / scaling for large-batch training.
    base_lr = config.training.learning_rate
    if getattr(args, "lr", None) is not None:
        config.training.learning_rate = float(args.lr)
    else:
        config.training.learning_rate = base_lr * float(getattr(args, "lr_scale", 1.0))
    if config.training.learning_rate != base_lr:
        print(
            f"[lr] override {base_lr:.2e} -> {config.training.learning_rate:.2e} "
            f"(lr={getattr(args, 'lr', None)}, lr_scale={getattr(args, 'lr_scale', 1.0)})"
        )

    # 2. 设置随机种子
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*80)
    print("Stage 0: Table 3 - Representation Sufficiency Ablation")
    print("="*80)
    print(f"Variant: {args.variant}")
    print(f"Device: {device}")
    print(f"Scenes: {args.num_scenes}")
    print(f"Epochs: {args.epochs}")
    print()

    # 3. 加载数据 (或复用)
    if loaders is None:
        train_loader, val_loader = _build_loaders(args, config)
    else:
        train_loader, val_loader = loaders

    # 4. 创建模型
    variant = ModelVariant(args.variant)
    model = DoorRLModelVariant(config.model, variant)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        print("[compile] torch.compile(mode='reduce-overhead') ...")
        model = torch.compile(model, mode="reduce-overhead")
    
    # 5. 训练或加载
    exp_dir = Path(args.output_dir) / args.variant
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = exp_dir / "model.pt"
    
    if args.evaluate_only and model_path.exists():
        print(f"\nLoading saved model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"\nTraining model...")
        trainer = DoorRLTrainer(
            model=model,
            config=config.training,
            device=device,
        )
        trainer.fit(train_loader, val_loader=val_loader)
        
        # 保存模型
        torch.save({
            'variant': args.variant,
            'model_state_dict': model.state_dict(),
            'config': {
                'variant': args.variant,
                'num_scenes': args.num_scenes,
                'epochs': args.epochs,
                'seed': args.seed,
            },
        }, model_path)
        print(f"Model saved to {model_path}")
    
    # 6. 评估 (Table 3指标)
    print("\n" + "="*80)
    print("Evaluating Table 3 Metrics")
    print("="*80)
    
    model.to(device)
    table3_metrics = evaluate_stage0(
        model=model,
        data_loader=val_loader,
        variant_name=args.variant,
        device=device,
        verbose=True,
    )
    
    # 7. 保存结果
    results = table3_metrics.compute_table3()
    
    results_path = exp_dir / "table3_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    
    print(f"\nResults saved to {results_path}")
    
    return results


def run_all_variants(args: argparse.Namespace) -> None:
    """Run all fair-stage-0 variants and emit the decision-oriented Table 3."""

    # Core "fair" comparison: all variants receive exactly 16 slots into the
    # world model. ``holistic`` (97-token upper bound) is kept as reference.
    base_variants = [
        "holistic_16slot",
        "object_only",
        "object_relation",
        "object_relation_visibility",
    ]
    decoupled_variants = [
        "object_relation_decoupled",
        "object_relation_decoupled_visibility",
    ]
    if args.variant == "all_with_decoupled":
        variants = base_variants + decoupled_variants + ["holistic"]
    else:
        variants = base_variants + ["holistic"]

    all_results = {}

    # Build data loaders ONCE and share across all variants. Tokenising 28k
    # nuScenes samples takes 5-15 minutes; doing it five times (once per
    # variant) wastes roughly 40 min with zero benefit since none of the
    # variants modify the underlying data.
    shared_config = DoorRLConfig.from_json(args.config)
    shared_config.training.epochs = args.epochs
    shared_config.training.batch_size = args.batch_size
    shared_config.seed = args.seed
    set_seed(shared_config.seed)
    shared_loaders = _build_loaders(args, shared_config)

    for variant in variants:
        variant_args = argparse.Namespace(**vars(args))
        variant_args.variant = variant

        try:
            results = run_stage0_experiment(variant_args, loaders=shared_loaders)
            all_results[variant] = results
        except Exception as e:
            print(f"\n\u2717 Failed for {variant}: {e}")
            import traceback
            traceback.print_exc()

    # --- Print fair Table 3 (Decision-Oriented Representation Analysis) ---
    variant_names = {
        "holistic_16slot": "Holistic-16Slot",
        "object_only": "Object-only-16",
        "object_relation": "Object+Relation-16",
        "object_relation_visibility": "Obj+Rel+Visibility-16",
        "object_relation_decoupled": "Object+Rel-Decoupled",
        "object_relation_decoupled_visibility": "Decoupled+Visibility",
        "holistic": "Holistic-full (ref)",
    }
    context_budget = {
        "holistic_16slot": 16,
        "object_only": 16,
        "object_relation": 16,
        "object_relation_visibility": 16,
        "object_relation_decoupled": 16,
        "object_relation_decoupled_visibility": 16,
        "holistic": 97,
    }

    print("\n" + "=" * 110)
    print("Table 3: Decision-Oriented Representation Analysis")
    print("=" * 110)
    header = (
        f"{'Variant':<26} | {'Ctx':<4} | "
        f"{'Dyn Rollout ↓':<14} | {'Action MSE ↓':<13} | "
        f"{'Coll. F1 ↑':<11} | {'Rare ADE ↓':<11} | {'IntRec@1m ↑':<12}"
    )
    print(header)
    print("-" * len(header))

    for variant in variants:
        if variant not in all_results:
            continue
        r = all_results[variant]
        name = variant_names[variant]
        ctx = context_budget[variant]
        ref_tag = "  (ref)" if variant == "holistic" else ""
        print(
            f"{name:<26} | {ctx:<4d} | "
            f"{r['dyn_rollout_mse']:.4f}         | "
            f"{r['action_mse']:.4f}        | "
            f"{r['collision_f1']:.4f}      | "
            f"{r['rare_ade']:.4f}      | "
            f"{r['interaction_recall_at_1m']:.4f}{ref_tag}"
        )
    print("-" * len(header))

    # --- LaTeX ---------------------------------------------------------
    print("\nLaTeX Table:")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Decision-Oriented Representation Analysis (fair Stage 0, 16-slot context budget).}")
    print(r"\label{tab:decision_oriented_repr}")
    print(r"\begin{tabular}{lccccccc}")
    print(r"\toprule")
    print(
        r"\textbf{Variant} & \textbf{Ctx} & "
        r"\textbf{Dyn Rollout} $\downarrow$ & \textbf{Action MSE} $\downarrow$ & "
        r"\textbf{Collision F1} $\uparrow$ & \textbf{Rare ADE} $\downarrow$ & "
        r"\textbf{Interaction Recall@1m} $\uparrow$ \\"
    )
    print(r"\midrule")
    # Insert a midrule before the holistic-full reference row to visually
    # separate the fair comparison block from the upper-bound reference.
    for variant in variants:
        if variant not in all_results:
            continue
        r = all_results[variant]
        name = variant_names[variant]
        ctx = context_budget[variant]
        last_fair_variant = decoupled_variants[-1] if (
            args.variant == "all_with_decoupled"
        ) else "object_relation_visibility"
        suffix = r" \\ \midrule" if variant == last_fair_variant else r" \\"
        print(
            f"{name} & {ctx} & "
            f"{r['dyn_rollout_mse']:.4f} & "
            f"{r['action_mse']:.4f} & "
            f"{r['collision_f1']:.4f} & "
            f"{r['rare_ade']:.4f} & "
            f"{r['interaction_recall_at_1m']:.4f}{suffix}"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    output_path = Path(args.output_dir) / "table3_complete.json"
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nComplete results saved to {output_path}")


def main():
    args = parse_args()
    
    if args.variant in ("all", "all_with_decoupled"):
        run_all_variants(args)
    else:
        run_stage0_experiment(args)


if __name__ == "__main__":
    main()
