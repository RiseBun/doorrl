"""消融实验结果分析工具"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_experiment_results(exp_dir: str) -> Dict:
    """加载单个实验的结果"""
    exp_path = Path(exp_dir)
    
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    
    # 加载配置
    config_path = exp_path / "config.json"
    config = json.loads(config_path.read_text())
    
    # 加载训练历史
    history_path = exp_path / "history.json"
    history = json.loads(history_path.read_text())
    
    return {
        "config": config,
        "history": history,
        "variant": config.get("variant", "unknown"),
    }


def load_all_experiments(base_dir: str) -> List[Dict]:
    """加载所有实验结果"""
    base_path = Path(base_dir)
    experiments = []
    
    for exp_dir in sorted(base_path.iterdir()):
        if exp_dir.is_dir() and (exp_dir / "history.json").exists():
            try:
                exp_data = load_experiment_results(str(exp_dir))
                experiments.append(exp_data)
                print(f"✓ Loaded: {exp_data['variant']} from {exp_dir.name}")
            except Exception as e:
                print(f"✗ Failed to load {exp_dir.name}: {e}")
    
    return experiments


def plot_training_curves(experiments: List[Dict], save_path: str = None):
    """绘制训练曲线对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 定义要绘制的指标
    metrics = [
        ("total", "Total Loss"),
        ("obs", "Observation Loss"),
        ("reward", "Reward Loss"),
        ("collision", "Collision Loss"),
    ]
    
    # 定义颜色
    colors = {
        "holistic": "#FF6B6B",
        "object_only": "#4ECDC4",
        "object_relation": "#45B7D1",
        "object_relation_visibility": "#96CEB4",
    }
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        for exp in experiments:
            variant = exp["variant"]
            history = exp["history"]
            
            # 提取训练和验证损失
            train_values = [h[metric_key] for h in history.get("train", [])]
            val_values = [h[metric_key] for h in history.get("val", [])]
            
            epochs = list(range(1, len(train_values) + 1))
            color = colors.get(variant, "#000000")
            
            # 绘制曲线
            ax.plot(epochs, train_values, 
                   label=f"{variant} (train)", 
                   color=color, 
                   linestyle="-",
                   alpha=0.7)
            ax.plot(epochs, val_values, 
                   label=f"{variant} (val)", 
                   color=color, 
                   linestyle="--",
                   alpha=0.7)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def print_comparison_table(experiments: List[Dict]):
    """打印对比表格"""
    print("\n" + "="*80)
    print("DOOR-RL Ablation Study - Results Comparison")
    print("="*80)
    
    # 表头
    print(f"{'Variant':<35} {'Final Val Loss':<15} {'Final Obs':<15} {'Final Reward':<15}")
    print("-"*80)
    
    # 按variant排序
    variant_order = ["holistic", "object_only", "object_relation", "object_relation_visibility"]
    sorted_exps = sorted(
        experiments,
        key=lambda e: variant_order.index(e["variant"]) if e["variant"] in variant_order else 999
    )
    
    for exp in sorted_exps:
        variant = exp["variant"]
        history = exp["history"]
        
        if history.get("val"):
            final_val = history["val"][-1]
            total = final_val.get("total", 0.0)
            obs = final_val.get("obs", 0.0)
            reward = final_val.get("reward", 0.0)
            
            print(f"{variant:<35} {total:<15.4f} {obs:<15.4f} {reward:<15.4f}")
    
    print("="*80)


def generate_paper_table(experiments: List[Dict]):
    """生成论文格式的表格 (LaTeX)"""
    print("\n" + "="*80)
    print("LaTeX Table for Paper")
    print("="*80)
    
    latex_code = r"""
\begin{table}[t]
\centering
\caption{Ablation Study on Object-Relational Representations}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
\textbf{Model Variant} & \textbf{Val Loss} & \textbf{Obs Loss} & \textbf{Reward Loss} & \textbf{Collision Loss} \\
\midrule
"""
    
    # 按variant排序
    variant_names = {
        "holistic": "Holistic (Baseline)",
        "object_only": "Object-Only",
        "object_relation": "Object-Relation (Ours)",
        "object_relation_visibility": "Object-Relation + Visibility",
    }
    
    variant_order = ["holistic", "object_only", "object_relation", "object_relation_visibility"]
    sorted_exps = sorted(
        experiments,
        key=lambda e: variant_order.index(e["variant"]) if e["variant"] in variant_order else 999
    )
    
    for exp in sorted_exps:
        variant = exp["variant"]
        history = exp["history"]
        
        if history.get("val"):
            final_val = history["val"][-1]
            total = final_val.get("total", 0.0)
            obs = final_val.get("obs", 0.0)
            reward = final_val.get("reward", 0.0)
            collision = final_val.get("collision", 0.0)
            
            name = variant_names.get(variant, variant)
            latex_code += f"{name} & {total:.4f} & {obs:.4f} & {reward:.4f} & {collision:.4f} \\\\\n"
    
    latex_code += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    print(latex_code)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ablation study results.")
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="experiments/ablation",
        help="Base directory containing experiment results.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate training curve plots.",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Print comparison table.",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table for paper.",
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        help="Path to save the plot image.",
    )
    
    args = parser.parse_args()
    
    # 加载所有实验
    print(f"Loading experiments from {args.exp_dir}...")
    experiments = load_all_experiments(args.exp_dir)
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"\nLoaded {len(experiments)} experiments")
    
    # 打印对比表格
    if args.table or args.latex:
        print_comparison_table(experiments)
    
    # 生成LaTeX表格
    if args.latex:
        generate_paper_table(experiments)
    
    # 绘制训练曲线
    if args.plot:
        plot_training_curves(experiments, save_path=args.save_plot)


if __name__ == "__main__":
    main()
