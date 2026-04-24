#!/usr/bin/env python
"""Plot Stage 0 variance figures (P3.1).

Reads ``experiments/table3_fair_fix2_aggregate.json`` (produced by
``aggregate_fix2_seeds.py``) and renders bar-with-error-bar charts for
the three decision-oriented metrics that most directly separate the
variants:

    * Interaction Recall @ 1 m   (higher is better, stability-sensitive)
    * Rare ADE                   (lower is better, tail-sensitive)
    * Dyn Rollout MSE            (lower is better, full-scene forecast)

The plot intentionally overlays per-seed dots on top of the mean ±
std bars so the reader can see that e.g. naive Object+Relation's huge
error bar comes from genuine cross-seed collapse (two crashed seeds +
one lucky one), not from uniform noise. The decoupled variants should
show tight error bars *and* tight dot clouds.

Output: three PNG files under ``experiments/figures/`` (auto-created)
and one stacked summary figure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Canonical ordering + display names + hatch for "ours" highlighting.
_VARIANT_DISPLAY = [
    ("holistic_16slot", "Holistic-16Slot", False),
    ("object_only", "Object-only-16", False),
    ("object_relation", "Object+Relation-16\n(naive)", False),
    ("object_relation_visibility", "Obj+Rel+Vis-16", False),
    ("object_relation_decoupled", "Obj+Rel-Decoupled\n(Ours)", True),
    ("object_relation_decoupled_visibility", "Decoupled+Vis\n(Ours)", True),
    ("holistic", "Holistic-full\n(ref, 97 tok)", False),
]

_METRICS = [
    ("interaction_recall_at_1m", "Interaction Recall @ 1 m", True),
    ("rare_ade", "Rare ADE (m)", False),
    ("dyn_rollout_mse", "Dyn Rollout MSE", False),
]


def _load_aggregate(path: Path):
    payload = json.loads(path.read_text())
    return payload["metrics"]


def _plot_one_metric(ax, metrics, metric_key, metric_label, higher_is_better):
    xs = np.arange(len(_VARIANT_DISPLAY))
    means = []
    stds = []
    seeds = []
    labels = []
    colors = []
    for variant_id, display, is_ours in _VARIANT_DISPLAY:
        entry = metrics.get(variant_id, {}).get(metric_key, None)
        if entry is None:
            means.append(np.nan)
            stds.append(0.0)
            seeds.append([])
        else:
            means.append(entry["mean"])
            stds.append(entry["std"])
            seeds.append(entry.get("values", []))
        labels.append(display)
        if variant_id == "holistic":
            colors.append("#888888")
        elif is_ours:
            colors.append("#1f77b4")
        elif variant_id == "object_relation":
            colors.append("#d62728")
        else:
            colors.append("#c0c0c0")

    bars = ax.bar(
        xs, means, yerr=stds, capsize=5,
        color=colors, edgecolor="black", linewidth=0.7, alpha=0.85,
    )
    # Hatch the "ours" bars.
    for bar, (_, _, is_ours) in zip(bars, _VARIANT_DISPLAY):
        if is_ours:
            bar.set_hatch("//")
    # Per-seed dots (jittered).
    rng = np.random.default_rng(0)
    for x, vals in zip(xs, seeds):
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            color="black", s=18, zorder=3, alpha=0.8,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    suffix = " ↑" if higher_is_better else " ↓"
    ax.set_ylabel(metric_label + suffix, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    # Annotate mean on top of each bar.
    for x, m, s in zip(xs, means, stds):
        if np.isnan(m):
            continue
        ax.text(
            x, m + s + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
            f"{m:.2f}", ha="center", va="bottom", fontsize=7,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default="experiments/table3_fair_fix2_aggregate.json",
        help="Path to aggregate JSON produced by aggregate_fix2_seeds.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures",
        help="Directory to write PNG files into.",
    )
    args = parser.parse_args()

    agg_path = Path(args.aggregate)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_aggregate(agg_path)

    # Per-metric single panels.
    for key, label, higher in _METRICS:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        _plot_one_metric(ax, metrics, key, label, higher)
        fig.suptitle(
            f"Stage 0: {label} — 3 seeds (mean ± std, dots = per-seed values)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = out_dir / f"stage0_variance_{key}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_path}")

    # Stacked summary figure (3 panels).
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    for ax, (key, label, higher) in zip(axes, _METRICS):
        _plot_one_metric(ax, metrics, key, label, higher)
    fig.suptitle(
        "Stage 0: decision-oriented metrics — 3 seeds, mean ± std "
        "(dots = per-seed values)\n"
        "Hatched bars = ours (decoupled typed budgets). "
        "Naive mixing (red) shows huge error bars due to cross-seed collapse.",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = out_dir / "stage0_variance_summary.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
