#!/usr/bin/env python
"""Plot per-variant slot-type distribution (P3.2).

Reads the pickle produced by ``extract_slot_selections.py`` and, for
every variant, counts how many of its 16 slots are spent on each
TokenType (EGO / VEHICLE / PEDESTRIAN / CYCLIST / MAP / SIGNAL /
RELATION). Output:

  * One stacked bar chart per variant  -> comparison panel
  * A compact summary table printed to stdout

The story this figure tells:
  - Object-only-16 spends all 16 slots on EGO + VEH/PED/CYC (by design;
    no relation tokens in its type pool).
  - Object+Relation-16 (naive) gets pulled towards RELATION: a large
    chunk of its 16-slot budget is claimed by relation tokens, leaving
    fewer slots for actual dynamic agents -> explains the downstream
    catastrophic Rare ADE / IntRec@1m numbers.
  - Obj+Rel-Decoupled keeps its dyn path clean (12 dyn slots) and its
    rel path separate (4 rel slots) -> no competition.
  - Holistic-16Slot uses learned queries (set-prediction) so slot type
    is undefined; it's included with a special "mixed/learned" bar.
  - Holistic-full (97 tokens, reference upper bound) simply shows the
    raw type composition of the scene.
"""
from __future__ import annotations

import argparse
import pickle
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# TokenType enum (mirror of doorrl.schema.TokenType to avoid import).
_TYPE_NAMES = {
    0: "EGO",
    1: "VEH",
    2: "PED",
    3: "CYC",
    4: "MAP",
    5: "SIG",
    6: "REL",
    7: "PAD",
}
_TYPE_ORDER = [0, 1, 2, 3, 4, 5, 6]  # drop PAD
_TYPE_COLORS = {
    0: "#d62728",   # EGO - red
    1: "#1f77b4",   # VEH - blue
    2: "#2ca02c",   # PED - green
    3: "#ff7f0e",   # CYC - orange
    4: "#9467bd",   # MAP - purple
    5: "#8c564b",   # SIG - brown
    6: "#e377c2",   # REL - pink
}


_VARIANT_DISPLAY = [
    ("holistic_16slot", "Holistic-16Slot\n(learned queries,\ntype undefined)"),
    ("object_only", "Object-only-16"),
    ("object_relation", "Object+Relation-16\n(naive mixing)"),
    ("object_relation_visibility", "Obj+Rel+Vis-16\n(naive + vis)"),
    ("object_relation_decoupled", "Obj+Rel-Decoupled\n(12 dyn + 4 rel)"),
    ("object_relation_decoupled_visibility", "Decoupled+Vis"),
    ("holistic", "Holistic-full\n(ref, 97 tok)"),
]


def _count_slot_types(samples: list[dict]) -> tuple[dict[int, float], int, bool]:
    """Return (mean slots per TokenType, total slots, is_set_prediction)."""
    if not samples:
        return {}, 0, False
    is_set_pred = bool(samples[0].get("is_set_prediction", False))

    # For set-prediction variants (Holistic-16Slot) the "slot types" are
    # undefined because learned queries attend over all tokens. We label
    # the whole budget as "learned/mixed" for display.
    if is_set_pred:
        k = samples[0]["selected_indices"].numel()
        return {"__learned__": float(k)}, k, True

    totals: Counter = Counter()
    n_samples = 0
    total_slots = 0
    for s in samples:
        sel = s["selected_indices"]          # [K]
        mask = s["selected_mask"].bool()     # [K]
        tt = s["token_types"]                # [S]
        # Gather the TokenType of each selected slot.
        sel_types = tt.gather(0, sel.long())
        for t, m in zip(sel_types.tolist(), mask.tolist()):
            if not m:
                continue
            totals[int(t)] += 1
        n_samples += 1
        total_slots = max(total_slots, sel.numel())

    if n_samples == 0:
        return {}, 0, False
    mean_per_sample = {t: totals[t] / n_samples for t in totals}
    return mean_per_sample, total_slots, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="experiments/figures/slot_selections_seed7.pkl",
    )
    parser.add_argument(
        "--out",
        default="experiments/figures/stage0_slot_distribution.png",
    )
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)

    variants_data = payload["variants"]

    # Collect counts.
    stats = {}
    for vid, _ in _VARIANT_DISPLAY:
        if vid not in variants_data:
            stats[vid] = (None, 0, False)
            continue
        stats[vid] = _count_slot_types(variants_data[vid])

    # Print summary.
    print("Average slot-type composition (per val sample):")
    print(f"{'variant':<40} {'K':<4} " + " ".join(
        f"{_TYPE_NAMES[t]:>4}" for t in _TYPE_ORDER
    ))
    print("-" * 90)
    for vid, display in _VARIANT_DISPLAY:
        counts, k, is_set = stats[vid]
        if counts is None:
            continue
        if is_set:
            row = [f"{k:.1f}" if t == 0 else "  - " for t in _TYPE_ORDER]
        else:
            row = [f"{counts.get(t, 0.0):>4.1f}" for t in _TYPE_ORDER]
        clean_name = display.replace("\n", " / ")[:38]
        print(f"{clean_name:<40} {k:<4} " + " ".join(row))

    # --- Stacked bar chart ---
    fig, ax = plt.subplots(figsize=(13, 5.2))
    xs = np.arange(len(_VARIANT_DISPLAY))
    labels = [d for _, d in _VARIANT_DISPLAY]

    bottoms = np.zeros(len(_VARIANT_DISPLAY))
    # First, draw "learned/mixed" bars where applicable (single grey block).
    learned_heights = []
    learned_mask = np.zeros(len(_VARIANT_DISPLAY), dtype=bool)
    for i, (vid, _) in enumerate(_VARIANT_DISPLAY):
        counts, k, is_set = stats[vid]
        if is_set and counts is not None:
            learned_heights.append(k)
            learned_mask[i] = True
        else:
            learned_heights.append(0.0)
    if learned_mask.any():
        ax.bar(
            xs, learned_heights, color="#777777", edgecolor="black",
            linewidth=0.6, label="learned/mixed (set-pred)", hatch="xx",
            alpha=0.85,
        )

    # Stack regular type counts (skip set-pred variants).
    for t in _TYPE_ORDER:
        heights = []
        for vid, _ in _VARIANT_DISPLAY:
            counts, k, is_set = stats[vid]
            if counts is None or is_set:
                heights.append(0.0)
            else:
                heights.append(counts.get(t, 0.0))
        heights = np.array(heights)
        ax.bar(
            xs, heights, bottom=bottoms, color=_TYPE_COLORS[t],
            edgecolor="black", linewidth=0.5,
            label=_TYPE_NAMES[t], alpha=0.9,
        )
        # Value labels inside each segment (only if >= 0.3 slot).
        for x, h, b in zip(xs, heights, bottoms):
            if h >= 0.3:
                ax.text(
                    x, b + h / 2, f"{h:.1f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if t in (0, 1, 4, 5) else "black",
                )
        bottoms = bottoms + heights

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("avg #slots per sample", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Draw a horizontal guide at K=16.
    ax.axhline(16, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(
        len(_VARIANT_DISPLAY) - 0.5, 16.1, "K=16 (budget)",
        ha="right", va="bottom", fontsize=8, color="black",
    )

    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=8, fontsize=9, frameon=False,
    )
    fig.suptitle(
        "Stage 0: per-variant slot-type composition (seed 7, 128 val samples)\n"
        "Naive mixing (red/orange labels) bleeds REL tokens into the 16-slot budget; "
        "decoupled path keeps typed budgets.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
