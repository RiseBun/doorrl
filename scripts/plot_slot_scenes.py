#!/usr/bin/env python
"""Scene-level slot visualisation + automatic case selection (P3.3 + P3.4).

Reads ``extract_slot_selections.py``'s pickle and, for a small
user-chosen set of val samples, draws an ego-centric bird-eye view
comparing four variants side-by-side:

    Object-only-16  |  Object+Relation-16 (naive)  |  Obj+Rel-Decoupled  |  Holistic-full (ref)

In each panel:
    * all dynamic GT agents (veh/ped/cyc) in the current frame are
      plotted as small dots colour-coded by TokenType;
    * tokens the variant's abstraction *selected* are drawn with a
      thick black edge + larger marker;
    * the ego is fixed at the origin (red star);
    * GT ground-truth next-frame position is drawn as a faint arrow
      from current -> next (so the reader can see "motion target");
    * a short caption lists: # dyn agents < 10 m from ego,
      # of those that each variant actually selected, and avg
      nearest-prediction error on those near-field agents.

Case selection (P3.4): by default we rank val samples by the metric
"#near-field dyn agents SELECTED by Decoupled minus #SELECTED by Naive".
The top K differentiating samples are exported. You can also pass
``--sample-indices`` to override and pick specific indices.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch


_TYPE_NAMES = {0: "EGO", 1: "VEH", 2: "PED", 3: "CYC", 4: "MAP",
               5: "SIG", 6: "REL", 7: "PAD"}
_TYPE_COLOR = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e",
               4: "#9467bd", 5: "#8c564b", 6: "#e377c2"}
_DYN_TYPES = (0, 1, 2, 3)

# 4-panel comparison (keep holistic-full as reference/upper-bound).
_PANEL_VARIANTS = [
    ("object_only", "Object-only-16"),
    ("object_relation", "Object+Relation-16\n(naive)"),
    ("object_relation_decoupled", "Obj+Rel-Decoupled\n(Ours, 12 dyn + 4 rel)"),
    ("holistic", "Holistic-full\n(ref, 97 tok)"),
]


def _to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _count_near_selected(sample: dict, near_r: float) -> int:
    """#dyn GT agents within `near_r` m from ego AND selected by this variant."""
    tokens = sample["tokens"]
    types = sample["token_types"]
    mask = sample["token_mask"]
    sel = sample["selected_indices"]
    sel_mask = sample["selected_mask"]
    is_set_pred = sample.get("is_set_prediction", False)
    if is_set_pred:
        return -1  # undefined for learned queries

    xy = tokens[:, :2]
    dists = xy.norm(dim=-1)

    dyn_mask = torch.zeros_like(mask)
    for t in _DYN_TYPES:
        dyn_mask |= (types == t)
    near_dyn_mask = dyn_mask & mask & (dists <= near_r)
    near_dyn_set = set(torch.nonzero(near_dyn_mask, as_tuple=False).flatten().tolist())
    near_dyn_set.discard(0)  # exclude ego itself

    selected = set()
    for s_idx, m in zip(sel.tolist(), sel_mask.tolist()):
        if m and s_idx in near_dyn_set:
            selected.add(s_idx)
    return len(selected)


def _near_field_agent_count(sample: dict, near_r: float) -> int:
    tokens = sample["tokens"]
    types = sample["token_types"]
    mask = sample["token_mask"]
    xy = tokens[:, :2]
    dists = xy.norm(dim=-1)
    dyn_mask = torch.zeros_like(mask)
    for t in _DYN_TYPES:
        dyn_mask |= (types == t)
    near_dyn_mask = dyn_mask & mask & (dists <= near_r) & (dists > 0.01)
    return int(near_dyn_mask.sum().item())


def _nearest_pred_err(sample: dict, near_r: float) -> float:
    """Avg of (min-distance) between each near-field GT agent and any selected-slot prediction."""
    tokens = sample["tokens"]
    types = sample["token_types"]
    mask = sample["token_mask"]
    next_tokens = sample["next_tokens"]
    pred_next = sample["predicted_next_tokens"]  # [K, D]
    sel = sample["selected_indices"]
    sel_mask = sample["selected_mask"]

    # near-field dyn GT agents at next frame
    xy_now = tokens[:, :2]
    dyn_mask = torch.zeros_like(mask)
    for t in _DYN_TYPES:
        dyn_mask |= (types == t)
    gt_sel = dyn_mask & mask & (xy_now.norm(dim=-1) <= near_r)
    gt_sel[0] = False  # exclude ego
    if gt_sel.sum() == 0:
        return float("nan")

    gt_next_xy = next_tokens[gt_sel][:, :2]     # [N_near, 2]
    pred_xy = pred_next[:, :2]                  # [K, 2]
    valid = sel_mask.bool()
    if valid.sum() == 0:
        return float("nan")
    pred_xy = pred_xy[valid]                    # [K_valid, 2]

    # pairwise distances
    diff = gt_next_xy.unsqueeze(1) - pred_xy.unsqueeze(0)
    dist = diff.norm(dim=-1)
    min_per_gt = dist.min(dim=-1).values
    return float(min_per_gt.mean().item())


def _pick_cases(
    variants_data: Dict[str, List[dict]],
    near_r: float,
    top_k: int,
    min_agents: int,
) -> List[Tuple[int, int, dict]]:
    """Return list of (global_idx_in_flat_list, naive_vs_dec_score, info)."""
    naive = variants_data.get("object_relation")
    dec = variants_data.get("object_relation_decoupled")
    if naive is None or dec is None:
        raise RuntimeError("need object_relation and object_relation_decoupled samples")

    # Samples are aligned (extracted with the same val loader order), so
    # index i in naive corresponds to index i in dec.
    assert len(naive) == len(dec)

    ranked = []
    for i, (n_s, d_s) in enumerate(zip(naive, dec)):
        n_near = _near_field_agent_count(n_s, near_r)
        if n_near < min_agents:
            continue
        n_sel = _count_near_selected(n_s, near_r)
        d_sel = _count_near_selected(d_s, near_r)
        score = d_sel - n_sel
        ranked.append((
            i, score, {
                "n_near": n_near,
                "naive_sel": n_sel,
                "dec_sel": d_sel,
                "naive_err": _nearest_pred_err(n_s, near_r),
                "dec_err":   _nearest_pred_err(d_s, near_r),
            },
        ))

    ranked.sort(key=lambda x: -x[1])  # largest advantage for decoupled first
    return ranked[:top_k]


def _draw_one_panel(ax, sample: dict, near_r: float, display: str):
    tokens = _to_np(sample["tokens"])
    types = _to_np(sample["token_types"])
    mask = _to_np(sample["token_mask"].bool())
    next_tokens = _to_np(sample["next_tokens"])
    sel_idx = set(_to_np(sample["selected_indices"]).tolist())
    sel_mask_arr = _to_np(sample["selected_mask"].bool())
    is_set_pred = sample.get("is_set_prediction", False)
    # For set-pred variants, "selected tokens" refer to learned queries,
    # not raw token indices. We can't highlight raw tokens for them; just
    # show the scene + a note.
    if is_set_pred:
        valid_sel_idx = set()
    else:
        valid_sel_idx = {
            int(i) for i, m in zip(
                _to_np(sample["selected_indices"]).tolist(),
                sel_mask_arr.tolist(),
            ) if m
        }

    # Draw near-field circle.
    circ = plt.Circle((0, 0), near_r, color="black", fill=False,
                      linestyle=":", linewidth=0.6, alpha=0.4)
    ax.add_patch(circ)
    ax.text(
        near_r * 0.72, near_r * 0.72, f"{int(near_r)} m",
        fontsize=7, color="gray", alpha=0.6,
    )

    # Plot all tokens.
    handles = {}
    for i in range(len(mask)):
        if not mask[i]:
            continue
        t = int(types[i])
        if t == 7:
            continue
        x, y = tokens[i, 0], tokens[i, 1]
        is_sel = (i in valid_sel_idx)
        marker = "*" if t == 0 else "o"
        base_size = 220 if t == 0 else (70 if is_sel else 22)
        edgecolor = "black" if is_sel else (
            _TYPE_COLOR.get(t, "#444") if t != 0 else "#8b0000"
        )
        lw = 1.8 if is_sel else 0.5
        face = _TYPE_COLOR.get(t, "#888")
        alpha = 1.0 if is_sel or t == 0 else 0.55
        sc = ax.scatter(
            x, y, s=base_size, c=face, edgecolors=edgecolor,
            linewidths=lw, marker=marker, alpha=alpha, zorder=3,
        )
        if _TYPE_NAMES.get(t, "?") not in handles:
            handles[_TYPE_NAMES[t]] = sc
        # GT next-frame arrow for dynamic agents.
        if t in _DYN_TYPES and t != 0:
            nx, ny = next_tokens[i, 0], next_tokens[i, 1]
            ax.annotate(
                "", xy=(nx, ny), xytext=(x, y),
                arrowprops=dict(
                    arrowstyle="->", lw=0.7,
                    color=_TYPE_COLOR.get(t, "#444"),
                    alpha=0.45,
                ),
                zorder=2,
            )

    # Limits, aspect, grid.
    ax.set_xlim(-near_r * 1.8, near_r * 1.8)
    ax.set_ylim(-near_r * 1.8, near_r * 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color="#cccccc", lw=0.5, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.5, zorder=0)
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_title(display, fontsize=9.5)
    ax.tick_params(labelsize=7)

    if is_set_pred:
        ax.text(
            0.02, 0.02,
            "(set-pred, learned queries\ncannot be traced to raw tokens)",
            transform=ax.transAxes, fontsize=7, color="gray",
            va="bottom", ha="left",
        )


def _add_variant_legend(fig):
    patches = [
        mpatches.Patch(color=c, label=_TYPE_NAMES[t])
        for t, c in _TYPE_COLOR.items()
    ]
    # Plus: black edge = selected
    sel_marker = plt.Line2D(
        [0], [0], marker="o", color="w",
        markerfacecolor="lightgray", markeredgecolor="black",
        markersize=10, markeredgewidth=1.8,
        label="selected by abstraction",
    )
    fig.legend(
        handles=patches + [sel_marker],
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=len(patches) + 1, fontsize=9, frameon=False,
    )


def _plot_one_case(
    variants_data: Dict[str, List[dict]],
    global_idx: int,
    near_r: float,
    out_path: Path,
    case_info: dict,
    header: str,
):
    fig, axes = plt.subplots(1, len(_PANEL_VARIANTS), figsize=(4.2 * len(_PANEL_VARIANTS), 4.6))
    for ax, (vid, display) in zip(axes, _PANEL_VARIANTS):
        samples = variants_data.get(vid)
        if samples is None or global_idx >= len(samples):
            ax.set_title(f"{display}\n(missing)")
            ax.axis("off")
            continue
        sample = samples[global_idx]
        n_sel = _count_near_selected(sample, near_r)
        err = _nearest_pred_err(sample, near_r)
        if n_sel >= 0:
            footer = f"near-sel={n_sel}/{case_info['n_near']}  err={err:.2f} m"
        else:
            footer = f"err={err:.2f} m (set-pred)"
        _draw_one_panel(ax, sample, near_r, display + "\n" + footer)
    _add_variant_legend(fig)
    fig.suptitle(header, fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        default="experiments/figures/slot_selections_seed7.pkl",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/figures/scenes",
    )
    parser.add_argument("--near-r", type=float, default=15.0,
                        help="Near-field radius in metres.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-agents", type=int, default=4)
    parser.add_argument("--sample-indices", type=int, nargs="*", default=None,
                        help="If given, override auto-selection.")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    variants_data = payload["variants"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_indices:
        cases = []
        for idx in args.sample_indices:
            n_near = _near_field_agent_count(
                variants_data["object_relation"][idx], args.near_r
            )
            cases.append((idx, 0, {
                "n_near": n_near,
                "naive_sel": _count_near_selected(
                    variants_data["object_relation"][idx], args.near_r
                ),
                "dec_sel": _count_near_selected(
                    variants_data["object_relation_decoupled"][idx], args.near_r
                ),
                "naive_err": _nearest_pred_err(
                    variants_data["object_relation"][idx], args.near_r
                ),
                "dec_err": _nearest_pred_err(
                    variants_data["object_relation_decoupled"][idx], args.near_r
                ),
            }))
    else:
        cases = _pick_cases(
            variants_data, near_r=args.near_r,
            top_k=args.top_k, min_agents=args.min_agents,
        )

    print(f"Selected {len(cases)} cases "
          f"(ranked by decoupled-minus-naive near-field selection advantage):")
    print(f"{'idx':<6}{'#near':<8}{'naive_sel':<11}{'dec_sel':<9}"
          f"{'naive_err':<12}{'dec_err':<10}")
    print("-" * 62)
    for idx, score, info in cases:
        print(
            f"{idx:<6}{info['n_near']:<8}{info['naive_sel']:<11}"
            f"{info['dec_sel']:<9}{info['naive_err']:<12.2f}"
            f"{info['dec_err']:<10.2f}"
        )

    for rank, (idx, score, info) in enumerate(cases):
        out_path = out_dir / f"case_{rank:02d}_idx{idx}.png"
        header = (
            f"Case {rank} (sample idx {idx}):  "
            f"{info['n_near']} near-field dyn agents (<{int(args.near_r)} m).  "
            f"naive captured {info['naive_sel']}, decoupled captured {info['dec_sel']};  "
            f"near-field pred err  naive={info['naive_err']:.2f} m  "
            f"vs decoupled={info['dec_err']:.2f} m"
        )
        _plot_one_case(variants_data, idx, args.near_r, out_path, info, header)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
