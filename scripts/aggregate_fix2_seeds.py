"""Aggregate Fix #2 multi-seed Stage 0 results into mean ± std table.

Reads N ``table3_complete.json`` files (one per seed), each shaped:

    {
      "<variant>": {
        "dyn_rollout_mse": float,
        "action_mse": float,
        "collision_f1": float,
        "rare_ade": float,
        "interaction_recall_at_1m": float,
        ...
      },
      ...
    }

Emits the consolidated ``aggregate.json`` plus a printed mean ± std table
for the four metrics the paper highlights:
    Dyn Rollout, Coll F1, Rare ADE, IntRec@1m
(``Action MSE`` is included but de-emphasised per user's instruction.)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


_METRIC_ORDER = [
    ("dyn_rollout_mse", "Dyn Rollout", False, "lower is better"),
    ("action_mse", "Action MSE", False, "de-emphasized"),
    ("collision_f1", "Coll. F1", True, "higher is better"),
    ("rare_ade", "Rare ADE", False, "lower is better"),
    ("interaction_recall_at_1m", "IntRec@1m", True, "higher is better"),
]

_VARIANT_ORDER = [
    ("holistic_16slot", "Holistic-16Slot", 16),
    ("object_only", "Object-only-16", 16),
    ("object_relation", "Object+Relation-16", 16),
    ("object_relation_visibility", "Obj+Rel+Vis-16", 16),
    ("object_relation_decoupled", "Obj+Rel-Decoupled", 16),
    ("object_relation_decoupled_visibility", "Decoupled+Vis", 16),
    ("holistic", "Holistic-full (ref)", 97),
]


def _mean_std(xs: List[float]) -> Dict[str, float]:
    n = len(xs)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(xs) / n
    if n < 2:
        std = 0.0
    else:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std = math.sqrt(var)
    return {"mean": mean, "std": std, "n": n, "values": xs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+", required=True,
                        help="Paths to per-seed table3_complete.json files.")
    parser.add_argument("--out", type=str, required=True,
                        help="Where to write aggregated JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_seed: List[Dict] = []
    for p in args.runs:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(p)
        per_seed.append(json.loads(path.read_text()))

    aggregate: Dict[str, Dict[str, Dict[str, float]]] = {}
    for variant_key, _, _ in _VARIANT_ORDER:
        aggregate[variant_key] = {}
        for metric_key, _, _, _ in _METRIC_ORDER:
            xs: List[float] = []
            for seed_results in per_seed:
                if variant_key in seed_results and metric_key in seed_results[variant_key]:
                    val = seed_results[variant_key][metric_key]
                    if isinstance(val, (int, float)) and not math.isnan(val):
                        xs.append(float(val))
            aggregate[variant_key][metric_key] = _mean_std(xs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_seeds": len(per_seed),
        "source_files": [str(Path(p).resolve()) for p in args.runs],
        "metrics": aggregate,
    }, indent=2))
    print(f"\nAggregated results written to: {out_path}")

    print("\n" + "=" * 110)
    print(f"Fix #2 Multi-Seed Aggregate (n={len(per_seed)} seeds): mean \u00b1 std")
    print("=" * 110)
    header_cols = [f"{'Variant':<22}", f"{'Ctx':<4}"]
    for _, label, _, _ in _METRIC_ORDER:
        header_cols.append(f"{label:<22}")
    print(" | ".join(header_cols))
    print("-" * 130)

    for variant_key, label, ctx in _VARIANT_ORDER:
        row = [f"{label:<22}", f"{ctx:<4d}"]
        for metric_key, _, _, _ in _METRIC_ORDER:
            stat = aggregate[variant_key][metric_key]
            if stat["n"] == 0:
                cell = f"{'n/a':<22}"
            else:
                cell = f"{stat['mean']:.4f} \u00b1 {stat['std']:.4f}"
                cell = f"{cell:<22}"
            row.append(cell)
        print(" | ".join(row))
    print("-" * 130)

    print(
        "\nReporting note: paper narrative should emphasize "
        "Dyn Rollout, Coll F1, Rare ADE, IntRec@1m. "
        "Action MSE is reported for completeness but de-emphasised."
    )


if __name__ == "__main__":
    main()
