#!/usr/bin/env python
"""Extract per-variant selected-slot metadata on the seed-7 val set.

For each model variant with a saved checkpoint under
``experiments/table3_fair_fix2_seed7/<variant>/model.pt``:
  1. Build the model, load weights, switch to eval.
  2. Run the model on a bounded number of val batches.
  3. Collect, per sample:
       - batch.tokens (all 97 raw tokens, [S, raw_dim])
       - batch.token_types, batch.token_mask
       - batch.next_tokens
       - abstraction.selected_indices  [K]
       - abstraction.selected_mask     [K]
       - abstraction.is_set_prediction (bool)
       - world_model.predicted_next_tokens [K, raw_dim]
  4. Dump to a pickle so plotting scripts can be re-run quickly
     without re-tokenising nuScenes.

The val loader is built with ``--num-scenes`` (default 100) rather than
the full 700 to keep runtime short (~2 min tokenisation). Since the
trained models are scene-agnostic this is statistically sufficient for
slot-type distribution and case-study visualisations.
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# Allow running from project root without installing the package.
_PROJ_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(_PROJ_ROOT / "src"))

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch


_VARIANTS = [
    "holistic_16slot",
    "object_only",
    "object_relation",
    "object_relation_visibility",
    "object_relation_decoupled",
    "object_relation_decoupled_visibility",
    "holistic",
]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_val_loader(args, config):
    """Mirror run_stage0_table3._build_loaders but return only val_loader."""
    dataset = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        num_scenes=args.num_scenes,
        version="v1.0-trainval",
    )
    scene_names_sorted = sorted(set(dataset.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(scene_names_sorted)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * n_total))) if n_total > 1 else 0
    n_train = n_total - n_val
    val_scenes = set(shuffled[n_train:])
    val_indices = dataset.indices_for_scenes(val_scenes)
    print(f"val scenes: {n_val}, val samples: {len(val_indices)}")
    return DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=0,
    )


def _extract_one_variant(
    variant_name: str,
    ckpt_path: Path,
    val_loader: DataLoader,
    config: DoorRLConfig,
    device: torch.device,
    max_batches: int,
) -> list[dict]:
    variant = ModelVariant(variant_name)
    model = DoorRLModelVariant(config.model, variant)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    out = []
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if bi >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            B = batch.tokens.size(0)
            is_set_pred = bool(getattr(
                output.abstraction, "is_set_prediction", False,
            ))
            for i in range(B):
                out.append({
                    "variant": variant_name,
                    "batch_idx": bi,
                    "sample_idx": i,
                    "tokens": batch.tokens[i].cpu(),
                    "next_tokens": batch.next_tokens[i].cpu(),
                    "token_types": batch.token_types[i].cpu(),
                    "token_mask": batch.token_mask[i].cpu(),
                    "selected_indices": output.abstraction.selected_indices[i].cpu(),
                    "selected_mask": output.abstraction.selected_mask[i].cpu(),
                    "predicted_next_tokens": output.world_model.predicted_next_tokens[i].cpu(),
                    "is_set_prediction": is_set_pred,
                })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/debug_mvp.json",
    )
    parser.add_argument(
        "--nuscenes-root", default="/mnt/datasets/e2e-nuscenes/20260302",
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="experiments/table3_fair_fix2_seed7",
        help=(
            "Directory containing <variant>/model.pt subfolders. "
            "Defaults to seed 7's fair run."
        ),
    )
    parser.add_argument(
        "--out",
        default="experiments/figures/slot_selections_seed7.pkl",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--num-scenes", type=int, default=100,
        help="Use a subset to keep tokenisation fast (~2 min).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-batches", type=int, default=8,
        help="Number of val batches to process per variant (8*16 = 128 samples).",
    )
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = DoorRLConfig.from_json(args.config)
    config.training.batch_size = args.batch_size
    config.seed = args.seed

    print("Building shared val loader ...")
    val_loader = _build_val_loader(args, config)

    out = {"variants": {}, "meta": {
        "seed": args.seed,
        "num_scenes": args.num_scenes,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "checkpoints_dir": args.checkpoints_dir,
    }}

    ck_dir = Path(args.checkpoints_dir)
    for variant_name in _VARIANTS:
        ckpt_path = ck_dir / variant_name / "model.pt"
        if not ckpt_path.exists():
            print(f"[skip] {variant_name}: no checkpoint at {ckpt_path}")
            continue
        print(f"[extract] {variant_name}")
        samples = _extract_one_variant(
            variant_name, ckpt_path, val_loader, config, device,
            args.max_batches,
        )
        out["variants"][variant_name] = samples
        print(f"  {len(samples)} samples captured")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
