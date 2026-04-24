"""Diagnostic: for each variant in seed 7, compute fraction of selected
slots that are dynamic-type vs relation-type vs map/signal/ego, averaged
over the val set. Also report number of dynamic slots per scene (avg/min).

Usage:
  python scripts/diagnose_selection.py --seed-dir experiments/table3_fair_fix2_seed7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
import random

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.config import DoorRLConfig
from doorrl.data.real_dataset import NuScenesSceneDataset
from doorrl.models.doorrl_variant import DoorRLModelVariant, ModelVariant
from doorrl.schema import SceneBatch, TokenType
from doorrl.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-dir", required=True, type=str)
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_mvp.json"))
    parser.add_argument("--nuscenes-root", default="/mnt/datasets/e2e-nuscenes/20260302")
    parser.add_argument("--num-scenes", type=int, default=700)
    parser.add_argument("--scene-val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = DoorRLConfig.from_json(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset (num_scenes={args.num_scenes})...")
    full = NuScenesSceneDataset(
        config=config,
        nuscenes_root=args.nuscenes_root,
        num_scenes=args.num_scenes,
        version="v1.0-trainval",
    )
    scene_names_sorted = sorted(set(full.cache_scene_names))
    rng = random.Random(args.seed)
    shuffled = list(scene_names_sorted)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = max(1, int(round(args.scene_val_ratio * n_total)))
    n_train = n_total - n_val
    val_scenes = set(shuffled[n_train:])
    val_indices = full.indices_for_scenes(val_scenes)
    val_loader = DataLoader(
        Subset(full, val_indices),
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=SceneBatch.collate,
        num_workers=0,
    )

    seed_dir = Path(args.seed_dir)
    variants_to_check = [
        "object_only",
        "object_relation",
        "object_relation_visibility",
        "holistic",
    ]

    print(f"\n{'Variant':<28} | {'%dyn':>6} | {'%rel':>6} | {'%map':>6} | "
          f"{'%sig':>6} | {'%ego':>6} | {'avg dyn slots / 16':>18} | "
          f"{'min dyn slots':>13}")
    print("-" * 110)

    DYN = {int(TokenType.EGO), int(TokenType.VEHICLE),
           int(TokenType.PEDESTRIAN), int(TokenType.CYCLIST)}

    for variant in variants_to_check:
        model_path = seed_dir / variant / "model.pt"
        if not model_path.exists():
            print(f"  {variant}: NO model.pt, skip")
            continue
        try:
            mv = ModelVariant(variant)
        except ValueError:
            continue
        model = DoorRLModelVariant(config.model, mv).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        counts = {"ego": 0, "veh": 0, "ped": 0, "cyc": 0,
                  "map": 0, "sig": 0, "rel": 0, "pad": 0}
        n_dyn_per_scene = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                sel = out.abstraction.selected_indices  # [B, K]
                mask = out.abstraction.selected_mask.bool()
                if getattr(out.abstraction, "is_set_prediction", False):
                    n_dyn_per_scene.extend([16] * sel.size(0))
                    counts["veh"] += int(mask.sum().item())  # treat as dyn
                    continue
                sel_types = batch.token_types.gather(1, sel)  # [B, K]
                for i in range(sel.size(0)):
                    n_dyn = 0
                    for k in range(sel.size(1)):
                        if not mask[i, k]:
                            continue
                        t = int(sel_types[i, k].item())
                        if t == int(TokenType.EGO): counts["ego"] += 1; n_dyn += 1
                        elif t == int(TokenType.VEHICLE): counts["veh"] += 1; n_dyn += 1
                        elif t == int(TokenType.PEDESTRIAN): counts["ped"] += 1; n_dyn += 1
                        elif t == int(TokenType.CYCLIST): counts["cyc"] += 1; n_dyn += 1
                        elif t == int(TokenType.MAP): counts["map"] += 1
                        elif t == int(TokenType.SIGNAL): counts["sig"] += 1
                        elif t == int(TokenType.RELATION): counts["rel"] += 1
                        else: counts["pad"] += 1
                    n_dyn_per_scene.append(n_dyn)

        total = sum(counts.values()) or 1
        n_dyn = counts["ego"] + counts["veh"] + counts["ped"] + counts["cyc"]
        avg_dyn = sum(n_dyn_per_scene) / max(1, len(n_dyn_per_scene))
        min_dyn = min(n_dyn_per_scene) if n_dyn_per_scene else 0
        print(f"{variant:<28} | "
              f"{100*n_dyn/total:6.1f} | "
              f"{100*counts['rel']/total:6.1f} | "
              f"{100*counts['map']/total:6.1f} | "
              f"{100*counts['sig']/total:6.1f} | "
              f"{100*counts['ego']/total:6.1f} | "
              f"{avg_dyn:18.2f} | "
              f"{min_dyn:13d}")


if __name__ == "__main__":
    main()
