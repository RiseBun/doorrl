#!/usr/bin/env python
"""Parallel pre-build of the NuScenesSceneDataset tokenisation cache.

The single-threaded tokenise loop in ``NuScenesSceneDataset.__init__`` takes
~18 min on trainval x 700 scenes because nuscenes-devkit is CPU-bound Python.
The machine has 192 cores sitting idle. This script fan-outs tokenisation
across N worker processes (each with its own NuScenes instance) and writes a
single cache file keyed identically to what ``NuScenesSceneDataset`` loads, so
subsequent runs start in seconds.

Usage::

    PYTHONPATH=src python scripts/build_token_cache.py \
        --config configs/debug_mvp.json \
        --nuscenes-root /mnt/datasets/e2e-nuscenes/20260302 \
        --num-scenes 700 --workers 16

It is safe to run this in parallel with an ongoing training job: each worker
loads its own read-only NuScenes instance into memory (~5 GB / worker) and
writes only to ``<cache_dir>/nusc_tokens_<sha1>.pt`` via an atomic rename.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

# Make the ``doorrl`` package importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.adapters.base import TokenizationSpec  # noqa: E402
from doorrl.config import DoorRLConfig  # noqa: E402


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------
def _tokenise_worker(
    scene_chunk: List[str],
    nuscenes_root: str,
    version: str,
    spec_dict: Dict[str, int],
    worker_id: int,
    n_workers: int,
) -> List[Tuple[str, Dict[str, torch.Tensor]]]:
    """Tokenise a subset of scenes with a private NuScenes adapter.

    Returns a list of ``(scene_name, scene_item)`` tuples preserving
    the input ordering so the parent can concatenate deterministically.
    """
    from doorrl.adapters.nuscenes_real_adapter import NuScenesRealDataAdapter

    spec = TokenizationSpec(**spec_dict)
    t0 = time.time()
    adapter = NuScenesRealDataAdapter(
        spec=spec,
        nuscenes_root=nuscenes_root,
        version=version,
    )
    print(
        f"[worker {worker_id}/{n_workers}] NuScenes loaded in "
        f"{time.time()-t0:.1f}s, {len(scene_chunk)} scenes to tokenise",
        flush=True,
    )

    out: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    t1 = time.time()
    for si, scene_name in enumerate(scene_chunk):
        try:
            samples = adapter.get_scene_samples(scene_name)
        except Exception as e:
            print(
                f"[worker {worker_id}] skip scene {scene_name}: "
                f"get_scene_samples failed ({e})",
                flush=True,
            )
            continue

        n = len(samples)
        for i, sample in enumerate(samples):
            next_sample = samples[i + 1] if i + 1 < n else None
            try:
                scene_item = adapter.convert_sample_to_scene_item(
                    sample=sample,
                    next_sample=next_sample,
                    compute_relations=True,
                )
            except Exception as e:
                print(
                    f"[worker {worker_id}] skip sample {i} in {scene_name}: {e}",
                    flush=True,
                )
                continue
            out.append((scene_name, scene_item))

        if (si + 1) % max(1, len(scene_chunk) // 10) == 0 or si + 1 == len(scene_chunk):
            print(
                f"[worker {worker_id}] {si+1}/{len(scene_chunk)} scenes "
                f"({100*(si+1)/len(scene_chunk):.0f}%), elapsed {time.time()-t1:.0f}s",
                flush=True,
            )

    print(
        f"[worker {worker_id}] done: {len(out)} samples, total "
        f"{time.time()-t0:.0f}s",
        flush=True,
    )
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def _compute_cache_key(
    num_scenes: int,
    scenes: List[str] | None,
    spec_dict: Dict[str, int],
    version: str,
) -> str:
    # Must match ``NuScenesSceneDataset.__init__`` key logic verbatim.
    key_parts = (
        f"raw{spec_dict['raw_dim']}_max{spec_dict['max_tokens']}"
        f"_dyn{spec_dict['max_dynamic_objects']}_map{spec_dict['max_map_tokens']}"
        f"_rel{spec_dict['max_relation_tokens']}_a{spec_dict['action_dim']}"
    )
    if scenes is not None:
        scene_sig = "scenes:" + ",".join(sorted(scenes))
    else:
        scene_sig = f"nscenes:{num_scenes}"
    return hashlib.sha1(
        f"{scene_sig}|{key_parts}|{version}".encode("utf-8")
    ).hexdigest()[:16]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--nuscenes-root", type=str, required=True)
    p.add_argument("--version", type=str, default="v1.0-trainval")
    p.add_argument("--num-scenes", type=int, default=700)
    p.add_argument(
        "--cache-dir", type=str,
        default=str(ROOT / "experiments" / "_token_cache"),
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--force", action="store_true",
        help="Rebuild cache even if a hit already exists.",
    )
    args = p.parse_args()

    config = DoorRLConfig.from_json(args.config)

    spec_dict = dict(
        raw_dim=config.model.raw_dim,
        max_tokens=config.model.max_tokens,
        max_dynamic_objects=config.data.max_dynamic_objects,
        max_map_tokens=config.data.max_map_tokens,
        max_relation_tokens=config.data.max_relation_tokens,
        action_dim=config.model.action_dim,
    )
    key = _compute_cache_key(
        args.num_scenes, None, spec_dict, args.version
    )
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_path = os.path.join(args.cache_dir, f"nusc_tokens_{key}.pt")

    if os.path.isfile(cache_path) and not args.force:
        print(f"Cache already exists: {cache_path}  ({os.path.getsize(cache_path)/1e6:.1f} MB). "
              f"Use --force to rebuild.")
        return

    # Determine the scene list with the SAME logic the dataset uses when only
    # ``num_scenes`` is passed: first N scene names from the devkit's scene
    # listing. We load NuScenes once here just to read the name list.
    print("Loading NuScenes table to enumerate scenes...", flush=True)
    t0 = time.time()
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.nuscenes_root, verbose=False)
    all_names = [s["name"] for s in nusc.scene]
    scenes = all_names[: max(0, args.num_scenes)]
    del nusc
    print(f"  enumerated {len(scenes)} scenes in {time.time()-t0:.1f}s", flush=True)

    # Partition scenes into roughly equal chunks, preserving input ordering.
    n_workers = max(1, min(args.workers, len(scenes)))
    chunks: List[List[str]] = [[] for _ in range(n_workers)]
    for i, s in enumerate(scenes):
        chunks[i % n_workers].append(s)
    chunk_sizes = [len(c) for c in chunks]
    print(
        f"Launching {n_workers} workers; chunk sizes {chunk_sizes[:5]}...",
        flush=True,
    )

    # Use 'spawn' to avoid forking a potentially large parent (we're often run
    # alongside another training job that has already loaded heavy modules).
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(n_workers) as pool:
        async_results = [
            pool.apply_async(
                _tokenise_worker,
                (chunks[i], args.nuscenes_root, args.version, spec_dict, i, n_workers),
            )
            for i in range(n_workers)
        ]
        pool.close()
        pool.join()
        worker_outputs = [r.get() for r in async_results]

    # Reassemble in the original scene order so sample_index is deterministic.
    # Since chunks were filled round-robin, re-interleave worker outputs by
    # scene name following `scenes` order.
    per_scene: Dict[str, List[Dict[str, torch.Tensor]]] = {s: [] for s in scenes}
    for out in worker_outputs:
        for scene_name, item in out:
            per_scene[scene_name].append(item)

    cache: List[Dict[str, torch.Tensor]] = []
    scene_names: List[str] = []
    for s in scenes:
        for item in per_scene[s]:
            cache.append(item)
            scene_names.append(s)

    print(
        f"All workers done in {time.time()-t0:.0f}s. Total samples: {len(cache)}",
        flush=True,
    )

    tmp = cache_path + ".tmp"
    torch.save(
        {
            "cache": cache,
            "scene_names": scene_names,
            "scenes": scenes,
            "key": key,
        },
        tmp,
    )
    os.replace(tmp, cache_path)
    size_mb = os.path.getsize(cache_path) / 1e6
    print(f"Wrote cache -> {cache_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
