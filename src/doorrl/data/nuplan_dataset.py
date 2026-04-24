"""nuPlan preprocessed-NPZ scene dataset.

Parallel to ``NuScenesSceneDataset`` so Stage 0 / Stage 1 orchestrators
can treat the two data sources interchangeably. Key differences from
the nuScenes variant:

* Each NPZ is a *single* anchor frame (with 21-frame history baked in
  and 80-frame future baked in); there is no notion of "samples within
  a scene" -- so we rebind ``scene_name`` to the NPZ filename (e.g.
  ``sg-one-north_0027e967af8952bc``) for interface compatibility with
  the scene-level splitter. That makes every "scene" a single sample,
  which means there is no within-scene temporal leak to worry about.
  Whether a dataset-level train/val split is perfectly log-disjoint
  depends on the upstream preprocessing; we do not assert that here.
* Tokenisation is already done in preprocessing, so the in-memory
  cache step is basically "np.load -> convert -> stash". A single file
  is <50 KB and ``convert_npz_to_scene_item`` is pure-Python; even on
  a modest machine this pre-materialises at ~500 samples/s.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Set

import torch
from torch.utils.data import Dataset

from doorrl.adapters.base import TokenizationSpec
from doorrl.adapters.nuplan_preprocessed_adapter import NuPlanPreprocessedAdapter
from doorrl.config import DoorRLConfig


class NuPlanPreprocessedDataset(Dataset):
    """Loader for Diffusion-Planner style preprocessed nuPlan NPZ files.

    Parameters
    ----------
    config : DoorRLConfig
        Same config used by the nuScenes pipeline; we reuse its
        ``model.raw_dim``, ``data.max_*`` fields to build a
        TokenizationSpec.
    data_root : str
        Parent folder containing ``part_*/\*.npz``. Typically
        ``/mnt/datasets/e2e-nuplan/v1.1/processed_agent64_split``.
    num_samples : int, optional
        Cap on the number of NPZ files to load. ``None`` loads all.
    index_json : str, optional
        Path to a precomputed list of NPZ paths (e.g.
        ``diffusion_planner_agent64_train_paths.json``). If given, we
        read from that index instead of walking the filesystem; this
        is dramatically faster on a multi-hundred-K-file split.
    seed : int
        Shuffle seed used *before* the ``num_samples`` cap so different
        seeds see different subsets.
    """

    def __init__(
        self,
        config: DoorRLConfig,
        data_root: str,
        num_samples: Optional[int] = None,
        index_json: Optional[str] = None,
        seed: int = 7,
    ) -> None:
        self.config = config
        self.data_root = Path(data_root)
        self.seed = seed

        spec = TokenizationSpec(
            raw_dim=config.model.raw_dim,
            max_tokens=config.model.max_tokens,
            max_dynamic_objects=config.data.max_dynamic_objects,
            max_map_tokens=config.data.max_map_tokens,
            max_relation_tokens=config.data.max_relation_tokens,
            action_dim=config.model.action_dim,
        )
        self.adapter = NuPlanPreprocessedAdapter(
            spec=spec,
            data_root=str(self.data_root),
            max_neighbors=config.data.max_dynamic_objects,
        )

        # ---- file discovery ------------------------------------------
        self.paths = self._discover(index_json, num_samples)
        print(
            f"NuPlanPreprocessedDataset: will load {len(self.paths)} NPZ "
            f"samples from {self.data_root}"
        )

        # ---- in-memory tokenisation cache ---------------------------
        # Same rationale as NuScenesSceneDataset: pay the decode cost
        # once, then every epoch is a list-index fetch.
        self._cache: List[dict] = []
        self._cache_scene_names: List[str] = []
        total = len(self.paths)
        report_every = max(1, total // 20)
        for index, path in enumerate(self.paths):
            try:
                item = self.adapter.convert_npz_to_scene_item(
                    path, compute_relations=True,
                )
            except Exception as exc:
                print(f"NuPlanPreprocessedDataset: skip {path.name}: {exc}")
                continue
            self._cache.append(item)
            self._cache_scene_names.append(path.stem)  # see class docstring

            if (index + 1) % report_every == 0 or index + 1 == total:
                pct = 100.0 * (index + 1) / total
                print(
                    f"  tokenised {index + 1}/{total} ({pct:.1f}%)",
                    flush=True,
                )

        print(
            f"NuPlanPreprocessedDataset: cache ready, {len(self._cache)} "
            f"usable samples (dropped {total - len(self._cache)})"
        )

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------
    def _discover(
        self,
        index_json: Optional[str],
        num_samples: Optional[int],
    ) -> List[Path]:
        """Return the list of NPZ paths to load."""
        if index_json is not None:
            idx_path = Path(index_json)
            if not idx_path.is_absolute():
                idx_path = self.data_root / idx_path
            with open(idx_path, "r") as f:
                rel_paths = json.load(f)
            paths = [self.data_root / p for p in rel_paths]
        else:
            # Walk the filesystem. We deliberately stop at the first
            # ``num_samples`` hits of ``glob`` to avoid enumerating all
            # one million files when a user only asks for 5000.
            paths = []
            cap = num_samples or float("inf")
            for part_dir in sorted(self.data_root.glob("part_*")):
                if len(paths) >= cap:
                    break
                for npz in sorted(part_dir.glob("*.npz")):
                    paths.append(npz)
                    if len(paths) >= cap:
                        break

        # Keep order deterministic, then shuffle-then-cap so different
        # seeds don't all land on part_1.
        rng = random.Random(self.seed)
        paths = sorted(paths)
        rng.shuffle(paths)
        if num_samples is not None:
            paths = paths[: num_samples]
        return paths

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, index: int):
        return self._cache[index]

    # ------------------------------------------------------------------
    # Helpers that mirror NuScenesSceneDataset's public surface so the
    # orchestrator's ``_build_loaders`` can reuse the same split logic.
    # ------------------------------------------------------------------
    @property
    def cache_scene_names(self) -> List[str]:
        return self._cache_scene_names

    def indices_for_scenes(self, scene_names: Iterable[str]) -> List[int]:
        wanted: Set[str] = set(scene_names)
        return [
            i for i, name in enumerate(self._cache_scene_names) if name in wanted
        ]
