"""真实驾驶数据集 - 基于nuScenes/nuPlan的真实数据"""
from __future__ import annotations

import hashlib
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
from torch.utils.data import Dataset

from doorrl.adapters.base import TokenizationSpec
from doorrl.config import DoorRLConfig


class RealDrivingDataset(Dataset):
    """
    真实驾驶数据集
    
    支持:
    - nuScenes离线token化
    - nuPlan闭环评估
    - 自定义场景采样策略
    """
    
    def __init__(
        self,
        config: DoorRLConfig,
        data_source: str = 'nuscenes',
        data_root: str = '',
        scenes: Optional[List[str]] = None,
        seed: int = 7,
    ) -> None:
        self.config = config
        self.data_source = data_source
        self.seed = seed
        
        # 创建tokenization spec
        self.spec = TokenizationSpec(
            raw_dim=config.model.raw_dim,
            max_tokens=config.model.max_tokens,
            max_dynamic_objects=config.data.max_dynamic_objects,
            max_map_tokens=config.data.max_map_tokens,
            max_relation_tokens=config.data.max_relation_tokens,
            action_dim=config.model.action_dim,
        )
        
        # 初始化对应的adapter
        if data_source == 'nuscenes':
            from doorrl.adapters.nuscenes_real_adapter import NuScenesRealDataAdapter
            self.adapter = NuScenesRealDataAdapter(
                spec=self.spec,
                nuscenes_root=data_root,
                version='v1.0-trainval',
            )
            self._load_scenes(scenes)
        
        elif data_source == 'nuplan':
            from doorrl.adapters.nuplan_real_adapter import NuPlanRealDataAdapter
            self.adapter = NuPlanRealDataAdapter(
                spec=self.spec,
                nuplan_root=data_root,
                reactive=True,
            )
            self._load_scenes(scenes)
        
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
    
    def _load_scenes(self, scenes: Optional[List[str]] = None) -> None:
        """加载场景列表"""
        if self.data_source == 'nuscenes':
            if scenes is None:
                # 使用所有场景
                self.scenes = [s['name'] for s in self.adapter.nusc.scene[:10]]  # 前10个场景
            else:
                self.scenes = scenes
            
            # 收集所有样本
            self.samples = []
            for scene_name in self.scenes:
                try:
                    scene_samples = self.adapter.get_scene_samples(scene_name)
                    self.samples.extend(scene_samples)
                except ValueError:
                    print(f"Warning: Scene {scene_name} not found, skipping")
            
            print(f"Loaded {len(self.samples)} samples from {len(self.scenes)} scenes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        
        # 获取下一帧 (用于计算action和next_tokens)
        next_sample = None
        if index + 1 < len(self.samples):
            # 检查是否同一场景
            if self._get_sample_scene(sample) == self._get_sample_scene(self.samples[index + 1]):
                next_sample = self.samples[index + 1]
        
        # 转换为scene item
        scene_item = self.adapter.convert_sample_to_scene_item(
            sample=sample,
            next_sample=next_sample,
            compute_relations=True,
        )
        
        return scene_item
    
    def _get_sample_scene(self, sample: Dict[str, Any]) -> str:
        """获取样本所属场景名称"""
        for scene in self.adapter.nusc.scene:
            if scene['token'] == sample['scene_token']:
                return scene['name']
        return ''


class NuScenesSceneDataset(Dataset):
    """
    nuScenes场景数据集 - 按场景组织
    
    用途:
    - 世界模型预训练
    - 序列建模
    - 时序关系学习
    """
    
    def __init__(
        self,
        config: DoorRLConfig,
        nuscenes_root: str,
        scenes: Optional[List[str]] = None,
        version: str = 'v1.0-trainval',
        num_scenes: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.config = config

        spec = TokenizationSpec(
            raw_dim=config.model.raw_dim,
            max_tokens=config.model.max_tokens,
            max_dynamic_objects=config.data.max_dynamic_objects,
            max_map_tokens=config.data.max_map_tokens,
            max_relation_tokens=config.data.max_relation_tokens,
            action_dim=config.model.action_dim,
        )
        self._spec = spec
        # Cache key covers every input that affects the materialised tensors:
        # tokenisation spec + scene selection. Keep it short (first 16 hex chars
        # is collision-safe for our purposes).
        key_parts = (
            f"raw{spec.raw_dim}_max{spec.max_tokens}_dyn{spec.max_dynamic_objects}"
            f"_map{spec.max_map_tokens}_rel{spec.max_relation_tokens}_a{spec.action_dim}"
        )
        if scenes is not None:
            scene_sig = "scenes:" + ",".join(sorted(scenes))
        else:
            scene_sig = f"nscenes:{num_scenes}"
        self._cache_key = hashlib.sha1(
            f"{scene_sig}|{key_parts}|{version}".encode("utf-8")
        ).hexdigest()[:16]
        self._cache_file: Optional[str] = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._cache_file = os.path.join(
                cache_dir, f"nusc_tokens_{self._cache_key}.pt"
            )

        # Fast path: load pre-tokenised scenes from disk and skip the devkit
        # entirely. Saves ~18 min of single-threaded Python on nuScenes trainval
        # x 700 scenes for every downstream run after the first.
        if self._try_load_disk_cache():
            return

        from doorrl.adapters.nuscenes_real_adapter import NuScenesRealDataAdapter
        self.adapter = NuScenesRealDataAdapter(
            spec=spec,
            nuscenes_root=nuscenes_root,
            version=version,
        )

        # Scene selection rules:
        #   1) explicit `scenes` list wins;
        #   2) otherwise use `num_scenes` as a cap over all available scenes;
        #   3) otherwise fall back to the legacy "first 10" behaviour, which
        #      several lightweight test scripts rely on.
        if scenes is not None:
            self.scenes = list(scenes)
        elif num_scenes is not None:
            all_names = [s['name'] for s in self.adapter.nusc.scene]
            self.scenes = all_names[: max(0, num_scenes)]
        else:
            self.scenes = [s['name'] for s in self.adapter.nusc.scene[:10]]
        
        # 构建样本索引
        self.sample_index = []  # (scene_name, sample)
        for scene_name in self.scenes:
            try:
                samples = self.adapter.get_scene_samples(scene_name)
                for sample in samples:
                    self.sample_index.append((scene_name, sample))
            except ValueError:
                continue

        print(f"NuScenesSceneDataset: {len(self.sample_index)} samples from {len(self.scenes)} scenes")

        # ---------------------------------------------------------------
        # In-memory tokenization cache.
        #
        # Rationale: ``convert_sample_to_scene_item`` calls into nuscenes-devkit
        # (ego_pose lookup, box_velocity, per-box translate/rotate, relation
        # computation, CAN bus queries) for every __getitem__ invocation,
        # costing roughly 0.1-0.3 s per sample. With 700 scenes / 28k samples
        # and 15 epochs x 4 variants, this dominates wall-clock time and
        # starves the GPU (<1 % utilisation observed during a live run).
        #
        # Pre-materialising all scene items once at dataset construction time
        # turns __getitem__ into an O(1) list lookup. Memory footprint is
        # modest (~4 KB/sample -> ~110 MB for 28k samples), well within host
        # RAM on H20 nodes.
        # ---------------------------------------------------------------
        print(
            "NuScenesSceneDataset: pre-tokenising all samples into memory "
            "(one-time cost, avoids per-step nuscenes-devkit overhead)..."
        )
        self._cache: List[Dict[str, torch.Tensor]] = []
        # Parallel to ``self._cache`` so callers (e.g. a scene-level
        # train/val splitter) can map cache indices back to scene names
        # without re-tokenising. Entries added in lockstep with _cache below.
        self._cache_scene_names: List[str] = []
        total = len(self.sample_index)
        report_every = max(1, total // 20)
        for index, (scene_name, sample) in enumerate(self.sample_index):
            next_sample = None
            if index + 1 < total:
                next_scene_name, next_s = self.sample_index[index + 1]
                if next_scene_name == scene_name:
                    next_sample = next_s

            try:
                scene_item = self.adapter.convert_sample_to_scene_item(
                    sample=sample,
                    next_sample=next_sample,
                    compute_relations=True,
                )
            except Exception as e:
                # Mirror the lenient construction used during live indexing:
                # drop the sample rather than abort the whole dataset build.
                print(
                    f"NuScenesSceneDataset: skip sample {index} "
                    f"(scene={scene_name}): {e}"
                )
                continue

            self._cache.append(scene_item)
            self._cache_scene_names.append(scene_name)

            if (index + 1) % report_every == 0 or index + 1 == total:
                pct = 100.0 * (index + 1) / total
                print(
                    f"  tokenized {index + 1}/{total} ({pct:.1f}%)",
                    flush=True,
                )

        print(
            f"NuScenesSceneDataset: cache ready, "
            f"{len(self._cache)} usable samples "
            f"(dropped {total - len(self._cache)})"
        )
        self._try_save_disk_cache()

    def _try_load_disk_cache(self) -> bool:
        """Populate ``_cache`` / ``_cache_scene_names`` from disk if possible.

        Returns True iff the cache was hit and the object is fully initialised.
        """
        self._cache = []  # always ensure attrs exist for fallback path
        self._cache_scene_names = []
        self.sample_index = []
        if not self._cache_file or not os.path.isfile(self._cache_file):
            return False
        t0 = time.time()
        try:
            blob = torch.load(self._cache_file, map_location="cpu")
        except Exception as e:
            print(f"NuScenesSceneDataset: cache {self._cache_file} unreadable ({e}); re-tokenising.")
            return False
        try:
            self._cache = list(blob["cache"])
            self._cache_scene_names = list(blob["scene_names"])
            self.scenes = list(blob.get("scenes", []))
        except Exception as e:
            print(f"NuScenesSceneDataset: cache {self._cache_file} malformed ({e}); re-tokenising.")
            self._cache = []
            self._cache_scene_names = []
            return False
        # ``adapter`` is intentionally NOT materialised on the cache path: every
        # caller that needs tokenised data goes through __getitem__, and loading
        # nuscenes-devkit itself costs ~30 s.
        self.adapter = None  # type: ignore[assignment]
        print(
            f"NuScenesSceneDataset: loaded {len(self._cache)} samples "
            f"from cache {os.path.basename(self._cache_file)} in {time.time()-t0:.1f}s"
        )
        return True

    def _try_save_disk_cache(self) -> None:
        if not self._cache_file:
            return
        tmp = self._cache_file + ".tmp"
        try:
            torch.save(
                {
                    "cache": self._cache,
                    "scene_names": self._cache_scene_names,
                    "scenes": getattr(self, "scenes", []),
                    "key": self._cache_key,
                },
                tmp,
            )
            os.replace(tmp, self._cache_file)
            print(
                f"NuScenesSceneDataset: dumped cache -> "
                f"{self._cache_file} ({os.path.getsize(self._cache_file)/1e6:.1f} MB)"
            )
        except Exception as e:
            print(f"NuScenesSceneDataset: cache dump failed ({e}); continuing.")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._cache[index]

    # ------------------------------------------------------------------
    # Helpers for scene-level train/val splits (cf. ``_build_loaders`` in
    # ``run_stage0_table3.py``). Splitting at sample level leaks frames
    # from the same 20 s nuScenes scene across train/val and inflates
    # validation metrics; downstream code must use these helpers.
    # ------------------------------------------------------------------
    @property
    def cache_scene_names(self) -> List[str]:
        return self._cache_scene_names

    def indices_for_scenes(self, scene_names: Iterable[str]) -> List[int]:
        """Return cache indices whose scene name belongs to ``scene_names``."""
        wanted: Set[str] = set(scene_names)
        return [
            i for i, name in enumerate(self._cache_scene_names) if name in wanted
        ]
    
    def get_scene_sequence(self, scene_name: str) -> List[Dict[str, torch.Tensor]]:
        """获取完整场景序列"""
        samples = self.adapter.get_scene_samples(scene_name)
        sequence = []
        
        for sample in samples:
            scene_item = self.adapter.convert_sample_to_scene_item(
                sample=sample,
                compute_relations=True,
            )
            sequence.append(scene_item)
        
        return sequence
