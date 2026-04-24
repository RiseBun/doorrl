from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doorrl.adapters import (
    BenchmarkMode,
    NAVSIMEvaluationAdapter,
    NuPlanClosedLoopAdapter,
    NuScenesSceneTokenAdapter,
    TokenizationSpec,
)
from doorrl.config import DoorRLConfig
from doorrl.schema import SceneBatch


def _normalized_record():
    return {
        "ego": {"x": 0.0, "y": 0.0, "speed": 5.0, "length": 4.5, "width": 1.8},
        "next_ego": {"x": 1.0, "y": 0.1, "speed": 5.2, "length": 4.5, "width": 1.8},
        "objects": [
            {
                "token_type": "vehicle",
                "x": 8.0,
                "y": 0.3,
                "vx": 4.0,
                "vy": 0.0,
                "length": 4.2,
                "width": 1.8,
                "risk": 0.2,
                "visibility": 1.0,
                "ttc": 3.5,
            }
        ],
        "map_elements": [{"x": 5.0, "y": 0.0, "priority": 1.0}],
        "relations": [
            {
                "x": 8.0,
                "y": 0.3,
                "distance": 8.01,
                "risk": 0.2,
                "visibility": 1.0,
                "lane_conflict": 1.0,
                "is_interactive": 1.0,
            }
        ],
        "action": [0.1, -0.2],
        "reward": 0.5,
        "continue": 1.0,
    }


def test_nuscenes_adapter_builds_scene_item() -> None:
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    spec = TokenizationSpec(
        raw_dim=config.model.raw_dim,
        max_tokens=config.model.max_tokens,
        max_dynamic_objects=config.data.max_dynamic_objects,
        max_map_tokens=config.data.max_map_tokens,
        max_relation_tokens=config.data.max_relation_tokens,
        action_dim=config.model.action_dim,
    )
    adapter = NuScenesSceneTokenAdapter(spec)
    item = adapter.build_scene_item(_normalized_record())
    batch = SceneBatch.collate([item, item])

    assert batch.tokens.shape == (2, config.model.max_tokens, config.model.raw_dim)
    assert batch.actions.shape == (2, config.model.action_dim)


def test_benchmark_adapters_report_expected_modes() -> None:
    config = DoorRLConfig.from_json(ROOT / "configs" / "debug_mvp.json")
    spec = TokenizationSpec(
        raw_dim=config.model.raw_dim,
        max_tokens=config.model.max_tokens,
        max_dynamic_objects=config.data.max_dynamic_objects,
        max_map_tokens=config.data.max_map_tokens,
        max_relation_tokens=config.data.max_relation_tokens,
        action_dim=config.model.action_dim,
    )
    nuplan = NuPlanClosedLoopAdapter(spec, reactive=True)
    navsim = NAVSIMEvaluationAdapter(spec)

    assert nuplan.describe().mode == BenchmarkMode.CLOSED_LOOP_REACTIVE
    assert navsim.describe().mode == BenchmarkMode.EXTERNAL_EVALUATION
