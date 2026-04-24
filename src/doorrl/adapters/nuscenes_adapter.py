from __future__ import annotations

from typing import Any, Dict, Mapping

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)


class NuScenesSceneTokenAdapter:
    def __init__(self, spec: TokenizationSpec) -> None:
        self.spec = spec
        self.converter = NormalizedSceneConverter(spec)

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="nuscenes",
            mode=BenchmarkMode.OFFLINE_DATASET,
            purpose="Offline tokenization and world-model pretraining from annotated scenes.",
            expected_inputs=[
                "ego state",
                "annotated objects",
                "map elements",
                "optional relation features",
            ],
            outputs=[
                "scene tokens",
                "token mask",
                "token types",
                "next-step token targets",
            ],
        )

    def build_scene_item(self, normalized_record: Mapping[str, Any]) -> Dict[str, Any]:
        return self.converter.build_scene_item(normalized_record)

    def expected_normalized_schema(self) -> Dict[str, str]:
        return {
            "ego": "Ego kinematics in ego-centric coordinates.",
            "objects": "List of annotated dynamic agents.",
            "map_elements": "Local topology or lane tokens.",
            "relations": "Decision-sufficient relation tokens.",
            "next_ego": "Next-step ego target.",
            "action": "Teacher or planner action embedding.",
            "reward": "Optional reward target for training.",
            "continue": "Episode continuation flag.",
        }
