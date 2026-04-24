from __future__ import annotations

from typing import Any, Dict, Mapping

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)


class NuPlanClosedLoopAdapter:
    def __init__(self, spec: TokenizationSpec, reactive: bool = True) -> None:
        self.spec = spec
        self.reactive = reactive
        self.converter = NormalizedSceneConverter(spec)

    @property
    def mode(self) -> BenchmarkMode:
        return (
            BenchmarkMode.CLOSED_LOOP_REACTIVE
            if self.reactive
            else BenchmarkMode.CLOSED_LOOP_NON_REACTIVE
        )

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="nuplan",
            mode=self.mode,
            purpose="Primary closed-loop benchmark for reactive vs non-reactive driving evaluation.",
            expected_inputs=[
                "planner observation",
                "tracked objects",
                "map context",
                "ego command or trajectory target",
            ],
            outputs=[
                "oracle scene tokens",
                "closed-loop metrics",
                "reactive/non-reactive experiment tags",
            ],
        )

    def build_scene_item_from_normalized(
        self,
        normalized_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self.converter.build_scene_item(normalized_record)

    def supported_experiments(self) -> Dict[str, str]:
        return {
            "replay_train_replay_test": "Train and test in non-reactive mode.",
            "replay_train_reactive_test": "Train in non-reactive mode, test in reactive mode.",
            "reactive_train_reactive_test": "Train and test in reactive mode.",
        }

    def convert_nuplan_observation(self, observation: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "Hook this method to the nuPlan devkit observation object and convert it "
            "into the normalized scene schema used by build_scene_item_from_normalized()."
        )
