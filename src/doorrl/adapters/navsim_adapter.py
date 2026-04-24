from __future__ import annotations

from typing import Any, Dict, Mapping

from doorrl.adapters.base import (
    AdapterDescription,
    BenchmarkMode,
    NormalizedSceneConverter,
    TokenizationSpec,
)


class NAVSIMEvaluationAdapter:
    def __init__(self, spec: TokenizationSpec) -> None:
        self.spec = spec
        self.converter = NormalizedSceneConverter(spec)

    def describe(self) -> AdapterDescription:
        return AdapterDescription(
            name="navsim",
            mode=BenchmarkMode.EXTERNAL_EVALUATION,
            purpose="External non-reactive transfer evaluation on log-derived scenarios.",
            expected_inputs=[
                "oracle or benchmark-provided scene state",
                "map context",
                "evaluation trajectory or action target",
            ],
            outputs=[
                "scene tokens",
                "transfer metrics",
                "external evaluation logs",
            ],
        )

    def build_scene_item_from_normalized(
        self,
        normalized_record: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return self.converter.build_scene_item(normalized_record)

    def convert_navsim_sample(self, sample: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "Hook this method to NAVSIM sample objects and convert them into the "
            "normalized scene schema used by build_scene_item_from_normalized()."
        )
