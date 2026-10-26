from __future__ import annotations

from vlkg.core.transformers.time_fact_core_pipeline import (
    SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
    SECOND_MAINLINE_CONCEPTUAL_NAME,
    SECOND_MAINLINE_DISPLAY_NAME,
    SECOND_MAINLINE_PROFILE_ID,
    TIME_FACT_CORE_DERIVATION_STEPS,
    TIME_FACT_CORE_DESIGN_PRINCIPLES,
    TimeFactCoreGraphResult,
    build_time_fact_core_graph,
)


TopologyPrunedAfterSceneFactResult = TimeFactCoreGraphResult
build_topology_pruned_after_scene_fact_graph = build_time_fact_core_graph


__all__ = [
    "SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID",
    "SECOND_MAINLINE_CONCEPTUAL_NAME",
    "SECOND_MAINLINE_DISPLAY_NAME",
    "SECOND_MAINLINE_PROFILE_ID",
    "TIME_FACT_CORE_DERIVATION_STEPS",
    "TIME_FACT_CORE_DESIGN_PRINCIPLES",
    "TimeFactCoreGraphResult",
    "TopologyPrunedAfterSceneFactResult",
    "build_time_fact_core_graph",
    "build_topology_pruned_after_scene_fact_graph",
]
