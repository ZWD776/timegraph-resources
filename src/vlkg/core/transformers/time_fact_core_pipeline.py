from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.transformers.canonical_graph_scene_fact_span_compaction import (
    compact_scene_facts_by_span,
)
from vlkg.core.transformers.canonical_graph_stable_core_compaction import (
    compact_canonical_graph_to_stable_core,
)
from vlkg.core.transformers.canonical_graph_time_axis_compaction import (
    compact_canonical_graph_by_time_axis,
)
from vlkg.core.transformers.temporal_graph_to_canonical_graph import (
    build_canonical_temporal_memory_graph,
)
from vlkg.core.transformers.temporal_graph_to_stable_identity import StableIdentityMergeMode


@dataclass(frozen=True)
class TimeFactCoreGraphResult:
    graph: CanonicalTemporalMemoryGraph
    report: dict[str, Any]
    seed_graph: CanonicalTemporalMemoryGraph

    @property
    def base_canonical_graph(self) -> CanonicalTemporalMemoryGraph:
        """Backward-compatible alias for older tests and reports.

        The second mainline builds a direct time-fact seed graph instead of a
        full first-mainline base canonical graph.
        """

        return self.seed_graph


SECOND_MAINLINE_PROFILE_ID = "time_fact_core"
SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID = "topology_pruned_after_scene_fact"
SECOND_MAINLINE_CONCEPTUAL_NAME = "time_fact_core_graph"
SECOND_MAINLINE_DISPLAY_NAME = "Time Fact Core Graph"

TIME_FACT_CORE_DESIGN_PRINCIPLES: tuple[dict[str, str], ...] = (
    {
        "id": "preserve_time_axis",
        "principle": (
            "Time remains an explicit first-class axis. Adjacent repeated "
            "observations can become spans, but their start/end frame and "
            "timestamp evidence is retained."
        ),
    },
    {
        "id": "preserve_answer_bearing_facts",
        "principle": (
            "Scene-local state facts, relation facts, entity references, stable "
            "identities, state values, and transition events remain in the final "
            "QA graph."
        ),
    },
    {
        "id": "normalize_referents_before_retrieval",
        "principle": (
            "Repeated visual fragments that denote the same stable referent are "
            "represented through stable identity or same scene/label/role anchors."
        ),
    },
    {
        "id": "collapse_redundant_observation_spans",
        "principle": (
            "Multiple observations with the same scene, referents, predicate or "
            "attribute value, and compatible time span are one temporal fact span."
        ),
    },
    {
        "id": "remove_derivable_topology_indexes",
        "principle": (
            "Scene/entity segment episodes and interaction-type hubs are treated "
            "as indexing or provenance scaffolding once their fact-bearing links "
            "have been materialized."
        ),
    },
    {
        "id": "dataset_agnostic_no_qa_supervision",
        "principle": (
            "The profile is defined only by graph semantics. It does not inspect "
            "question text, gold answers, retrieval output, or per-sequence QA "
            "performance."
        ),
    },
)

TIME_FACT_CORE_DERIVATION_STEPS: tuple[dict[str, str], ...] = (
    {
        "stage": "hsgg_json_to_temporal_graph",
        "graph_semantics": (
            "Upstream HSGG observations are normalized into a temporal document "
            "with scenes, entities, relations, states, and frame/time metadata."
        ),
    },
    {
        "stage": "temporal_graph_to_time_fact_core_seed_graph",
        "graph_semantics": (
            "The temporal document is converted directly into the canonical "
            "time-fact seed graph. Scene nodes, entity nodes, stable identities, "
            "state facts, relation facts, transition events, and direct "
            "entity-scene presence edges are materialized, while final-graph "
            "index layers such as scene-segment episodes, entity-segment "
            "episodes, and interaction-type hubs are not created."
        ),
    },
    {
        "stage": "stable_identity_time_axis_span_gap30",
        "graph_semantics": (
            "Repeated same-signature observations that are adjacent or overlapping "
            "in time become one explicit time span instead of many duplicate "
            "instantaneous episode records."
        ),
    },
    {
        "stage": "stable_core_entity_compaction",
        "graph_semantics": (
            "Entity fragments are reference-normalized through stable identities "
            "or scene-local label/role groups, so later retrieval sees the object "
            "being described rather than every duplicate detection fragment."
        ),
    },
    {
        "stage": "scene_fact_span_compaction",
        "graph_semantics": (
            "Repeated state and relation facts with the same scene, bound "
            "entities, predicate or attribute value are represented once as "
            "scene-local temporal fact spans."
        ),
    },
)


def build_time_fact_core_graph(
    document: TemporalGraphDocument,
    *,
    source_graph_path: str | Path | None = None,
    compact_scene_label_entities: bool = True,
    compact_scene_label_min_members: int = 4,
    compact_scene_label_entity_segments: bool = True,
    time_axis_max_gap_frames: int = 30,
    time_axis_max_gap_seconds: float = 3.0,
    time_axis_sample_limit: int = 12,
    time_axis_signature_entity_scope: str = "stable_identity",
    stable_core_min_scene_label_group_size: int = 2,
    stable_core_max_raw_visual_attributes: int = 24,
    stable_core_sample_limit: int = 16,
    entity_segment_scene_span_sample_limit: int = 16,
    scene_fact_span_sample_limit: int = 16,
    stable_identity_merge_mode: StableIdentityMergeMode = "semantic",
    stable_identity_include_fragments: bool = False,
    stable_identity_include_noise: bool = False,
    stable_core_compact_scene_label_entities: bool = True,
) -> TimeFactCoreGraphResult:
    """Build the second-mainline time fact core graph from temporal graph data.

    This is a canonical graph profile: it is derived from graph semantics, not
    from QA results. It starts from a temporal graph document and directly
    materializes the target time/fact seed graph instead of first building the
    full first-mainline topology and deleting index layers. Retrieval,
    Question IR, answering, and judging are not modified by this profile.
    """

    seed_graph = build_canonical_temporal_memory_graph(
        document,
        source_graph_path=source_graph_path,
        compact_scene_label_entities=compact_scene_label_entities,
        compact_scene_label_min_members=compact_scene_label_min_members,
        compact_scene_label_entity_segments=compact_scene_label_entity_segments,
        include_scene_segment_episodes=False,
        include_entity_segment_episodes=False,
        include_interaction_type_nodes=False,
        stable_identity_merge_mode=stable_identity_merge_mode,
        stable_identity_include_fragments=stable_identity_include_fragments,
        stable_identity_include_noise=stable_identity_include_noise,
    )
    time_axis = compact_canonical_graph_by_time_axis(
        seed_graph,
        max_gap_frames=time_axis_max_gap_frames,
        max_gap_seconds=time_axis_max_gap_seconds,
        sample_limit=time_axis_sample_limit,
        signature_entity_scope=time_axis_signature_entity_scope,  # type: ignore[arg-type]
    )
    stable_core = compact_canonical_graph_to_stable_core(
        time_axis.graph,
        compact_scene_label_entities=stable_core_compact_scene_label_entities,
        min_scene_label_group_size=stable_core_min_scene_label_group_size,
        max_raw_visual_attributes=stable_core_max_raw_visual_attributes,
        sample_limit=stable_core_sample_limit,
    )
    scene_fact_span = compact_scene_facts_by_span(
        stable_core.graph,
        sample_limit=scene_fact_span_sample_limit,
    )

    report = {
        "contract": "time_fact_core_hsgg_to_graph_pipeline_v1",
        "source_video_id": document.video_id,
        "source_graph_path": str(source_graph_path) if source_graph_path is not None else None,
        "profile": {
            "id": SECOND_MAINLINE_PROFILE_ID,
            "conceptual_name": SECOND_MAINLINE_CONCEPTUAL_NAME,
            "display_name": SECOND_MAINLINE_DISPLAY_NAME,
            "compatibility_id": SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
            "legacy_profile_id": SECOND_MAINLINE_COMPATIBILITY_PROFILE_ID,
            "profile_family": "canonical_temporal_memory_graph",
            "role": "second_mainline_graph_profile",
            "definition": (
                "A time-axis-first canonical graph profile that keeps "
                "answer-bearing temporal facts and avoids materializing "
                "derivable topology indexes in the final QA graph."
            ),
            "defined_by": "graph_semantics_not_qa_or_retrieval_results",
        },
        "design_principles": [
            dict(principle) for principle in TIME_FACT_CORE_DESIGN_PRINCIPLES
        ],
        "derivation_steps": [
            dict(step) for step in TIME_FACT_CORE_DERIVATION_STEPS
        ],
        "policy": {
            "scope": (
                "Second-mainline time fact core graph build from an HSGG-derived "
                "temporal graph. Retrieval, Question IR, QA answering, and "
                "judging are not changed."
            ),
            "profile_id": SECOND_MAINLINE_PROFILE_ID,
            "conceptual_name": SECOND_MAINLINE_CONCEPTUAL_NAME,
            "defined_by_graph_semantics": True,
            "from_hsgg_temporal_graph": True,
            "does_not_read_first_mainline_canonical_graph": True,
            "does_not_inspect_questions_or_gold_answers": True,
            "does_not_inspect_retrieval_or_qa_outputs": True,
            "stages": [
                "hsgg_json_to_temporal_graph",
                "temporal_graph_to_time_fact_core_seed_graph",
                "stable_identity_time_axis_span_gap30",
                "stable_core_entity_compaction",
                "scene_fact_span_compaction",
            ],
            "direct_from_temporal_graph": True,
            "does_not_build_first_mainline_topology": True,
            "does_not_materialize_scene_segment_episodes": True,
            "does_not_materialize_entity_segment_episodes": True,
            "does_not_materialize_interaction_type_nodes": True,
        },
        "parameters": {
            "compact_scene_label_entities": compact_scene_label_entities,
            "compact_scene_label_min_members": compact_scene_label_min_members,
            "compact_scene_label_entity_segments": compact_scene_label_entity_segments,
            "time_axis_max_gap_frames": time_axis_max_gap_frames,
            "time_axis_max_gap_seconds": time_axis_max_gap_seconds,
            "time_axis_sample_limit": time_axis_sample_limit,
            "time_axis_signature_entity_scope": time_axis_signature_entity_scope,
            "stable_core_min_scene_label_group_size": stable_core_min_scene_label_group_size,
            "stable_core_max_raw_visual_attributes": stable_core_max_raw_visual_attributes,
            "stable_core_compact_scene_label_entities": stable_core_compact_scene_label_entities,
            "stable_core_sample_limit": stable_core_sample_limit,
            "entity_segment_scene_span_sample_limit": entity_segment_scene_span_sample_limit,
            "scene_fact_span_sample_limit": scene_fact_span_sample_limit,
            "stable_identity_merge_mode": stable_identity_merge_mode,
            "stable_identity_include_fragments": stable_identity_include_fragments,
            "stable_identity_include_noise": stable_identity_include_noise,
            "direct_seed_include_scene_segment_episodes": False,
            "direct_seed_include_entity_segment_episodes": False,
            "direct_seed_include_interaction_type_nodes": False,
        },
        "sizes": {
            "time_fact_core_seed": _graph_size_report(seed_graph),
            "after_stable_identity_time_axis_span": _graph_size_report(time_axis.graph),
            "after_stable_core": _graph_size_report(stable_core.graph),
            "after_scene_fact_span": _graph_size_report(scene_fact_span.graph),
            "final": _graph_size_report(scene_fact_span.graph),
        },
        "stage_reports": {
            "stable_identity_time_axis_span": time_axis.report,
            "stable_core": stable_core.report,
            "scene_fact_span": scene_fact_span.report,
        },
    }
    return TimeFactCoreGraphResult(
        graph=scene_fact_span.graph,
        report=report,
        seed_graph=seed_graph,
    )


TopologyPrunedAfterSceneFactResult = TimeFactCoreGraphResult
build_topology_pruned_after_scene_fact_graph = build_time_fact_core_graph


def _graph_size_report(graph: CanonicalTemporalMemoryGraph) -> dict[str, int]:
    return {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "scene_node_count": len(graph.scene_nodes),
        "entity_node_count": len(graph.entity_nodes),
        "stable_identity_node_count": len(graph.stable_identity_nodes),
        "state_value_node_count": len(graph.state_value_nodes),
        "interaction_type_node_count": len(graph.interaction_type_nodes),
        "episode_node_count": len(graph.episode_nodes),
        "transition_event_node_count": len(graph.transition_event_nodes),
        "relation_observed_sample_count": len(graph.relation_observed_samples),
    }
