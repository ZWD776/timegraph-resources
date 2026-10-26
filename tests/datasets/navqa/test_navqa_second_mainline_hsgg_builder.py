from __future__ import annotations

from vlkg.core.transformers import (
    build_time_fact_core_graph,
    build_topology_pruned_after_scene_fact_graph,
)

from tests.test_canonical_temporal_memory_graph import _sample_temporal_graph_document


def test_time_fact_core_builds_from_temporal_document() -> None:
    document = _sample_temporal_graph_document()

    result = build_time_fact_core_graph(document)

    assert result.report["policy"]["direct_from_temporal_graph"] is True
    assert result.report["policy"]["does_not_build_first_mainline_topology"] is True
    assert result.report["policy"]["does_not_materialize_scene_segment_episodes"] is True
    assert result.report["policy"]["does_not_materialize_entity_segment_episodes"] is True
    assert result.report["policy"]["does_not_materialize_interaction_type_nodes"] is True
    assert result.report["policy"]["from_hsgg_temporal_graph"] is True
    assert result.report["policy"]["does_not_read_first_mainline_canonical_graph"] is True
    assert result.report["policy"]["does_not_inspect_questions_or_gold_answers"] is True
    assert result.report["policy"]["does_not_inspect_retrieval_or_qa_outputs"] is True
    assert result.report["policy"]["defined_by_graph_semantics"] is True
    assert result.report["profile"]["id"] == "time_fact_core"
    assert result.report["profile"]["compatibility_id"] == "topology_pruned_after_scene_fact"
    assert result.report["profile"]["legacy_profile_id"] == "topology_pruned_after_scene_fact"
    assert result.report["profile"]["conceptual_name"] == "time_fact_core_graph"
    assert (
        result.report["profile"]["defined_by"]
        == "graph_semantics_not_qa_or_retrieval_results"
    )
    principle_ids = {
        principle["id"] for principle in result.report["design_principles"]
    }
    assert {
        "preserve_time_axis",
        "preserve_answer_bearing_facts",
        "normalize_referents_before_retrieval",
        "collapse_redundant_observation_spans",
        "remove_derivable_topology_indexes",
        "dataset_agnostic_no_qa_supervision",
    }.issubset(principle_ids)
    assert [
        step["stage"] for step in result.report["derivation_steps"]
    ] == result.report["policy"]["stages"]
    assert result.report["sizes"]["time_fact_core_seed"]["node_count"] == (
        result.seed_graph.node_count
    )
    assert result.report["sizes"]["final"]["node_count"] == (
        result.graph.node_count
    )
    assert not result.seed_graph.interaction_type_nodes
    assert not result.graph.interaction_type_nodes
    assert {
        node.episode_kind
        for node in result.graph.episode_nodes
    }.isdisjoint({"SCENE_SEGMENT", "ENTITY_SEGMENT"})
    assert {"STATE", "RELATION"}.issubset(
        {node.episode_kind for node in result.graph.episode_nodes}
    )


def test_time_fact_core_uses_fixed_second_mainline_stages() -> None:
    result = build_time_fact_core_graph(_sample_temporal_graph_document())

    assert result.report["policy"]["stages"] == [
        "hsgg_json_to_temporal_graph",
        "temporal_graph_to_time_fact_core_seed_graph",
        "stable_identity_time_axis_span_gap30",
        "stable_core_entity_compaction",
        "scene_fact_span_compaction",
    ]
    assert result.report["parameters"]["compact_scene_label_entities"] is True
    assert result.report["parameters"]["stable_core_compact_scene_label_entities"] is True
    assert result.report["parameters"]["stable_identity_merge_mode"] == "semantic"
    assert result.report["parameters"]["stable_identity_include_fragments"] is False
    assert result.report["parameters"]["stable_identity_include_noise"] is False
    assert result.report["parameters"]["direct_seed_include_scene_segment_episodes"] is False
    assert result.report["parameters"]["direct_seed_include_entity_segment_episodes"] is False
    assert result.report["parameters"]["direct_seed_include_interaction_type_nodes"] is False
    assert result.report["parameters"]["time_axis_max_gap_frames"] == 30
    assert result.report["parameters"]["time_axis_max_gap_seconds"] == 3.0
    assert (
        result.report["parameters"]["time_axis_signature_entity_scope"]
        == "stable_identity"
    )


def test_legacy_topology_pruned_alias_uses_formal_second_mainline_profile() -> None:
    result = build_topology_pruned_after_scene_fact_graph(
        _sample_temporal_graph_document()
    )

    assert result.report["profile"]["id"] == "time_fact_core"
    assert result.report["profile"]["conceptual_name"] == "time_fact_core_graph"
