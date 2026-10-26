from __future__ import annotations

from vlkg.core.graph_retrieval.canonical_compressed_fact_projection import (
    build_projection_from_compressed_fact_layer,
)


def test_compressed_fact_projection_preserves_same_label_entities() -> None:
    layer = {
        "contract": "canonical_compressed_fact_layer_v1",
        "source_video_id": "video",
        "facts": [
            {
                "fact_id": "scene_profile:s1",
                "kind": "scene_profile",
                "source_node_id": "scene:1",
                "scene_id": "scene:1",
                "time_range": {"start_ts": 0.0, "end_ts": 1.0},
                "label": "scene",
                "labels": ["scene"],
                "entity_ids": ["entity:a", "entity:b"],
                "text": "Scene scene:1 spans 0-1.",
            },
            {
                "fact_id": "entity_profile:a",
                "kind": "entity_profile",
                "source_node_id": "node:a",
                "entity_ids": ["entity:a"],
                "scene_ids": ["scene:1"],
                "time_range": {"start_ts": 0.1, "end_ts": 0.2},
                "label": "car",
                "labels": ["car"],
                "raw_visual_attributes": ["red car"],
                "text": "Entity entity:a (car) appears during 0.1-0.2.",
            },
            {
                "fact_id": "entity_profile:b",
                "kind": "entity_profile",
                "source_node_id": "node:b",
                "entity_ids": ["entity:b"],
                "scene_ids": ["scene:1"],
                "time_range": {"start_ts": 0.2, "end_ts": 0.3},
                "label": "car",
                "labels": ["car"],
                "raw_visual_attributes": ["blue car"],
                "text": "Entity entity:b (car) appears during 0.2-0.3.",
            },
            {
                "fact_id": "episode_fact_span:1",
                "kind": "episode_fact_span",
                "episode_kind": "RELATION",
                "scene_id": "scene:1",
                "entity_ids": ["entity:a", "entity:b"],
                "subject_entity_id": "entity:a",
                "subject_entity_label": "car",
                "object_entity_id": "entity:b",
                "object_entity_label": "car",
                "predicate": "RAW_RELATION_NEAR",
                "raw_predicate": "near",
                "time_range": {"start_ts": 0.2, "end_ts": 0.2},
                "source_episode_count": 1,
                "source_node_id_samples": ["episode:1"],
                "text": "Relation fact: entity:a near entity:b.",
            },
        ],
    }

    projection, report = build_projection_from_compressed_fact_layer(layer)

    assert report["projection_record_count_by_kind"]["entity"] == 2
    assert projection.records["node:a"].entity_ids == ("entity:a",)
    assert projection.records["node:b"].entity_ids == ("entity:b",)
    assert "episode:1" in projection.records
    assert any(
        link.neighbor_id == "episode:1"
        for link in projection.adjacency.get("node:a", [])
    )
