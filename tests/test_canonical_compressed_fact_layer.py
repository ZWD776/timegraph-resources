from __future__ import annotations

from vlkg.core.graph_retrieval.canonical_compressed_fact_layer import build_compressed_fact_layer
from vlkg.core.models.canonical_graph import (
    CanonicalEntityNode,
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)


def test_compressed_fact_layer_does_not_merge_same_label_entities() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="video",
        entity_nodes=[
            entity("entity:a", "car"),
            entity("entity:b", "car"),
        ],
        episode_nodes=[
            relation("rel1", subject="entity:a", obj="entity:b", frame=10),
            relation("rel2", subject="entity:a", obj="entity:b", frame=12),
        ],
    )

    layer = build_compressed_fact_layer(graph)

    entity_facts = [
        fact for fact in layer["facts"] if fact["kind"] == "entity_profile"
    ]
    assert len(entity_facts) == 2
    assert {tuple(fact["entity_ids"]) for fact in entity_facts} == {
        ("entity:a",),
        ("entity:b",),
    }

    relation_facts = [
        fact for fact in layer["facts"] if fact["kind"] == "episode_fact_span"
    ]
    assert len(relation_facts) == 1
    assert relation_facts[0]["subject_entity_id"] == "entity:a"
    assert relation_facts[0]["object_entity_id"] == "entity:b"
    assert relation_facts[0]["source_episode_count"] == 2


def entity(entity_id: str, label: str) -> CanonicalEntityNode:
    return CanonicalEntityNode(
        node_id=entity_id,
        video_id="video",
        entity_id=entity_id,
        label=label,
        time_range=CanonicalTimeRange(start_frame=1, end_frame=2),
        observation_count=2,
        attributes={"raw_visual_attributes": [f"{label} visual attribute"]},
    )


def relation(
    suffix: str,
    *,
    subject: str,
    obj: str,
    frame: int,
) -> CanonicalEpisodeNode:
    return CanonicalEpisodeNode(
        node_id=f"episode:{suffix}",
        video_id="video",
        episode_id=suffix,
        episode_kind="RELATION",
        label=f"{subject} near {obj}",
        time_range=CanonicalTimeRange(start_frame=frame, end_frame=frame),
        scene_id="scene:1",
        primary_entity_id=subject,
        subject_entity_id=subject,
        object_entity_id=obj,
        participant_entity_ids=[subject, obj],
        predicate="RAW_RELATION_NEAR",
        raw_predicate="near",
        source_ref_id=suffix,
    )
