from __future__ import annotations

from vlkg.datasets.navqa.retrieval.frame_faiss_grounding import (
    cluster_frame_results,
    expand_frame_faiss_graph_evidence,
    frame_query_texts,
    frame_faiss_payload_as_graph_anchor_payload,
    fuse_frame_faiss_evidence,
)
from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)


def test_frame_query_texts_are_question_ir_derived_without_keyword_filters() -> None:
    texts = frame_query_texts(
        question="Did the robot stop near the stop sign?",
        question_ir={
            "target_state": "robot stopped near stop sign",
            "target_entities": ["robot", "stop sign"],
            "spatial_constraints": [{"relation": "near", "target": "stop sign"}],
            "evidence_requirements": [
                "ego trajectory with stop events and proximity to stop sign"
            ],
        },
        limit=6,
    )

    assert texts[0] == "a video frame of Did the robot stop near the stop sign?"
    assert "a video frame of robot stopped near stop sign" in texts
    assert "a video frame of robot" in texts
    assert "a video frame of stop sign" in texts
    assert "a video frame of near" in texts


def test_cluster_frame_results_groups_adjacent_frames_generically() -> None:
    rows = [
        {
            "seq_id": "0",
            "graph_frame_index": 100,
            "elapsed_sec_from_sequence_start": 10.0,
            "similarity": 0.2,
            "faiss_id": 1,
            "image_name": "a.jpg",
            "_query_text": "a video frame of first query",
        },
        {
            "seq_id": "0",
            "graph_frame_index": 110,
            "elapsed_sec_from_sequence_start": 11.0,
            "similarity": 0.3,
            "faiss_id": 2,
            "image_name": "b.jpg",
            "_query_text": "a video frame of second query",
        },
        {
            "seq_id": "0",
            "graph_frame_index": 300,
            "elapsed_sec_from_sequence_start": 30.0,
            "similarity": 0.25,
            "faiss_id": 3,
            "image_name": "c.jpg",
            "_query_text": "a video frame of third query",
        },
    ]

    clusters = cluster_frame_results(
        rows,
        max_frame_gap=30,
        max_sec_gap=None,
        max_clusters=5,
    )

    assert len(clusters) == 2
    assert clusters[0]["start_frame"] == 100
    assert clusters[0]["end_frame"] == 110
    assert clusters[0]["frame_count"] == 2
    assert clusters[0]["max_similarity"] == 0.3
    assert clusters[1]["start_frame"] == 300


def test_frame_faiss_fusion_promotes_clusters_into_evidence_pool() -> None:
    graph_answer = {
        "retrieval_bundle": {
            "focus_frames": [{"frame_index": 10}],
            "evidence_digest": [
                {
                    "rank": 1,
                    "source_record_id": "graph:1",
                    "source_kind": "entity",
                    "summary": "graph evidence",
                    "score": 1.0,
                }
            ],
        },
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy\nEvidence lines:\n- graph evidence",
            "evidence_lines": ["rank=1 kind=entity score=1 :: graph evidence"],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    frame_payload = {
        "contract": "navqa_frame_faiss_query_recall_v1",
        "available": True,
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "frame_count": 2,
                "start_frame": 42,
                "end_frame": 44,
                "start_elapsed_sec": 4.2,
                "end_elapsed_sec": 4.4,
                "max_similarity": 0.55,
                "best_frame": {
                    "seq_id": "toy",
                    "graph_frame_index": 42,
                    "elapsed_sec_from_sequence_start": 4.2,
                    "similarity": 0.55,
                },
                "query_texts": ["a video frame of red sign near doorway"],
                "frames": [],
            }
        ],
        "visual_focus_frames": [
            {
                "frame_index": 42,
                "timestamp_sec": 4.2,
                "score": 0.55,
                "reasons": ["frame_faiss_query_recall"],
                "query_text": "a video frame of red sign near doorway",
                "cluster_rank": 1,
            }
        ],
    }

    updated = fuse_frame_faiss_evidence(graph_answer, frame_payload)
    bundle = updated["llm_ready_bundle"]

    assert any("frame_faiss_cluster" in line for line in bundle["evidence_lines"])
    assert any(
        fact["fact_type"] == "frame_faiss_cluster_recall"
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
    )
    assert any(
        item["source_kind"] == "frame_faiss_cluster"
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    assert any(
        frame.get("frame_index") == 42
        for frame in updated["retrieval_bundle"]["focus_frames"]
    )


def test_frame_faiss_payload_can_be_used_as_graph_anchor_payload() -> None:
    frame_payload = {
        "contract": "navqa_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a video frame of doorway"],
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "start_frame": 42,
                "end_frame": 44,
                "max_similarity": 0.55,
                "best_frame": {
                    "seq_id": "toy",
                    "graph_frame_index": 42,
                    "elapsed_sec_from_sequence_start": 4.2,
                    "similarity": 0.55,
                },
                "query_texts": ["a video frame of doorway"],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42}],
    }

    anchor_payload = frame_faiss_payload_as_graph_anchor_payload(frame_payload)

    assert anchor_payload["contract"] == "navqa_frame_faiss_graph_anchor_payload_v1"
    assert anchor_payload["groups"][0]["query_phrase"] == "a video frame of doorway"
    result = anchor_payload["groups"][0]["results"][0]
    assert result["graph_frame_index"] == 42
    assert result["label"] == "full_frame"


def test_frame_faiss_graph_anchor_expansion_promotes_graph_evidence_only() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="red sign is visible beside the doorway",
                time_range=CanonicalTimeRange(start_frame=42, end_frame=43),
                scene_id="scene_1",
                primary_entity_id="entity_sign",
                attr_key="visibility",
                attr_value="visible",
                source_ref_id="state_1",
            )
        ],
    )
    graph_answer = {
        "retrieval_bundle": {
            "focus_frames": [],
            "evidence_digest": [],
        },
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": [],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    frame_payload = {
        "contract": "navqa_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a video frame of red sign"],
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "start_frame": 42,
                "end_frame": 42,
                "max_similarity": 0.55,
                "best_frame": {
                    "seq_id": "toy",
                    "graph_frame_index": 42,
                    "elapsed_sec_from_sequence_start": 4.2,
                    "similarity": 0.55,
                },
                "query_texts": ["a video frame of red sign"],
            }
        ],
        "visual_focus_frames": [{"frame_index": 42}],
    }

    updated = expand_frame_faiss_graph_evidence(
        graph_answer,
        frame_payload,
        graph=graph,
        frame_radius=2,
        max_items=5,
    )
    bundle = updated["llm_ready_bundle"]

    assert not bundle["visual_faiss_evidence_lines"]
    assert any("red sign is visible" in line for line in bundle["evidence_lines"])
    assert any("frame_faiss_anchor" in line for line in bundle["evidence_lines"])
    assert not any("visual_faiss_anchor" in line for line in bundle["evidence_lines"])
    assert any(
        fact["fact_type"] == "frame_faiss_anchored_graph_evidence"
        for fact in bundle["evidence_aggregation"]["candidate_facts"]
    )
    assert bundle["frame_faiss_anchor_expansion"]["contract"] == (
        "navqa_frame_faiss_graph_anchor_expansion_v1"
    )
