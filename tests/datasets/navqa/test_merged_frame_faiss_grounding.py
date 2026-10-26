from __future__ import annotations

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
)
from vlkg.datasets.navqa.retrieval.merged_frame_faiss_grounding import (
    aggregate_frame_rows_to_windows,
    attach_merged_frame_faiss_evidence,
    cluster_merged_frame_results,
    expand_merged_frame_faiss_graph_evidence,
    frame_payload_as_window_payload,
    merged_frame_payload_as_frame_payload,
    merged_frame_query_texts,
)


def test_merged_frame_query_texts_are_question_ir_derived_generically() -> None:
    texts = merged_frame_query_texts(
        question="Did you see a person opening the door?",
        question_ir={
            "target_state": "person opening door",
            "target_entities": ["person", "door"],
            "spatial_constraints": [{"relation": "near", "target": "entrance"}],
            "evidence_requirements": ["visual doorway event evidence"],
        },
        limit=6,
    )

    assert texts[0] == "a storyboard video window of Did you see a person opening the door?"
    assert "a storyboard video window of person opening door" in texts
    assert "a storyboard video window of person" in texts
    assert "a storyboard video window of door" in texts
    assert "a storyboard video window of near" in texts


def test_cluster_merged_frame_results_groups_overlapping_windows() -> None:
    rows = [
        {
            "seq_id": "0",
            "window_index": 1,
            "start_graph_frame_index": 100,
            "end_graph_frame_index": 108,
            "start_elapsed_sec": 10.0,
            "end_elapsed_sec": 14.0,
            "similarity": 0.2,
            "faiss_id": 1,
            "_query_text": "a storyboard video window of doorway",
        },
        {
            "seq_id": "0",
            "window_index": 2,
            "start_graph_frame_index": 112,
            "end_graph_frame_index": 120,
            "start_elapsed_sec": 16.0,
            "end_elapsed_sec": 20.0,
            "similarity": 0.3,
            "faiss_id": 2,
            "_query_text": "a storyboard video window of doorway",
        },
        {
            "seq_id": "0",
            "window_index": 9,
            "start_graph_frame_index": 300,
            "end_graph_frame_index": 308,
            "start_elapsed_sec": 60.0,
            "end_elapsed_sec": 64.0,
            "similarity": 0.25,
            "faiss_id": 3,
            "_query_text": "a storyboard video window of hallway",
        },
    ]

    clusters = cluster_merged_frame_results(
        rows,
        max_frame_gap=12,
        max_sec_gap=None,
        max_clusters=5,
    )

    assert len(clusters) == 2
    assert clusters[0]["start_frame"] == 100
    assert clusters[0]["end_frame"] == 120
    assert clusters[0]["window_count"] == 2
    assert clusters[0]["max_similarity"] == 0.3
    assert clusters[0]["best_window"]["window_index"] == 2
    assert clusters[1]["start_frame"] == 300


def test_frame_rows_aggregate_to_generic_temporal_windows() -> None:
    rows = [
        {
            "seq_id": "0",
            "faiss_id": 1,
            "graph_frame_index": 100,
            "source_frame": 100,
            "sampled_index": 10,
            "elapsed_sec_from_sequence_start": 10.0,
            "similarity": 0.2,
            "image_path": "frame_100.jpg",
            "_query_text": "a video frame of doorway",
        },
        {
            "seq_id": "0",
            "faiss_id": 2,
            "graph_frame_index": 110,
            "source_frame": 110,
            "sampled_index": 11,
            "elapsed_sec_from_sequence_start": 11.0,
            "similarity": 0.5,
            "image_path": "frame_110.jpg",
            "_query_text": "a video frame of doorway",
        },
        {
            "seq_id": "0",
            "faiss_id": 3,
            "graph_frame_index": 900,
            "source_frame": 900,
            "sampled_index": 90,
            "elapsed_sec_from_sequence_start": 90.0,
            "similarity": 0.4,
            "image_path": "frame_900.jpg",
            "_query_text": "a video frame of hallway",
        },
    ]

    windows = aggregate_frame_rows_to_windows(
        rows,
        window_size_sec=5.0,
        window_stride_sec=2.0,
        max_windows=4,
        max_frames_per_window=4,
    )

    assert windows[0]["max_similarity"] == 0.5
    assert windows[0]["start_elapsed_sec"] == 8.0
    assert windows[0]["end_elapsed_sec"] == 13.0
    assert windows[0]["start_graph_frame_index"] == 80
    assert windows[0]["end_graph_frame_index"] == 130
    assert windows[0]["frame_count"] == 2
    assert windows[1]["max_similarity"] == 0.4


def test_frame_payload_as_window_payload_uses_merged_frame_contract_shape() -> None:
    frame_payload = {
        "contract": "navqa_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a video frame of red sign"],
        "groups": [
            {
                "query_text": "a video frame of red sign",
                "results": [
                    {
                        "rank": 1,
                        "faiss_id": 1,
                        "seq_id": "toy",
                        "graph_frame_index": 44,
                        "source_frame": 440,
                        "sampled_index": 44,
                        "elapsed_sec_from_sequence_start": 4.4,
                        "similarity": 0.55,
                        "image_path": "frame_44.jpg",
                    }
                ],
            }
        ],
    }

    window_payload = frame_payload_as_window_payload(
        frame_payload,
        window_size_sec=5.0,
        window_stride_sec=2.0,
        max_clusters=3,
        max_lines=3,
        max_frames_per_window=4,
    )
    frame_anchor_payload = merged_frame_payload_as_frame_payload(window_payload)

    assert window_payload["contract"] == "navqa_frame_window_faiss_query_recall_v1"
    assert window_payload["source_contract"] == "navqa_frame_faiss_query_recall_v1"
    assert window_payload["clusters"][0]["source_kind"] == "frame_window_faiss"
    assert window_payload["clusters"][0]["best_window"]["aggregation"] == (
        "frame_window_faiss"
    )
    assert window_payload["lines"]
    assert "frame_window_faiss_cluster" in window_payload["lines"][0]
    assert frame_anchor_payload["contract"] == (
        "navqa_merged_frame_faiss_as_frame_query_recall_v1"
    )
    assert frame_anchor_payload["clusters"][0]["best_frame"]["image_path"] == "frame_44.jpg"


def test_merged_frame_payload_adapts_to_frame_anchor_payload() -> None:
    merged_payload = {
        "contract": "navqa_merged_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a storyboard video window of red sign"],
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "start_frame": 40,
                "end_frame": 48,
                "start_elapsed_sec": 4.0,
                "end_elapsed_sec": 8.0,
                "max_similarity": 0.55,
                "query_texts": ["a storyboard video window of red sign"],
                "best_window": {
                    "rank": 1,
                    "seq_id": "toy",
                    "window_index": 7,
                    "start_graph_frame_index": 40,
                    "end_graph_frame_index": 48,
                    "start_elapsed_sec": 4.0,
                    "end_elapsed_sec": 8.0,
                    "similarity": 0.55,
                },
                "windows": [],
            }
        ],
        "visual_focus_frames": [{"frame_index": 44}],
    }

    frame_payload = merged_frame_payload_as_frame_payload(merged_payload)

    assert frame_payload["contract"] == "navqa_merged_frame_faiss_as_frame_query_recall_v1"
    assert frame_payload["clusters"][0]["best_frame"]["graph_frame_index"] == 44
    assert frame_payload["clusters"][0]["best_frame"]["merged_frame_window"]["window_index"] == 7
    assert frame_payload["source_merged_frame_faiss"] is merged_payload


def test_merged_frame_attach_keeps_evidence_in_merged_channel_only() -> None:
    existing_frame_payload = {"contract": "existing_frame"}
    existing_visual_payload = {"contract": "existing_visual"}
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "frame_faiss_grounding": existing_frame_payload,
            "frame_faiss_payloads": [existing_frame_payload],
            "frame_faiss_evidence_lines": ["EXISTING_FRAME_LINE"],
            "visual_faiss_grounding": existing_visual_payload,
            "visual_faiss_payloads": [existing_visual_payload],
            "visual_faiss_evidence_lines": ["EXISTING_VISUAL_LINE"],
            "structured_graph_context": {
                "frame_faiss_grounding": existing_frame_payload,
                "frame_faiss_payloads": [existing_frame_payload],
                "frame_faiss_evidence_lines": ["EXISTING_FRAME_LINE"],
                "visual_faiss_grounding": existing_visual_payload,
                "visual_faiss_payloads": [existing_visual_payload],
                "visual_faiss_evidence_lines": ["EXISTING_VISUAL_LINE"],
            },
        },
    }
    merged_payload = {
        "contract": "navqa_merged_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a storyboard video window of red sign"],
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "start_frame": 40,
                "end_frame": 48,
                "start_elapsed_sec": 4.0,
                "end_elapsed_sec": 8.0,
                "max_similarity": 0.55,
                "query_texts": ["a storyboard video window of red sign"],
                "best_window": {
                    "seq_id": "toy",
                    "window_index": 7,
                    "start_graph_frame_index": 40,
                    "end_graph_frame_index": 48,
                    "start_elapsed_sec": 4.0,
                    "similarity": 0.55,
                },
                "windows": [],
            }
        ],
        "visual_focus_frames": [{"frame_index": 44}],
    }

    updated = attach_merged_frame_faiss_evidence(graph_answer, merged_payload)
    bundle = updated["llm_ready_bundle"]

    assert bundle["frame_faiss_grounding"] is existing_frame_payload
    assert bundle["frame_faiss_payloads"] == [existing_frame_payload]
    assert bundle["frame_faiss_evidence_lines"] == ["EXISTING_FRAME_LINE"]
    assert bundle["visual_faiss_grounding"] is existing_visual_payload
    assert bundle["visual_faiss_payloads"] == [existing_visual_payload]
    assert bundle["visual_faiss_evidence_lines"] == ["EXISTING_VISUAL_LINE"]
    assert any(
        "merged_frame_faiss_cluster" in line
        for line in bundle["merged_frame_faiss_evidence_lines"]
    )
    assert not any(
        "merged_frame_faiss_cluster" in line
        for line in bundle["frame_faiss_evidence_lines"]
    )


def test_merged_frame_graph_anchor_expansion_promotes_graph_evidence_only() -> None:
    graph = CanonicalTemporalMemoryGraph(
        video_id="toy",
        episode_nodes=[
            CanonicalEpisodeNode(
                node_id="episode_1",
                video_id="toy",
                episode_id="episode_1",
                episode_kind="STATE",
                label="the red sign is visible near the doorway",
                time_range=CanonicalTimeRange(start_frame=44, end_frame=45),
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
    merged_payload = {
        "contract": "navqa_merged_frame_faiss_query_recall_v1",
        "available": True,
        "seq_id": "toy",
        "query_texts": ["a storyboard video window of red sign"],
        "clusters": [
            {
                "rank": 1,
                "seq_id": "toy",
                "start_frame": 40,
                "end_frame": 48,
                "start_elapsed_sec": 4.0,
                "end_elapsed_sec": 8.0,
                "max_similarity": 0.55,
                "query_texts": ["a storyboard video window of red sign"],
                "best_window": {
                    "seq_id": "toy",
                    "window_index": 7,
                    "start_graph_frame_index": 40,
                    "end_graph_frame_index": 48,
                    "start_elapsed_sec": 4.0,
                    "similarity": 0.55,
                },
                "windows": [],
            }
        ],
        "visual_focus_frames": [{"frame_index": 44}],
    }

    updated = expand_merged_frame_faiss_graph_evidence(
        graph_answer,
        merged_payload,
        graph=graph,
        frame_radius=2,
        max_items=5,
    )
    bundle = updated["llm_ready_bundle"]

    assert any("the red sign is visible" in line for line in bundle["evidence_lines"])
    assert any("frame_faiss_anchor" in line for line in bundle["evidence_lines"])
    assert bundle["merged_frame_faiss_grounding"] is merged_payload
    assert bundle["merged_frame_faiss_evidence_lines"]
    assert not bundle.get("frame_faiss_evidence_lines", [])
    assert not bundle.get("visual_faiss_evidence_lines", [])
