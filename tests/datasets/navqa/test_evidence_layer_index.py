from __future__ import annotations

import json
from pathlib import Path

from vlkg.datasets.navqa.retrieval.evidence_layer_index import (
    EvidenceBuildConfig,
    EvidenceLayerIndex,
    EvidenceSequenceSpec,
    build_evidence_layer_index,
    build_frame_evidence_records,
    build_spatial_facts,
    spatial_relation_candidates,
)
from vlkg.core.models.temporal_graph import BoundingBox
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import fuse_visual_faiss_evidence_layer


def test_frame_hsgg_payload_builds_observations_and_facts(tmp_path: Path) -> None:
    image_path = tmp_path / "000001_2d_rect_cam0_6_10.jpg"
    image_path.write_bytes(b"not-a-real-image")
    spec = EvidenceSequenceSpec(
        seq_id="6",
        cam="0",
        stride=10,
        sample_root=tmp_path,
        images_dir=tmp_path,
    )
    frame_payload = {
        "image_path": str(image_path),
        "objects": {
            "image_path": str(image_path),
            "labels": ["person", "glass door"],
            "boxes": [[10, 20, 50, 80], [45, 20, 90, 85]],
            "scores": [0.9, 0.8],
        },
        "attributes": {
            "0": "person in blue jacket touching the glass door",
            "1": "transparent glass door",
        },
        "relations": [
            {
                "idx": [0, 1],
                "subject_label": "person",
                "predicate": "touching",
                "object_label": "glass door",
                "raw_output": "(person, touching, glass door)",
            }
        ],
    }
    frame_time_index = {
        image_path.name: {
            "sampled_index": 1,
            "graph_frame_index": 10,
            "source_frame": 10,
            "elapsed_sec_from_sequence_start": 1.0,
        }
    }

    records = build_frame_evidence_records(
        spec=spec,
        frame_key=image_path.stem,
        frame_payload=frame_payload,
        frame_time_index=frame_time_index,
        max_neighbors_per_object=2,
    )

    assert len(records.observations) == 2
    assert records.observations[0]["observation_id"].endswith("object:0")
    assert records.observations[0]["label"] == "person"
    assert records.observations[0]["graph_frame_index"] == 10
    assert records.observations[0]["image_exists"] is True
    assert records.attribute_facts[0]["attribute"] == (
        "person in blue jacket touching the glass door"
    )
    assert records.relation_facts[0]["predicate"] == "touching"
    assert records.relation_facts[0]["subject_observation_id"] == (
        records.observations[0]["observation_id"]
    )
    assert records.spatial_facts


def test_spatial_relation_candidates_are_generic_geometry() -> None:
    left = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    right = BoundingBox(x1=20, y1=0, x2=30, y2=10)
    relations = spatial_relation_candidates(left, right)

    assert "subject_left_of_object" in relations


def test_spatial_facts_keep_object_level_anchors() -> None:
    observations = [
        {
            "seq_id": "toy",
            "frame_key": "frame1",
            "graph_frame_index": 10,
            "observation_id": "toy:frame1:object:0",
            "object_index": 0,
            "label": "person",
            "bbox": [0, 0, 10, 10],
        },
        {
            "seq_id": "toy",
            "frame_key": "frame1",
            "graph_frame_index": 10,
            "observation_id": "toy:frame1:object:1",
            "object_index": 1,
            "label": "door",
            "bbox": [8, 0, 18, 10],
        },
    ]

    facts = build_spatial_facts(observations=observations, max_neighbors_per_object=1)

    assert len(facts) == 1
    assert facts[0]["subject_observation_id"] == "toy:frame1:object:0"
    assert facts[0]["object_observation_id"] == "toy:frame1:object:1"
    assert "overlap" in facts[0]["relations"]


def test_build_index_writes_sqlite_query_store(tmp_path: Path) -> None:
    sample_root = tmp_path / ".scratch" / "navqa" / "seq6" / "cam0" / "every10"
    hsgg_dir = sample_root / "hsgg_local_paths" / "frame_scene_graphs"
    images_dir = sample_root / "images"
    hsgg_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    image_path = images_dir / "000001_2d_rect_cam0_6_10.jpg"
    image_path.write_bytes(b"not-a-real-image")
    frame_payload = {
        "image_path": str(image_path),
        "objects": {
            "image_path": str(image_path),
            "labels": ["person", "door"],
            "boxes": [[10, 20, 50, 80], [45, 20, 90, 85]],
            "scores": [0.9, 0.8],
        },
        "attributes": {"0": "person wearing blue jacket", "1": "glass door"},
        "relations": [{"idx": [0, 1], "predicate": "touching"}],
    }
    frame_name = image_path.stem
    (hsgg_dir / f"{frame_name}.json").write_text(
        json.dumps(frame_payload),
        encoding="utf-8",
    )
    frame_time_index = sample_root / "frame_time_index.json"
    frame_time_index.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image_name": image_path.name,
                        "sampled_index": 1,
                        "graph_frame_index": 10,
                        "source_frame": 10,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "seq_id": "6",
                        "cam": "0",
                        "stride": 10,
                        "sample_root": str(sample_root),
                        "hsgg_input_dir": str(sample_root / "hsgg_local_paths"),
                        "frame_time_index_json": str(frame_time_index),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence_index"

    summary = build_evidence_layer_index(
        EvidenceBuildConfig(
            sequence_manifest=manifest_path,
            cg_manifests=(),
            output_dir=output_dir,
            force=True,
        )
    )

    assert summary["files"]["sqlite"]["exists"] is True
    assert not (output_dir / "spatial_facts.jsonl").exists()
    index = EvidenceLayerIndex(output_dir / "evidence_layer.sqlite")
    try:
        assert index.count("observations") == 2
        assert index.count("attribute_facts") == 2
        assert index.count("relation_facts") == 1
        assert index.count("spatial_facts") == 0
        pack = index.local_evidence_pack(f"seq6:frame:{frame_name}:object:0")
    finally:
        index.close()

    assert pack["observation"]["label"] == "person"
    assert pack["attributes"][0]["attribute"] == "person wearing blue jacket"
    assert pack["relations"][0]["predicate"] == "touching"
    assert pack["bbox_neighbors"]


def test_visual_faiss_evidence_layer_fusion_uses_object_observation_pack(
    tmp_path: Path,
) -> None:
    sample_root = tmp_path / ".scratch" / "navqa" / "seq6" / "cam0" / "every10"
    hsgg_dir = sample_root / "hsgg_local_paths" / "frame_scene_graphs"
    images_dir = sample_root / "images"
    hsgg_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    image_path = images_dir / "000001_2d_rect_cam0_6_10.jpg"
    image_path.write_bytes(b"not-a-real-image")
    frame_name = image_path.stem
    frame_payload = {
        "image_path": str(image_path),
        "objects": {
            "image_path": str(image_path),
            "labels": ["person", "glass door"],
            "boxes": [[10, 20, 50, 80], [45, 20, 90, 85]],
            "scores": [0.9, 0.8],
        },
        "attributes": {
            "0": "person wearing blue jacket",
            "1": "transparent glass door",
        },
        "relations": [
            {
                "idx": [0, 1],
                "subject_label": "person",
                "predicate": "touching",
                "object_label": "glass door",
                "raw_output": "(person, touching, glass door)",
            }
        ],
    }
    (hsgg_dir / f"{frame_name}.json").write_text(
        json.dumps(frame_payload),
        encoding="utf-8",
    )
    frame_time_index = sample_root / "frame_time_index.json"
    frame_time_index.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "image_name": image_path.name,
                        "sampled_index": 1,
                        "graph_frame_index": 10,
                        "source_frame": 10,
                        "elapsed_sec_from_sequence_start": 1.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "seq_id": "6",
                        "cam": "0",
                        "stride": 10,
                        "sample_root": str(sample_root),
                        "hsgg_input_dir": str(sample_root / "hsgg_local_paths"),
                        "frame_time_index_json": str(frame_time_index),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence_index"
    build_evidence_layer_index(
        EvidenceBuildConfig(
            sequence_manifest=manifest_path,
            cg_manifests=(),
            output_dir=output_dir,
            force=True,
        )
    )
    graph_answer = {
        "retrieval_bundle": {"focus_frames": [], "evidence_digest": []},
        "llm_ready_bundle": {
            "llm_context_text": "Video: toy",
            "evidence_lines": ["rank=1 kind=entity score=1 :: graph context"],
            "visual_faiss_evidence_lines": [],
            "evidence_aggregation": {"candidate_facts": []},
            "evidence_aggregation_lines": [],
            "structured_graph_context": {},
        },
    }
    visual_payload = {
        "contract": "experimental_visual_query_faiss_recall_v1",
        "available": True,
        "question": "What is touching the glass door?",
        "groups": [
            {
                "query_phrase": "person touching glass door",
                "results": [
                    {
                        "seq_id": "6",
                        "frame_key": frame_name,
                        "image_name": image_path.name,
                        "graph_frame_index": 10,
                        "object_index": 0,
                        "label": "person",
                        "similarity": 0.71,
                    }
                ],
            }
        ],
    }

    updated = fuse_visual_faiss_evidence_layer(
        graph_answer,
        visual_payload,
        evidence_layer_sqlite=output_dir / "evidence_layer.sqlite",
        object_limit=4,
        neighbor_limit=2,
    )
    bundle = updated["llm_ready_bundle"]
    facts = bundle["evidence_aggregation"]["candidate_facts"]

    assert bundle["visual_faiss_evidence_layer_lines"]
    assert bundle["evidence_lines"][0] == bundle["visual_faiss_evidence_layer_lines"][0]
    assert facts[0]["fact_type"] == "visual_faiss_evidence_layer_fact"
    assert facts[0]["source_kind"] == "visual_faiss_evidence_layer_fact"
    assert "person wearing blue jacket" in facts[0]["fact_text"]
    assert "touching" in facts[0]["fact_text"]
    assert bundle["structured_graph_context"]["visual_faiss_evidence_layer_lines"]
    assert any(
        item["source_kind"] == "visual_faiss_evidence_layer_fact"
        and item.get("observation_id", "").endswith("object:0")
        for item in updated["retrieval_bundle"]["evidence_digest"]
    )
    metadata = bundle["visual_faiss_evidence_layer"]
    assert metadata["contract"] == "experimental_visual_faiss_evidence_layer_v1"
    assert metadata["uses_gold_answers"] is False
    assert metadata["uses_question_type_branches"] is False
    assert metadata["uses_dataset_or_domain_wordlist"] is False
