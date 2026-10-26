from __future__ import annotations

import json
import sqlite3
import sys
from importlib import import_module
from pathlib import Path

import numpy as np

from vlkg.datasets.openeqa.retrieval.evidence_router import compile_evidence_demand
from vlkg.datasets.openeqa.runtime.visual_object_index import (
    GraphObservationBindingIndex,
    OBJECT_VISUAL_INDEX_CONTRACT,
    ObjectVisualSearchIndex,
    validate_object_visual_index_manifest,
)


SCRIPT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / "scripts" / "openeqa"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

builder = import_module("scripts.datasets.openeqa.build.build_openeqa_object_visual_index")


def _full_manifest(data_source: str, *, legacy_counters: bool = False) -> dict:
    totals = {
        "hsgg_objects": 3,
        "indexed": 3,
        "missing_embedding": 0,
    }
    if not legacy_counters:
        totals.update(
            {
                "selected_visual_anchor_objects": 3,
                "skipped_non_visual_anchor_objects": 0,
            }
        )
    return {
        "contract": OBJECT_VISUAL_INDEX_CONTRACT,
        "status": "passed",
        "data_source": data_source,
        "graph_manifest_run_id": f"{data_source}-run",
        "ntotal": 3,
        "metadata_rows": 3,
        "image_root": "full_frames",
        "totals": totals,
    }


def test_full_index_validation_is_identical_for_all_openeqa_sources() -> None:
    for data_source, legacy in (("hm3d-v0", True), ("scannet-v0", False)):
        issues = validate_object_visual_index_manifest(
            _full_manifest(data_source, legacy_counters=legacy),
            expected_data_source=data_source,
            expected_graph_manifest_run_id=f"{data_source}-run",
            require_full_coverage=True,
            forbidden_image_root_markers=("uniform32",),
        )

        assert issues == []


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    episode_history = "hm3d-v0/000-hm3d-test"
    hsgg_dir = tmp_path / "hsgg" / Path(episode_history)
    frames_dir = tmp_path / "frames" / Path(episode_history)
    graph_dir = tmp_path / "graphs" / Path(episode_history)
    image_path = frames_dir / "00000-rgb.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"not-read")
    box = [1.0, 2.0, 10.0, 20.0]
    _write_json(
        hsgg_dir / "objects.json",
        {
            "00000-rgb": {
                "image_path": "/cloud/data/00000-rgb.png",
                "boxes": [box, [5.0, 6.0, 15.0, 16.0]],
                "labels": ["chair", "table"],
                "scores": [0.9, 0.8],
            }
        },
    )
    frame_index_path = graph_dir / "frame_time_index.json"
    _write_json(
        frame_index_path,
        {
            "frames": [
                {
                    "frame_name": "frame_0",
                    "frame_index": 0,
                    "graph_frame_index": 0,
                    "source_frame_name": "00000-rgb",
                    "image_name": "00000-rgb.png",
                }
            ]
        },
    )
    temporal_path = graph_dir / "temporal_graph.json"
    _write_json(
        temporal_path,
        {
            "observations": [
                {
                    "observation_id": "obs-1",
                    "frame_name": "frame_0",
                    "frame_index": 0,
                    "object_index": 0,
                    "entity_id": "source-entity-1",
                    "scene_id": "scene-1",
                    "image_path": str(image_path),
                    "label": "chair",
                },
                {
                    "observation_id": "obs-2",
                    "frame_name": "frame_0",
                    "frame_index": 0,
                    "object_index": 1,
                    "entity_id": "source-entity-2",
                    "scene_id": "scene-1",
                    "image_path": str(image_path),
                    "label": "table",
                },
            ]
        },
    )
    canonical_path = graph_dir / "canonical_graph.json"
    _write_json(
        canonical_path,
        {
            "entity_nodes": [
                {
                    "entity_id": "canonical-chair",
                    "attributes": {"source_entity_ids": ["source-entity-1"]},
                }
            ]
        },
    )
    manifest_path = tmp_path / "graph_manifest.json"
    _write_json(
        manifest_path,
        {
            "run_id": "test-run",
            "data_source": "hm3d-v0",
            "episodes": [
                {
                    "episode_history": episode_history,
                    "hsgg_dir": str(hsgg_dir),
                    "frames_dir": str(frames_dir),
                    "frame_time_index_json": str(frame_index_path),
                    "temporal_graph_json": str(temporal_path),
                    "canonical_graph_json": str(canonical_path),
                    "status": "built",
                }
            ],
        },
    )
    cache_path = tmp_path / "embedding_cache.db"
    connection = sqlite3.connect(cache_path)
    connection.execute(
        """
        CREATE TABLE embedding_cache (
            image_path TEXT NOT NULL,
            bbox_key TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            PRIMARY KEY (image_path, bbox_key)
        )
        """
    )
    vector = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    connection.execute(
        "INSERT INTO embedding_cache VALUES (?, ?, ?, ?)",
        (str(image_path), builder.bbox_key(box), vector.tobytes(), vector.size),
    )
    connection.commit()
    connection.close()
    return {
        "manifest": manifest_path,
        "cache": cache_path,
        "output": tmp_path / "index",
        "temporal": temporal_path,
        "canonical": canonical_path,
    }


def test_graph_binding_requires_canonical_entity_mapping(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    index = GraphObservationBindingIndex.from_graphs(
        temporal_graph_path=paths["temporal"],
        canonical_graph_path=paths["canonical"],
    )

    bound = index.bind(frame_index=0, object_index=0)
    assert bound is not None
    assert bound.observation_id == "obs-1"
    assert bound.source_entity_id == "source-entity-1"
    assert bound.canonical_entity_id == "canonical-chair"
    observation_only = index.bind(frame_index=0, object_index=1)
    assert observation_only is not None
    assert observation_only.graph_bound is False
    assert observation_only.binding_confidence == 0.45


def test_builder_indexes_only_cached_exactly_bound_objects(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    args = builder.build_parser().parse_args(
        [
            "--graph-manifest",
            str(paths["manifest"]),
            "--cache-db",
            str(paths["cache"]),
            "--output-dir",
            str(paths["output"]),
        ]
    )

    manifest = builder.build_index(args)

    assert manifest["contract"] == OBJECT_VISUAL_INDEX_CONTRACT
    assert manifest["ntotal"] == 1
    assert manifest["totals"] == {
        "hsgg_objects": 2,
        "selected_visual_anchor_objects": 2,
        "skipped_non_visual_anchor_objects": 0,
        "visual_anchor_frames": 1,
        "valid_boxes": 2,
        "exact_graph_bindings": 1,
        "observation_only_bindings": 1,
        "indexed_exact_graph_bindings": 1,
        "indexed_observation_only_bindings": 0,
        "missing_graph_binding": 0,
        "embedding_cache_hits": 1,
        "missing_embedding": 1,
        "indexed": 1,
    }
    connection = sqlite3.connect(manifest["metadata_path"])
    row = connection.execute(
        "SELECT observation_id, source_entity_id, canonical_entity_id FROM object_metadata"
    ).fetchone()
    connection.close()
    assert row == ("obs-1", "source-entity-1", "canonical-chair")


def test_search_returns_only_episode_local_graph_bound_hits(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    args = builder.build_parser().parse_args(
        [
            "--graph-manifest",
            str(paths["manifest"]),
            "--cache-db",
            str(paths["cache"]),
            "--output-dir",
            str(paths["output"]),
        ]
    )
    builder.build_index(args)
    search = ObjectVisualSearchIndex(
        index_dir=paths["output"],
        text_encoder=lambda texts: [
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in texts
        ],
    )
    demand = compile_evidence_demand(
        "What color is the chair?",
        category="attribute recognition",
        question_ir={
            "target_entities": ["chair"],
            "attribute_dimension": "color",
        },
    ).to_dict()

    payload = search.search_evidence_demand(
        episode_history="hm3d-v0/000-hm3d-test",
        evidence_demand=demand,
        top_k=2,
    )
    search.close()

    assert payload["available"] is True
    hit = payload["groups"][0]["hits"][0]
    assert hit["canonical_entity_id"] == "canonical-chair"
    assert hit["binding_kind"] == "exact_frame_object_to_canonical_entity"
    assert hit["lexical_label_bonus"] > 0.0
    assert hit["uses_gold_answer"] is False
    assert payload["episode_scoped_search"] is True
    assert payload["global_topk_then_filter"] is False
    unit_counts = payload["retrieved_unit_group_counts"]
    assert unit_counts["object_crop"] >= 1
    assert unit_counts["owner_attribute"] >= 1
    owner_group = next(
        group for group in payload["groups"] if group["retrieved_unit"] == "owner_attribute"
    )
    assert owner_group["hits"][0]["retrieval_kind"] == (
        "canonical_entity_tracklet_embedding_aggregate"
    )


def test_open_spatial_slot_expands_objects_from_anchor_frames(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    connection = sqlite3.connect(paths["cache"])
    vector = np.asarray([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
    connection.execute(
        "INSERT INTO embedding_cache VALUES (?, ?, ?, ?)",
        (
            str(tmp_path / "frames" / "hm3d-v0" / "000-hm3d-test" / "00000-rgb.png"),
            builder.bbox_key([5.0, 6.0, 15.0, 16.0]),
            vector.tobytes(),
            vector.size,
        ),
    )
    connection.commit()
    connection.close()
    args = builder.build_parser().parse_args(
        [
            "--graph-manifest",
            str(paths["manifest"]),
            "--cache-db",
            str(paths["cache"]),
            "--output-dir",
            str(paths["output"]),
        ]
    )
    builder.build_index(args)
    search = ObjectVisualSearchIndex(
        index_dir=paths["output"],
        text_encoder=lambda texts: [
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            for _ in texts
        ],
    )
    demand = compile_evidence_demand(
        "What is to the left of the chair?",
        category="spatial understanding",
        question_ir={
            "target_entities": ["chair"],
            "spatial_constraints": [{"relation": "left of", "target": "chair"}],
        },
    ).to_dict()

    payload = search.search_evidence_demand(
        episode_history="hm3d-v0/000-hm3d-test",
        evidence_demand=demand,
        top_k=1,
    )
    search.close()

    expansion = next(
        group
        for group in payload["groups"]
        if group["query_intent"]["role"] == "spatial_target"
    )
    assert payload["coframe_expansion_hit_count"] == 1
    assert expansion["hits"][0]["label"] == "table"
    assert expansion["hits"][0]["retrieval_kind"] == "anchor_coframe_expansion"


def test_full_visual_manifest_rejects_uniform32_partial_index() -> None:
    issues = validate_object_visual_index_manifest(
        {
            "contract": OBJECT_VISUAL_INDEX_CONTRACT,
            "status": "passed",
            "data_source": "scannet-v0",
            "graph_manifest_run_id": "full-graph-v1",
            "image_root": "D:/frames/scannet_uniform32_v1",
            "ntotal": 32,
            "metadata_rows": 32,
            "totals": {
                "hsgg_objects": 100,
                "selected_visual_anchor_objects": 32,
                "skipped_non_visual_anchor_objects": 68,
                "missing_embedding": 0,
                "indexed": 32,
            },
        },
        expected_data_source="scannet-v0",
        expected_graph_manifest_run_id="full-graph-v1",
        require_full_coverage=True,
        forbidden_image_root_markers=("uniform32",),
    )

    assert any("forbidden marker" in issue for issue in issues)
    assert any("does not cover all HSGG objects" in issue for issue in issues)
    assert any("skipped non-anchor objects" in issue for issue in issues)


def test_full_visual_manifest_accepts_complete_full_index() -> None:
    issues = validate_object_visual_index_manifest(
        {
            "contract": OBJECT_VISUAL_INDEX_CONTRACT,
            "status": "passed",
            "data_source": "scannet-v0",
            "graph_manifest_run_id": "full-graph-v1",
            "image_root": "D:/frames/scannet_full_stride2_v1",
            "ntotal": 100,
            "metadata_rows": 100,
            "totals": {
                "hsgg_objects": 100,
                "selected_visual_anchor_objects": 100,
                "skipped_non_visual_anchor_objects": 0,
                "missing_embedding": 0,
                "indexed": 100,
            },
        },
        expected_data_source="scannet-v0",
        expected_graph_manifest_run_id="full-graph-v1",
        require_full_coverage=True,
        forbidden_image_root_markers=("uniform32",),
    )

    assert issues == []
