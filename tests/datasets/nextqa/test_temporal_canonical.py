from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vlkg.datasets.nextqa.graph.hsgg_input_adapter import normalize_hsgg_archive
from vlkg.datasets.nextqa.graph.temporal_canonical import (
    build_video_graph,
    load_nextqa_graph_config,
)

from .conftest import make_archive_and_video


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())


class FakeVisionFeatureExtractor:
    def __init__(self) -> None:
        self.request_count = 0

    def encode_batch(self, items):
        self.request_count += len(items)
        embeddings = []
        for _, bbox in items:
            vector = np.asarray(
                [bbox.x1, bbox.y1, bbox.x2, bbox.y2, 1.0], dtype=np.float32
            )
            embeddings.append(vector / np.linalg.norm(vector))
        return embeddings


class FakeEmbeddingCache:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.values = {}
        self.reset_stats()

    @staticmethod
    def bbox_to_key(bbox) -> str:
        return f"{bbox.x1:.3f},{bbox.y1:.3f},{bbox.x2:.3f},{bbox.y2:.3f}"

    def reset_stats(self) -> None:
        self.stats = {"requests": 0, "hits": 0, "misses": 0, "writes": 0}

    def get_many(self, items):
        result = {}
        for image_path, bbox in items:
            self.stats["requests"] += 1
            key = (str(Path(image_path)), self.bbox_to_key(bbox))
            if key in self.values:
                self.stats["hits"] += 1
                result[key] = self.values[key]
            else:
                self.stats["misses"] += 1
        return result

    def set_many(self, items) -> None:
        self.stats["writes"] += len(items)
        for image_path, bbox, embedding in items:
            key = (str(Path(image_path)), self.bbox_to_key(bbox))
            self.values[key] = embedding

    def get_stats(self):
        requests = self.stats["requests"]
        return {
            **self.stats,
            "backend": "fake_run_scoped",
            "hit_rate": self.stats["hits"] / requests if requests else 0.0,
        }

    def close(self) -> None:
        return None


def _config():
    return load_nextqa_graph_config(
        REPO_ROOT / 'configs' / 'datasets' / 'nextqa' / 'graph.yaml',
        config_dir=REPO_ROOT / "configs",
    )


def _build(nextqa_tmp_dir: Path, video: dict):
    [normalized] = normalize_hsgg_archive(
        [video], output_root=nextqa_tmp_dir / "normalized"
    )
    extractor = FakeVisionFeatureExtractor()
    cache = FakeEmbeddingCache(nextqa_tmp_dir / "cache" / "siglip.sqlite")
    built = build_video_graph(
        normalized,
        config=_config(),
        output_root=nextqa_tmp_dir / "graphs",
        vision_feature_extractor=extractor,
        embedding_cache=cache,
    )
    return normalized, extractor, cache, built


def test_graph_build_uses_crop_embeddings_only_during_construction(
    nextqa_tmp_dir: Path,
) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[
            ("frame_000000_0000.000s", "completed"),
            ("frame_000030_0001.001s", "completed"),
        ],
    )

    _, extractor, cache, built = _build(nextqa_tmp_dir, video)

    assert built.audit["status"] == "passed"
    assert extractor.request_count == 4
    assert cache.get_stats()["requests"] == 4
    assert built.audit["construction_vs_qa_purity"] == {
        "uses_visual_embeddings_for_graph_construction": True,
        "qa_reads_rgb": False,
        "qa_reads_object_crop": False,
        "qa_reads_embedding_cache": False,
        "qa_uses_faiss": False,
    }
    temporal = json.loads(built.temporal_graph_path.read_text())
    report = json.loads(built.canonical_report_path.read_text())
    assert temporal["frame_count"] == 2
    assert temporal["observations"][0]["timestamp_sec"] == 0.0
    assert report["contract"] == "time_fact_core_hsgg_to_graph_pipeline_v1"
    assert "questions" not in built.temporal_graph_path.read_text().lower()
    assert "gold_choice_index" not in built.canonical_graph_path.read_text().lower()


def test_zero_hsgg_input_builds_an_empty_graph_without_invented_evidence(
    nextqa_tmp_dir: Path,
) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[("frame_000000_0000.000s", "skipped_no_scene_graph")],
        include_conflicting_aggregate=False,
    )

    _, extractor, cache, built = _build(nextqa_tmp_dir, video)

    temporal = json.loads(built.temporal_graph_path.read_text())
    canonical = json.loads(built.canonical_graph_path.read_text())
    assert temporal["frame_count"] == 1
    assert temporal["observations"] == []
    assert temporal["entities"] == []
    assert canonical["entity_nodes"] == []
    assert extractor.request_count == 0
    assert cache.get_stats()["requests"] == 0
    assert built.audit["input"]["skipped_frame_count"] == 1


def test_materialized_frames_override_stale_paths_without_rewriting_inputs(
    nextqa_tmp_dir: Path,
) -> None:
    frame_name = "frame_000000_0000.000s"
    _, video = make_archive_and_video(
        nextqa_tmp_dir, frame_statuses=[(frame_name, "completed")]
    )
    [normalized] = normalize_hsgg_archive(
        [video], output_root=nextqa_tmp_dir / "normalized"
    )
    objects_path = normalized.input_dir / "objects.json"
    objects = json.loads(objects_path.read_text())
    objects[frame_name]["image_path"] = "/missing/historical/frame.png"
    objects_path.write_text(json.dumps(objects), encoding="utf-8")
    input_bytes = objects_path.read_bytes()

    class ExistingFramesExtractor(FakeVisionFeatureExtractor):
        def encode_batch(self, items):
            for image_path, _ in items:
                assert Path(image_path).is_file()
                assert Path(image_path).parent == normalized.materialized_frame_root
            return super().encode_batch(items)

    built = build_video_graph(
        normalized,
        config=_config(),
        output_root=nextqa_tmp_dir / "graphs",
        vision_feature_extractor=ExistingFramesExtractor(),
        embedding_cache=FakeEmbeddingCache(nextqa_tmp_dir / "cache" / "siglip.sqlite"),
    )
    assert built.audit["status"] == "passed"
    assert objects_path.read_bytes() == input_bytes
    temporal = json.loads(built.temporal_graph_path.read_text())
    assert all(Path(item["image_path"]).is_file() for item in temporal["observations"])


def test_transformer_resolves_item_relative_image_paths_without_frame_override(nextqa_tmp_dir: Path) -> None:
    from vlkg.core.transformers.json_to_graph import JSONToGraphTransformer
    name = 'frame_000000_0000.000s'
    _, video = make_archive_and_video(nextqa_tmp_dir, frame_statuses=[(name, 'completed')])
    [normalized] = normalize_hsgg_archive([video], output_root=nextqa_tmp_dir / 'normalized')
    path = normalized.input_dir / 'objects.json'
    payload = json.loads(path.read_text())
    payload[name]['image_path'] = f'frames/{name}.png'
    path.write_text(json.dumps(payload), encoding='utf-8')
    source_bytes = path.read_bytes()
    class ExistingFramesExtractor(FakeVisionFeatureExtractor):
        def encode_batch(self, items):
            assert all(Path(image).is_file() for image, _ in items)
            return super().encode_batch(items)
    graph = JSONToGraphTransformer(input_dir=normalized.input_dir, video_id=normalized.episode_key, config=_config(), vision_feature_extractor=ExistingFramesExtractor(), embedding_cache=FakeEmbeddingCache(nextqa_tmp_dir / 'cache/test.sqlite')).transform()
    assert graph.observations
    assert all(Path(item.image_path).parent == normalized.materialized_frame_root for item in graph.observations)
    assert path.read_bytes() == source_bytes


def test_component_objects_and_relations_are_retained(nextqa_tmp_dir: Path) -> None:
    frame_name = "frame_000000_0000.000s"
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[(frame_name, "completed")],
        second_labels={frame_name: "shirt"},
        component_inside_host_frames={frame_name},
    )

    _, _, _, built = _build(nextqa_tmp_dir, video)

    temporal = json.loads(built.temporal_graph_path.read_text())
    retention = built.audit["temporal_graph"]["object_retention"]
    disposition = built.audit["temporal_graph"]["relation_source_disposition"]
    assert built.audit["status"] == "passed"
    assert len(temporal["observations"]) == 2
    assert {item["label"] for item in temporal["observations"]} == {"person", "shirt"}
    assert retention["component_policy_excluded_count"] == 0
    assert disposition["disposition_counts"] == {"retained": 1}


def test_static_lexical_label_remains_a_dynamic_tracklet(nextqa_tmp_dir: Path) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[
            ("frame_000000_0000.000s", "completed"),
            ("frame_000030_0001.001s", "completed"),
        ],
        labels={
            "frame_000000_0000.000s": "wall",
            "frame_000030_0001.001s": "wall",
        },
    )

    _, _, _, built = _build(nextqa_tmp_dir, video)

    temporal = json.loads(built.temporal_graph_path.read_text())
    wall_entities = [entity for entity in temporal["entities"] if entity["category"] == "wall"]
    assert len(wall_entities) == 1
    assert wall_entities[0]["is_static"] is False
    assert wall_entities[0]["observation_count"] == 2


def test_skipped_slot_breaks_local_tracklet_continuity(nextqa_tmp_dir: Path) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[
            ("frame_000000_0000.000s", "completed"),
            ("frame_000030_0001.001s", "skipped_no_scene_graph"),
            ("frame_000060_0002.002s", "completed"),
        ],
    )

    _, _, _, built = _build(nextqa_tmp_dir, video)

    temporal = json.loads(built.temporal_graph_path.read_text())
    assert len(temporal["observations"]) == 4
    assert len(temporal["entities"]) == 4
    assert built.audit["temporal_graph"]["tracklets"]["skipped_slot_bridge_count"] == 0


def test_time_fact_core_keeps_one_stable_identity_per_tracklet(
    nextqa_tmp_dir: Path,
) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[("frame_000000_0000.000s", "completed")],
    )

    _, _, _, built = _build(nextqa_tmp_dir, video)

    temporal = json.loads(built.temporal_graph_path.read_text())
    canonical = json.loads(built.canonical_graph_path.read_text())
    report = json.loads(built.canonical_report_path.read_text())
    assert report["parameters"]["stable_identity_merge_mode"] == "none"
    assert report["parameters"]["stable_identity_include_fragments"] is True
    assert report["parameters"]["stable_identity_include_noise"] is True
    assert len(canonical["stable_identity_nodes"]) == len(temporal["entities"])
