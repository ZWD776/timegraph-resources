from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from vlkg.datasets.navqa.retrieval.evidence_layer_index import (
    build_frame_evidence_records,
    iter_frame_payloads,
    load_evidence_sequence_specs,
)
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import _hsgg_root_for_metadata_row
from vlkg.datasets.navqa.runtime import paths


def write_json(path: Path, payload: object) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload).encode("utf-8")
    path.write_bytes(data)
    return data


@pytest.mark.parametrize("old_image_exists", [False, True])
def test_explicit_source_paths_read_native_json_without_rewriting_old_image_path(
    tmp_path: Path, old_image_exists: bool
) -> None:
    sample = tmp_path / "seq6" / "cam0" / "every10"
    hsgg = sample / "hsgg"
    images = tmp_path / "selected_images"
    images.mkdir()
    name = "000001_2d_rect_cam0_6_10"
    image = images / f"{name}.jpg"
    image.write_bytes(b"selected-image")
    old_image = tmp_path / "old-location" / image.name
    if old_image_exists:
        old_image.parent.mkdir()
        old_image.write_bytes(b"obsolete-image")
    native_path = hsgg / "frame_scene_graphs" / f"{name}.json"
    original = write_json(native_path, {
        "image_path": str(old_image),
        "objects": {"image_path": str(old_image), "labels": ["person"], "boxes": [[0, 0, 10, 10]], "scores": [0.9]},
        "attributes": {"0": "person wearing blue"},
        "relations": [],
    })
    manifest = tmp_path / "source_manifest.json"
    write_json(manifest, {"sequences": [{"seq_id": "6", "sample_root": str(sample), "hsgg_input_dir": str(hsgg), "images_dir": str(images)}]})
    spec = load_evidence_sequence_specs(sequence_manifest=manifest)[0]
    assert spec.hsgg_input_dir == hsgg
    assert spec.images_dir == images
    frames = list(iter_frame_payloads(spec))
    assert len(frames) == 1
    records = build_frame_evidence_records(spec=spec, frame_key=frames[0][0], frame_payload=frames[0][1])
    assert records.observations[0]["image_path"] == str(image)
    assert records.observations[0]["image_exists"] is True
    assert native_path.read_bytes() == original


def test_legacy_sequence_manifest_keeps_original_layout_defaults(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    sample = tmp_path / "sample"
    write_json(manifest, {"sequences": [{"seq_id": "6", "sample_root": str(sample)}]})
    spec = load_evidence_sequence_specs(sequence_manifest=manifest)[0]
    assert spec.hsgg_input_dir == sample / "hsgg_local_paths"
    assert spec.images_dir == sample / "images"


def test_default_layout_uses_source_manifest_but_explicit_legacy_root_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paths, "workspace_root", lambda: tmp_path)
    default_root = paths.DEFAULT_NAVQA_LAYOUT_ROOT
    sample = default_root / "seq6" / "cam0" / "every10"
    write_json(tmp_path / default_root.parent / "source_manifest.json", {"sequences": [{
        "seq_id": "6", "cam": "0", "stride": 10,
        "hsgg_input_dir": (sample / "hsgg").as_posix(),
        "images_dir": (sample / "images").as_posix(),
    }]})
    layout = paths.default_navqa_sequence_layout(seq_id="6")
    assert layout.hsgg_local_paths_dir == sample / "hsgg"
    absolute = paths.default_navqa_sequence_layout(seq_id="6", layout_root=tmp_path / default_root)
    assert absolute.hsgg_local_paths_dir == tmp_path / sample / "hsgg"
    legacy_root = tmp_path / "legacy"
    legacy = paths.default_navqa_sequence_layout(seq_id="6", layout_root=legacy_root)
    assert legacy.hsgg_local_paths_dir == legacy.sample_root / "hsgg_local_paths"


def test_visual_metadata_explicit_hsgg_root_precedes_path_inference(tmp_path: Path) -> None:
    sample = tmp_path / "sample"
    inferred = sample / "hsgg"
    explicit = tmp_path / "selected_hsgg"
    inferred.mkdir(parents=True)
    explicit.mkdir()
    assert _hsgg_root_for_metadata_row({"hsgg_input_dir": str(explicit), "image_path": str(sample / "images" / "frame.jpg")}) == explicit


@pytest.mark.parametrize("directory_name", ["hsgg", "hsgg_local_paths"])
def test_visual_metadata_supports_new_and_legacy_adjacent_hsgg_roots(
    tmp_path: Path, directory_name: str
) -> None:
    hsgg = tmp_path / directory_name
    hsgg.mkdir()
    assert _hsgg_root_for_metadata_row({"image_path": str(tmp_path / "images" / "frame.jpg")}) == hsgg


def test_visual_index_respects_manifest_input_dirs_and_emits_hsgg_reference(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.datasets.navqa.build.build_navqa_visual_faiss_index")
    sample = tmp_path / "sample"
    hsgg = sample / "hsgg"
    images = tmp_path / "frames"
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"sequences": [{"seq_id": "6", "sample_root": str(sample), "hsgg_input_dir": str(hsgg), "images_dir": str(images)}]})
    spec = module.load_sequences(manifest, None)[0]
    assert spec.hsgg_input_dir == hsgg
    assert spec.images_dir == images
    metadata = module.metadata_for_object(faiss_id=0, spec=spec, frame_key="frame", payload={"labels": ["person"]}, frame_info={}, object_index=0, box=[0, 0, 10, 10], cache_image_path="original-cache-key.jpg", cache_bbox_key="0,0,10,10", image_path=images / "frame.jpg")
    assert metadata["hsgg_input_dir"] == str(hsgg)
    assert metadata["cache_image_path"] == "original-cache-key.jpg"
