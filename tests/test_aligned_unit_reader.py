from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/release/read_aligned_unit.py"
SPEC = importlib.util.spec_from_file_location("aligned_reader", SCRIPT)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_tree(tmp_path):
    unit = "navqa/seq0/cam0/every10"
    group = "navqa_seq0_cam0_every10"
    base = "__group_features__/" + group
    binding = {"dataset": "navqa", "unit_id": "0", "group_id": group, "unit_archive_root": unit,
               "timegraph": {"canonical": unit + "/timegraph/canonical_temporal_memory_graph.json",
                             "temporal": unit + "/timegraph/temporal_graph.json"},
               "group_object_metadata": base + "/object.metadata.jsonl", "group_object_index": base + "/object.faiss"}
    dump(tmp_path / unit / "visual_features/unit_binding.json", binding)
    metadata = {"unit_id": "0", "faiss_id": 0, "local_faiss_id": 0, "original_faiss_id": 91,
                "image_path": unit + "/frames/f.jpg", "frame_key": "f"}
    dump(tmp_path / base / "object.metadata.jsonl", metadata)
    header = bytearray(45)
    header[:4] = b"IxFI"
    struct.pack_into("<i", header, 4, 768)
    struct.pack_into("<q", header, 8, 1)
    header[32] = 1
    struct.pack_into("<Q", header, 37, 768)
    index = tmp_path / base / "object.faiss"
    index.write_bytes(header + struct.pack("<768f", *([0.25] * 768)))
    for path in binding["timegraph"].values():
        dump(tmp_path / path, {})
    images, hsgg = tmp_path / "flat_images", tmp_path / "old_hsgg"
    images.mkdir()
    (images / "f.jpg").write_bytes(b"existing-image")
    dump(hsgg / "frame_scene_graphs/f.json", {"objects": []})
    return unit, images, hsgg, index


def test_real_layout_uses_explicit_flat_image_and_hsgg_roots(tmp_path):
    unit, images, hsgg, _ = fixture_tree(tmp_path)
    result = reader.read_unit(tmp_path, unit, images_root=images, hsgg_root=hsgg, include_vector=True)
    assert result["status"] == "passed"
    assert result["original_faiss_id"] == 91
    assert result["local_faiss_id"] == 0
    assert len(result["vector"]) == 768
    assert result["vector"][0] == 0.25
    assert result["binding_checks"]["hsgg_record_key"] == "f"
    assert result["models_or_apis_called"] is False


def test_does_not_guess_flat_image_location(tmp_path):
    unit, _, hsgg, _ = fixture_tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="images-root"):
        reader.read_unit(tmp_path, unit, hsgg_root=hsgg)


def test_truncated_or_wrong_metric_index_rejected(tmp_path):
    unit, images, hsgg, index = fixture_tree(tmp_path)
    data = bytearray(index.read_bytes())
    struct.pack_into("<i", data, 33, 1)
    index.write_bytes(data)
    with pytest.raises(ValueError, match="header"):
        reader.read_unit(tmp_path, unit, images_root=images, hsgg_root=hsgg)
    data[33] = 0
    index.write_bytes(data[:-4])
    with pytest.raises(ValueError, match="payload size"):
        reader.read_unit(tmp_path, unit, images_root=images, hsgg_root=hsgg)


def test_requested_local_id_must_belong_to_this_unit(tmp_path):
    unit, images, hsgg, _ = fixture_tree(tmp_path)
    with pytest.raises(ValueError, match="No existing vector"):
        reader.read_unit(tmp_path, unit, images_root=images, hsgg_root=hsgg, row_id=91)


@pytest.mark.parametrize("value", [".", "../escape", "/absolute", "C:/machine", "unit\\name"])
def test_untrusted_package_path_cannot_escape_selected_root(tmp_path, value):
    with pytest.raises(ValueError, match="relative POSIX"):
        reader.bounded(tmp_path, value)


def test_nextqa_requires_frame_key_in_all_four_normalized_tables(tmp_path):
    hsgg = tmp_path / "hsgg"
    for name in ("objects", "attributes", "relations", "hierarchy_mapping"):
        dump(hsgg / (name + ".json"), {"frame1": []} if name != "relations" else {})
    image = tmp_path / "images/frame1.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    binding = {"dataset": "nextqa", "unit_archive_root": "nextqa-val/1"}
    row = {"frame_name": "frame1", "image_path": "nextqa-val/1/frames/frame1.png"}
    with pytest.raises(ValueError, match="all four"):
        reader.check_observation(binding, row, tmp_path, image.parent, hsgg, None)


def test_nextqa_cloud_sidecar_layout_and_actual_image_binding(tmp_path):
    unit = "nextqa-val/1"
    image = tmp_path / unit / "frames/f.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    dump(tmp_path / unit / "hsgg/frame_scene_graphs/f.json", {"image_path": "/old/machine/f.png", "objects": {}})
    graphs = {name: unit + "/timegraph/" + name + ".json" for name in ("canonical", "temporal")}
    for graph in graphs.values():
        dump(tmp_path / graph, {})
    binding = {"dataset": "nextqa", "unit_archive_root": unit, "timegraph": graphs}
    result = reader.check_observation(binding, {"image_path": unit + "/frames/f.png", "frame_name": "f"}, tmp_path, None, None, None)
    assert result["hsgg_representation"] == "completed_frame_sidecar"
    assert result["sidecar_image_basename_matches"] is True
    dump(tmp_path / unit / "hsgg/frame_scene_graphs/f.json", {"image_path": "/old/machine/other.png"})
    with pytest.raises(ValueError, match="different image basename"):
        reader.check_observation(binding, {"image_path": unit + "/frames/f.png", "frame_name": "f"}, tmp_path, None, None, None)


def test_metadata_frame_key_cannot_traverse_hsgg_root(tmp_path):
    with pytest.raises(ValueError, match="relative POSIX|single safe"):
        reader.check_observation({"dataset": "navqa", "unit_archive_root": "navqa/seq0"},
                                 {"image_path": "navqa/seq0/f.png", "frame_key": "../outside"},
                                 tmp_path, None, None, None)
