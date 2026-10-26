from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
    / "scripts"
    / "build_navqa_frame_time_index.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("build_navqa_frame_time_index", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_image_supports_coda_sampled_name_without_graph_frame_prefix() -> None:
    module = _load_module()

    row = module.parse_image(Path("000000_2d_rect_cam0_3_0.jpg"))

    assert row["sampled_index"] == 0
    assert row["graph_frame_index"] == 0
    assert row["source_frame"] == 0
    assert row["cam"] == 0
    assert row["seq"] == 3


def test_parse_image_keeps_legacy_graph_frame_prefix() -> None:
    module = _load_module()

    row = module.parse_image(Path("000079_000158_2d_rect_cam0_0_790.jpg"))

    assert row["sampled_index"] == 79
    assert row["graph_frame_index"] == 158
    assert row["source_frame"] == 790
