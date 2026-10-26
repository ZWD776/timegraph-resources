from __future__ import annotations

import argparse
from pathlib import Path

from scripts.prepare_navqa_sequence import build_steps
from vlkg.datasets.navqa.runtime.paths import default_navqa_sequence_layout


def _args(**overrides: object) -> argparse.Namespace:
    values = {
        "stages": ["temporal-graph"],
        "hsgg_output_dir": Path("upstream_hsgg"),
        "use_raw_hsgg_paths": False,
        "qa_unfilled_json": None,
        "timestamps": None,
        "layout_root": Path(".scratch/navqa"),
        "seq_id": "3",
        "cam": "0",
        "stride": 10,
        "csv": Path(".scratch/navqa_official_data.csv"),
        "config": "default.yaml",
        "answer_mode": "live",
        "retrieval_limit": 10,
        "domain_profile": "generic",
        "question_ir_json": None,
        "video_id": None,
        "workers": 24,
        "include_manifest_images": False,
        "force_rewrite": False,
        "compact_scene_label_entities": False,
        "include_position": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_temporal_graph_stage_reuses_existing_local_hsgg_paths(tmp_path: Path) -> None:
    layout_root = tmp_path / "navqa"
    layout = default_navqa_sequence_layout(
        seq_id="3",
        cam="0",
        stride=10,
        layout_root=layout_root,
    )
    layout.hsgg_local_paths_dir.mkdir(parents=True)
    for name in (
        "objects.json",
        "attributes.json",
        "relations.json",
        "hierarchy_mapping.json",
    ):
        (layout.hsgg_local_paths_dir / name).write_text("{}", encoding="utf-8")

    steps = build_steps(
        _args(layout_root=layout_root, hsgg_output_dir=tmp_path / "upstream_hsgg"),
        layout,
    )

    temporal_step = next(step for step in steps if step.name == "temporal-graph")
    assert temporal_step.command is not None
    assert str(layout.hsgg_local_paths_dir) in temporal_step.command
    assert str(tmp_path / "upstream_hsgg") not in temporal_step.command


def test_temporal_graph_stage_can_force_raw_hsgg_paths(tmp_path: Path) -> None:
    layout_root = tmp_path / "navqa"
    layout = default_navqa_sequence_layout(
        seq_id="3",
        cam="0",
        stride=10,
        layout_root=layout_root,
    )
    raw_hsgg = tmp_path / "upstream_hsgg"

    steps = build_steps(
        _args(
            layout_root=layout_root,
            hsgg_output_dir=raw_hsgg,
            use_raw_hsgg_paths=True,
        ),
        layout,
    )

    temporal_step = next(step for step in steps if step.name == "temporal-graph")
    assert temporal_step.command is not None
    assert str(raw_hsgg) in temporal_step.command
