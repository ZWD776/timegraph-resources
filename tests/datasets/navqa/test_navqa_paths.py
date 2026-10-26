from __future__ import annotations

from pathlib import Path

from vlkg.datasets.navqa.runtime.paths import (
    coda_sequence_zip_url,
    default_navqa_sequence_layout,
)


def test_default_navqa_sequence_layout_uses_sequence_sample_namespace() -> None:
    layout = default_navqa_sequence_layout(
        seq_id="12",
        cam="cam1",
        stride=5,
        layout_root=Path(".scratch/navqa"),
    )

    assert layout.sequence_root == Path(".scratch/navqa/seq12")
    assert layout.sample_root == Path(".scratch/navqa/seq12/cam1/every5")
    assert layout.images_dir == layout.sample_root / "images"
    assert layout.hsgg_local_paths_dir == layout.sample_root / "hsgg_local_paths"
    assert layout.frame_time_index_json == layout.sample_root / "frame_time_index.json"
    assert layout.temporal_graph_json == layout.sample_root / "temporal_graph.json"
    assert layout.canonical_graph_json == (
        layout.sample_root / "canonical_temporal_memory_graph.json"
    )
    assert layout.qa_results_json == layout.sample_root / "qa_non_position_live_results.json"


def test_coda_sequence_zip_url_is_parameterized_by_sequence() -> None:
    assert coda_sequence_zip_url("7").endswith("/sequences/7.zip")
