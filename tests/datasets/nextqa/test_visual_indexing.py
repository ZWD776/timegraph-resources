from __future__ import annotations

import numpy as np
from pathlib import Path

from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import (
    bbox_key,
    build_before_after_candidates,
    build_temporal_window_candidates,
    l2_normalize,
    mean_normalized,
)


def test_vector_normalization_and_bbox_key_are_stable() -> None:
    vector = l2_normalize([3.0, 4.0])
    assert np.allclose(vector, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(mean_normalized([vector, vector])), 1.0)
    assert bbox_key({"x1": 1, "y1": 2.1234, "x2": 3.5, "y2": 4}) == (
        "1.000,2.123,3.500,4.000"
    )


def test_visual_index_uses_selected_item_frames_instead_of_historical_graph_path(tmp_path: Path) -> None:
    from scripts.datasets.nextqa.build.build_visual_indexes import resolve_selected_frame_dir
    video = {'input_storage': 'normalized_directory', 'normalized_input_dir': 'items/100', 'frames_dir': 'items/100/frames'}
    graph = {'source_dir': str(tmp_path / 'removed_first_half_batch' / '0001_100')}
    assert resolve_selected_frame_dir(video, graph, repository_root=tmp_path) == tmp_path / 'items/100/frames'


def test_visual_index_keeps_legacy_archive_selection_frame_resolution(tmp_path: Path) -> None:
    from scripts.datasets.nextqa.build.build_visual_indexes import resolve_selected_frame_dir
    assert resolve_selected_frame_dir({}, {'source_dir': 'old_inputs/0001_100'}, repository_root=tmp_path) == tmp_path / 'old_inputs/0001_100/frames'


def _frames() -> tuple[list[dict[str, object]], list[np.ndarray]]:
    frames = [
        {
            "video_id": "100",
            "frame_name": f"frame_{index:06d}_{timestamp:04.1f}s",
            "timestamp_sec": timestamp,
            "image_path": f"frame-{index}.png",
            "labels": [label],
            "states": [],
            "entity_ids": [f"entity:{index}"],
            "graph_bound": True,
        }
        for index, timestamp, label in (
            (0, 0.0, "person"),
            (30, 1.0, "cup"),
            (60, 2.0, "table"),
            (90, 3.0, "person"),
        )
    ]
    vectors = [l2_normalize([1.0, float(index + 1)]) for index in range(4)]
    return frames, vectors


def test_temporal_windows_use_seconds_not_frame_ordinals() -> None:
    frames, vectors = _frames()
    metadata, output_vectors = build_temporal_window_candidates(
        {"video_id": "nextqa-val/100", "relation_episodes": [], "transition_events": []},
        frames,
        vectors,
        window_size_sec=2.0,
        window_stride_sec=2.0,
    )
    assert [(item["start_ts"], item["end_ts"]) for item in metadata] == [
        (0.0, 2.0),
        (2.0, 3.0),
    ]
    assert all(np.isclose(np.linalg.norm(vector), 1.0) for vector in output_vectors)


def test_before_after_pair_keeps_both_sides() -> None:
    frames, vectors = _frames()
    graph = {
        "video_id": "nextqa-val/100",
        "transition_events": [
            {"event_id": "event:1", "event_type": "RELATION_START", "timestamp_sec": 1.5}
        ],
    }
    metadata, output_vectors = build_before_after_candidates(graph, frames, vectors)
    assert metadata[0]["before_ts"] == 1.0
    assert metadata[0]["after_ts"] == 2.0
    assert np.isclose(np.linalg.norm(output_vectors[0]), 1.0)
