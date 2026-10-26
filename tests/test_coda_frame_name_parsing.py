from __future__ import annotations

from vlkg.core.transformers.json_to_graph import JSONToGraphTransformer


def test_json_to_graph_parses_coda_sampled_name_without_graph_frame_prefix() -> None:
    transformer = JSONToGraphTransformer(
        ".",
        frame_time_index_path=None,
    )
    transformer.frame_time_index = {0: 123.4}

    meta = transformer._parse_frame_name("000000_2d_rect_cam0_3_0")

    assert meta.frame_index == 0
    assert meta.timestamp_sec == 123.4


def test_json_to_graph_keeps_legacy_coda_graph_frame_prefix() -> None:
    transformer = JSONToGraphTransformer(
        ".",
        frame_time_index_path=None,
    )
    transformer.frame_time_index = {158: 456.7}

    meta = transformer._parse_frame_name("000079_000158_2d_rect_cam0_0_790")

    assert meta.frame_index == 158
    assert meta.timestamp_sec == 456.7
