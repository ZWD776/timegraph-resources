"""Directory inputs remain usable after local archive copies are removed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.datasets.activate_nextqa_mainline import validate_source_manifest_paths
from vlkg.datasets.nextqa.runtime.contracts import (
    ARCHIVE_VIDEO_FIELDS,
    graph_safe_video_record,
    validate_manifest_input_storage,
)


def directory_video() -> dict:
    return {
        'episode_key': 'nextqa-val/100',
        'official_video_id': '100',
        'vidor_id': '0001/100',
        'local_sequential_id': 1,
        'hsgg_record_source': 'frame_scene_graphs_members',
        'hsgg_generation_pass': 'test',
        'hsgg_frame_count': 1,
        'sampled_image_count': 2,
        'skipped_frame_count': 1,
        'start_timestamp_seconds': 0.0,
        'end_timestamp_seconds': 1.001,
        'official_frame_count': 100,
        'inferred_source_fps': 29.97,
        'median_sample_interval_seconds': 1.001,
        'timestamp_consistency_status': 'passed',
        'input_storage': 'normalized_directory',
        'normalized_input_dir': 'datasets/nextqa/source/selected/payload/0001_100',
        'frames_dir': 'datasets/nextqa/source/selected/payload/0001_100/frames',
    }


def test_directory_record_requires_no_archive_fields_and_excludes_evaluator_data():
    video = directory_video()
    video.update(question='hidden', answer='hidden', answer_choices=['hidden'])
    record = graph_safe_video_record(video)
    assert not set(record).intersection(ARCHIVE_VIDEO_FIELDS)
    assert not {'question', 'answer', 'answer_choices'}.intersection(record)
    assert record['hsgg_frame_count'] == 1
    assert record['skipped_frame_count'] == 1
    assert record['end_timestamp_seconds'] == 1.001


@pytest.mark.parametrize('field', ['normalized_input_dir', 'frames_dir'])
def test_directory_record_rejects_missing_and_empty_paths(field):
    video = directory_video()
    del video[field]
    with pytest.raises(ValueError, match='missing fields'):
        graph_safe_video_record(video)
    video[field] = ''
    with pytest.raises(ValueError, match='nonempty'):
        graph_safe_video_record(video)


def test_manifest_rejects_storage_mismatch_and_unknown_storage():
    video = directory_video()
    with pytest.raises(ValueError, match='must agree'):
        validate_manifest_input_storage({'input_storage': 'archive', 'videos': [video]})
    with pytest.raises(ValueError, match='unsupported'):
        graph_safe_video_record(dict(video, input_storage='unknown'))


def test_activation_checks_relative_directory_inputs_without_archive_files(tmp_path: Path):
    video = directory_video()
    video.update(hsgg_archive='deleted/hsgg.tar.gz', frame_archive='deleted/frames.tar.gz')
    input_dir = tmp_path / video['normalized_input_dir']
    frames_dir = tmp_path / video['frames_dir']
    frames_dir.mkdir(parents=True)
    for name in ('objects.json', 'attributes.json', 'relations.json', 'hierarchy_mapping.json', 'timeline.json', 'source_trace.json'):
        (input_dir / name).write_text(json.dumps({}), encoding='utf-8')
    manifest = {'input_storage': 'normalized_directory', 'videos': [video]}
    validate_source_manifest_paths(manifest, root=tmp_path)
    (input_dir / 'timeline.json').unlink()
    with pytest.raises(RuntimeError, match='timeline.json'):
        validate_source_manifest_paths(manifest, root=tmp_path)
