from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from vlkg.datasets.nextqa.graph.hsgg_input_adapter import load_normalized_video_input, normalize_hsgg_archive

from .conftest import make_archive_and_video


def test_per_frame_members_are_authoritative_and_stale_paths_are_replaced(
    nextqa_tmp_dir: Path,
) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[
            ("frame_000000_0000.000s", "completed"),
            ("frame_000030_0001.001s", "skipped_no_scene_graph"),
        ],
        labels={"frame_000000_0000.000s": "woman"},
    )

    [normalized] = normalize_hsgg_archive(
        [video], output_root=nextqa_tmp_dir / "normalized"
    )

    objects = json.loads((normalized.input_dir / "objects.json").read_text())
    assert set(objects) == {
        "frame_000000_0000.000s",
        "frame_000030_0001.001s",
    }
    assert objects["frame_000000_0000.000s"]["labels"][0] == "woman"
    rebuilt_path = objects["frame_000000_0000.000s"]["image_path"]
    assert "/cloud/" not in rebuilt_path
    assert Path(rebuilt_path).as_posix().endswith(
        "0001_100/frames/frame_000000_0000.000s.png"
    )
    assert Path(rebuilt_path).is_file()
    assert objects["frame_000030_0001.001s"]["boxes"] == []
    assert objects["frame_000030_0001.001s"]["stage3_skipped"] is True
    trace = json.loads(normalized.source_trace_path.read_text())
    assert trace["authoritative_record_source"] == "frame_scene_graphs/*.json"
    assert trace["aggregate_json_read"] is False
    assert trace["embedded_hsgg_image_paths_used"] is False
    assert trace["frames_materialized"] is True
    assert trace["materialized_frame_count"] == 2
    assert all(frame["frame_member_sha256"] for frame in trace["sampled_frames"])
    assert normalized.materialized_frame_count == 2


def test_mixed_fps_filename_timestamps_are_preserved(nextqa_tmp_dir: Path) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[
            ("frame_000000_0000.000s", "completed"),
            ("frame_000030_0001.251s", "completed"),
            ("frame_000060_0002.503s", "completed"),
        ],
    )

    [normalized] = normalize_hsgg_archive(
        [video], output_root=nextqa_tmp_dir / "normalized"
    )
    timeline = json.loads(normalized.timeline_path.read_text())

    assert [frame["frame_index"] for frame in timeline["frames"]] == [0, 30, 60]
    assert [frame["timestamp_sec"] for frame in timeline["frames"]] == [0.0, 1.251, 2.503]
    assert normalized.end_timestamp_seconds == 2.503


def test_zero_hsgg_video_becomes_an_empty_timeline_slot(nextqa_tmp_dir: Path) -> None:
    _, video = make_archive_and_video(
        nextqa_tmp_dir,
        frame_statuses=[("frame_000000_0000.000s", "skipped_no_scene_graph")],
        include_conflicting_aggregate=False,
    )

    [normalized] = normalize_hsgg_archive(
        [video], output_root=nextqa_tmp_dir / "normalized"
    )
    objects = json.loads((normalized.input_dir / "objects.json").read_text())
    relations = json.loads((normalized.input_dir / "relations.json").read_text())

    assert normalized.completed_frame_count == 0
    assert normalized.skipped_frame_count == 1
    assert objects["frame_000000_0000.000s"]["labels"] == []
    assert relations["frame_000000_0000.000s"] == []


def _materialized_video(root: Path) -> tuple[dict, Path]:
    _, video = make_archive_and_video(
        root,
        frame_statuses=[
            ('frame_000000_0000.000s', 'completed'),
            ('frame_000030_0001.001s', 'skipped_no_scene_graph'),
        ],
    )
    [item] = normalize_hsgg_archive([video], output_root=root / 'old_inputs')
    new_dir = root / 'relocated_inputs' / item.input_dir.name
    new_dir.parent.mkdir()
    item.input_dir.rename(new_dir)
    video.update(input_storage='normalized_directory', normalized_input_dir=str(new_dir.relative_to(root)), frames_dir=str((new_dir / 'frames').relative_to(root)))
    for field in ('hsgg_archive', 'hsgg_archive_root', 'frame_archive', 'frame_archive_root', 'frame_member_template'):
        video.pop(field)
    return video, new_dir


def test_directory_input_reuses_relocated_files_without_tar_or_writes(nextqa_tmp_dir: Path, monkeypatch) -> None:
    video, input_dir = _materialized_video(nextqa_tmp_dir)
    before = {path.relative_to(input_dir): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in input_dir.rglob('*') if path.is_file()}
    for archive in nextqa_tmp_dir.glob('*.tar.gz'):
        archive.unlink()
    def deny_tar(*args, **kwargs):
        raise AssertionError('directory source must not open archives')
    monkeypatch.setattr('tarfile.open', deny_tar)

    item = load_normalized_video_input(video, repository_root=nextqa_tmp_dir, verify_frame_hashes=True)

    assert item.input_dir == input_dir
    assert item.materialized_frame_root == input_dir / 'frames'
    assert (item.sampled_frame_count, item.completed_frame_count, item.skipped_frame_count) == (2, 1, 1)
    assert item.end_timestamp_seconds == 1.001
    after = {path.relative_to(input_dir): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in input_dir.rglob('*') if path.is_file()}
    assert before == after


def _portable_item(root: Path) -> tuple[dict, Path]:
    video, prior_dir = _materialized_video(root)
    item_dir = root / 'items' / video['official_video_id']
    item_dir.parent.mkdir()
    prior_dir.rename(item_dir)
    objects_path, trace_path = item_dir / 'objects.json', item_dir / 'source_trace.json'
    objects = json.loads(objects_path.read_text(encoding='utf-8'))
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    for name, payload in objects.items():
        payload['image_path'] = f'frames/{name}.png'
    for frame in trace['sampled_frames']:
        frame['materialized_image_path'] = f"frames/{frame['frame_name']}.png"
    objects_path.write_text(json.dumps(objects), encoding='utf-8')
    trace_path.write_text(json.dumps(trace), encoding='utf-8')
    video.update(normalized_input_dir=item_dir.relative_to(root).as_posix(), frames_dir=(item_dir / 'frames').relative_to(root).as_posix(), image_path_base='input_directory')
    return video, item_dir


def test_official_id_items_reuse_portable_paths_and_preserve_source_bytes(nextqa_tmp_dir: Path, monkeypatch) -> None:
    video, input_dir = _portable_item(nextqa_tmp_dir)
    assert input_dir.name == video['official_video_id'] == '100'
    before = {path.relative_to(input_dir): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in input_dir.rglob('*') if path.is_file()}
    for archive in nextqa_tmp_dir.glob('*.tar.gz'):
        archive.unlink()
    def deny_tar(*args, **kwargs):
        raise AssertionError('portable items must not open archives')
    monkeypatch.setattr('tarfile.open', deny_tar)

    item = load_normalized_video_input(video, repository_root=nextqa_tmp_dir, verify_frame_hashes=True)

    assert item.input_dir == input_dir
    assert item.materialized_frame_root == input_dir / 'frames'
    assert (item.sampled_frame_count, item.completed_frame_count, item.skipped_frame_count) == (2, 1, 1)
    assert item.end_timestamp_seconds == 1.001
    after = {path.relative_to(input_dir): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest()) for path in input_dir.rglob('*') if path.is_file()}
    assert before == after


@pytest.mark.parametrize('target', ['objects', 'source_trace'])
def test_portable_item_rejects_stale_paths_even_when_png_exists(nextqa_tmp_dir: Path, target: str) -> None:
    video, input_dir = _portable_item(nextqa_tmp_dir)
    path = input_dir / f'{target}.json'
    value = json.loads(path.read_text(encoding='utf-8'))
    if target == 'objects':
        value['frame_000000_0000.000s']['image_path'] = '/historical/old.png'
    else:
        value['sampled_frames'][0]['materialized_image_path'] = 'frames/other.png'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='portable image_path mismatch'):
        load_normalized_video_input(video, repository_root=nextqa_tmp_dir)


@pytest.mark.parametrize('fault, expected', [
    ('missing_table_frame', 'table frame keys'),
    ('timeline_status', 'HSGG status'),
    ('timeline_schema', 'timeline schema'),
    ('source_provenance', 'invalid aggregate_json_read'),
    ('manifest_count', 'manifest count mismatch'),
    ('missing_frame', 'frame_000030'),
    ('frame_hash', 'SHA256 mismatch'),
])
def test_directory_input_rejects_incomplete_or_inconsistent_sources(nextqa_tmp_dir: Path, fault: str, expected: str) -> None:
    video, input_dir = _materialized_video(nextqa_tmp_dir)
    if fault == 'missing_table_frame':
        path = input_dir / 'attributes.json'
        payload = json.loads(path.read_text())
        payload.pop('frame_000030_0001.001s')
    elif fault in {'timeline_status', 'timeline_schema'}:
        path = input_dir / 'timeline.json'
        payload = json.loads(path.read_text())
        if fault == 'timeline_status':
            payload['frames'][0]['has_scene_graph'] = False
        else:
            payload['schema_version'] = 'unknown'
    elif fault in {'source_provenance', 'frame_hash'}:
        path = input_dir / 'source_trace.json'
        payload = json.loads(path.read_text())
        if fault == 'source_provenance':
            payload['aggregate_json_read'] = True
        else:
            payload['sampled_frames'][0]['frame_member_sha256'] = '0' * 64
    elif fault == 'manifest_count':
        video['hsgg_frame_count'] += 1
        path, payload = None, None
    else:
        (input_dir / 'frames' / 'frame_000030_0001.001s.png').unlink()
        path, payload = None, None
    if path is not None:
        path.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises((ValueError, FileNotFoundError), match=expected):
        load_normalized_video_input(video, repository_root=nextqa_tmp_dir, verify_frame_hashes=True)


def test_input_batches_keep_different_archive_imports_separate() -> None:
    from scripts.datasets.nextqa.build.build_graph_full import input_batches
    rows = [
        {'official_video_id': '1', 'hsgg_archive': 'base.tar.gz'},
        {'official_video_id': '2', 'hsgg_archive': 'repair.tar.gz'},
        {'official_video_id': '3', 'hsgg_archive': 'base.tar.gz'},
        {'official_video_id': '4', 'input_storage': 'normalized_directory'},
    ]
    batches = list(input_batches(rows, 285))
    assert [[row['official_video_id'] for row in batch] for _, batch in batches] == [['1', '3'], ['2'], ['4']]
    assert [storage for storage, _ in batches] == ['archive', 'archive', 'normalized_directory']


def test_full_build_directory_route_does_not_normalize_or_copy_inputs(nextqa_tmp_dir: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace
    from scripts.datasets.nextqa.build import build_graph_full as builder

    video, input_dir = _portable_item(nextqa_tmp_dir)
    manifest = {'videos': [video]}
    manifest_path, config_path = nextqa_tmp_dir / 'manifest.json', nextqa_tmp_dir / 'config.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    config_path.write_text('{}', encoding='utf-8')
    run_root = nextqa_tmp_dir / 'run'
    run_root.mkdir()
    for archive in nextqa_tmp_dir.glob('*.tar.gz'):
        archive.unlink()
    before = {path.relative_to(input_dir): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_dir.rglob('*') if path.is_file()}
    args = SimpleNamespace(run_id='directory-route-test', manifest=manifest_path, config=config_path, device='cpu', normalization_batch_size=285, resume=False)
    monkeypatch.setattr(builder, 'parse_args', lambda: args)
    monkeypatch.setattr(builder, 'REPO_ROOT', nextqa_tmp_dir)
    monkeypatch.setattr(builder, 'PROJECT_PYTHON', Path(sys.executable).resolve())
    monkeypatch.setattr(builder, 'load_full_manifest', lambda path: manifest)
    config = SimpleNamespace(profile_name='nextqa_cg6_human_compat_v2', model_dump=lambda **kwargs: {})
    monkeypatch.setattr(builder, 'load_nextqa_graph_config', lambda *args, **kwargs: config)
    monkeypatch.setattr(builder, 'prepare_full_graph_run_root', lambda *args, **kwargs: run_root)
    monkeypatch.setattr(builder, 'collect_imported_repo_sources', lambda **kwargs: [])
    monkeypatch.setattr(builder, 'VisionFeatureExtractor', lambda **kwargs: object())
    monkeypatch.setattr(builder, 'EmbeddingCache', lambda path: SimpleNamespace(close=lambda: None))
    def deny_archive(*args, **kwargs):
        raise AssertionError('default directory build must not read or normalize archives')
    monkeypatch.setattr('tarfile.open', deny_archive)
    monkeypatch.setattr(builder, 'normalize_hsgg_archive', deny_archive)
    received = []
    def fake_build(item, **kwargs):
        received.append(item)
        assert item.input_dir == input_dir
        assert item.materialized_frame_root == input_dir / 'frames'
        return SimpleNamespace(official_video_id=item.official_video_id, local_sequential_id=item.local_sequential_id, temporal_graph_path=run_root / 'temporal.json', canonical_graph_path=run_root / 'canonical.json', canonical_report_path=run_root / 'canonical_report.json', audit_path=run_root / 'audit.json')
    monkeypatch.setattr(builder, 'build_video_graph', fake_build)
    def fake_audit(**kwargs):
        assert len(kwargs['built']) == 1 and not kwargs['errors']
        assert kwargs['expected_counts']['sampled_frames'] == 2
        return {'status': 'passed'}
    monkeypatch.setattr(builder, 'build_graph_audit', fake_audit)
    monkeypatch.setattr(builder, 'persist_graph_reports', lambda *args, **kwargs: None)

    assert builder.main() == 0
    assert len(received) == 1
    assert not (run_root / 'normalized_inputs').exists()
    assert json.loads((run_root / 'run_config.json').read_text())['contract']['input_storage_modes'] == ['normalized_directory']
    after = {path.relative_to(input_dir): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_dir.rglob('*') if path.is_file()}
    assert before == after
