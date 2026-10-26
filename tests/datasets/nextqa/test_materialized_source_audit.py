"""Directory source verification must work after the source archives are retired."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

from scripts.datasets.nextqa.audit.verify_materialized_source import verify_materialized_source
from vlkg.datasets.nextqa.graph.hsgg_input_adapter import normalize_hsgg_archive

from .conftest import make_archive_and_video

COUNTS = {
    'videos': 1, 'sampled_frames': 2, 'completed_hsgg_frames': 1,
    'skipped_frames': 1, 'materialized_images': 2,
}


def _directory_source(root: Path) -> tuple[Path, Path]:
    _, video = make_archive_and_video(
        root,
        frame_statuses=[
            ('frame_000000_0000.000s', 'completed'),
            ('frame_000030_0001.001s', 'skipped_no_scene_graph'),
        ],
    )
    [normalized] = normalize_hsgg_archive([video], output_root=root / 'normalized')
    Path(video['hsgg_archive']).unlink()
    Path(video['frame_archive']).unlink()
    for name in ('hsgg_archive', 'hsgg_archive_root', 'frame_archive',
                 'frame_archive_root', 'frame_member_template'):
        video.pop(name)
    video.update({
        'input_storage': 'normalized_directory',
        'normalized_input_dir': str(normalized.input_dir),
        'frames_dir': str(normalized.materialized_frame_root),
    })
    manifest = root / 'directory_manifest.json'
    manifest.write_text(json.dumps({
        'dataset': 'NExT-QA', 'split': 'val', 'video_count': 1,
        'input_storage': 'normalized_directory', 'videos': [video],
    }), encoding='utf-8')
    return manifest, normalized.input_dir


def test_audit_verifies_materialized_source_without_archive_reads(nextqa_tmp_dir, monkeypatch):
    manifest, _ = _directory_source(nextqa_tmp_dir)

    def reject_archive_reads(*args, **kwargs):
        raise AssertionError('directory verifier must not open an archive')

    monkeypatch.setattr(tarfile, 'open', reject_archive_reads)
    report = verify_materialized_source(
        manifest, expected_counts=COUNTS, verify_image_sha256=True,
    )
    assert report['status'] == 'passed', report['issues']
    assert report['counts'] == COUNTS
    assert len(report['videos'][0]['json_fingerprints']) == 6
    assert report['operations']['archives_opened'] is False


def test_audit_rejects_missing_materialized_image(nextqa_tmp_dir):
    manifest, directory = _directory_source(nextqa_tmp_dir)
    (directory / 'frames' / 'frame_000030_0001.001s.png').unlink()
    report = verify_materialized_source(manifest, expected_counts=COUNTS)
    assert report['status'] == 'failed'
    assert report['videos'][0]['status'] == 'failed'


def test_audit_rejects_same_size_image_corruption(nextqa_tmp_dir):
    manifest, directory = _directory_source(nextqa_tmp_dir)
    image = directory / 'frames' / 'frame_000000_0000.000s.png'
    raw = bytearray(image.read_bytes())
    raw[-1] ^= 1
    image.write_bytes(raw)
    report = verify_materialized_source(
        manifest, expected_counts=COUNTS, verify_image_sha256=True,
    )
    assert report['status'] == 'failed'
    assert report['videos'][0]['status'] == 'failed'


def test_audit_baseline_detects_structurally_valid_attribute_change(nextqa_tmp_dir):
    manifest, directory = _directory_source(nextqa_tmp_dir)
    report = verify_materialized_source(manifest, expected_counts=COUNTS)
    assert report['status'] == 'passed', report['issues']
    baseline = nextqa_tmp_dir / 'baseline.json'
    baseline.write_text(json.dumps(report), encoding='utf-8')
    attributes_path = directory / 'attributes.json'
    attributes = json.loads(attributes_path.read_text(encoding='utf-8'))
    attributes['frame_000000_0000.000s']['0'] = 'Different description'
    attributes_path.write_text(json.dumps(attributes), encoding='utf-8')
    checked = verify_materialized_source(
        manifest, expected_counts=COUNTS, baseline_report=baseline,
    )
    assert checked['status'] == 'failed'
    assert any('fingerprints differ' in issue['error'] for issue in checked['issues'])
