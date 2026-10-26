"""Verify the complete directory-backed NExT-QA source without opening archives."""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath

_BOOTSTRAP_ROOT = next(
    p for p in _BootstrapPath(__file__).resolve().parents
    if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()
)
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

EXPECTED_COUNTS = {
    'videos': 570,
    'sampled_frames': 23381,
    'completed_hsgg_frames': 23044,
    'skipped_frames': 337,
    'materialized_images': 23381,
}
SOURCE_FILES = (
    'objects.json', 'attributes.json', 'relations.json',
    'hierarchy_mapping.json', 'timeline.json', 'source_trace.json',
)


def _fingerprint(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def verify_materialized_source(
    manifest_path: str | Path,
    *,
    repository_root: str | Path = _BOOTSTRAP_ROOT,
    verify_image_sha256: bool = False,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
    baseline_report: str | Path | None = None,
) -> dict[str, Any]:
    """Check materialized content and retain fresh, immutable JSON fingerprints.

    The loader checks tables, timestamps, frame coverage and recorded provenance.
    An optional image check recomputes every PNG hash. Original HSGG member hashes
    remain provenance records; validating them against the original bytes belongs
    to the separate pre-retirement audit, not this archive-free verifier.
    """
    from vlkg.datasets.nextqa.graph.hsgg_input_adapter import load_normalized_video_input

    manifest_path = Path(manifest_path).resolve()
    root = Path(repository_root).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    baseline = {}
    if baseline_report is not None:
        previous = json.loads(Path(baseline_report).read_text(encoding='utf-8-sig'))
        baseline = {row['official_video_id']: row for row in previous['videos']}
    totals = dict.fromkeys(EXPECTED_COUNTS, 0)
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    videos = manifest.get('videos', [])
    if manifest.get('input_storage') != 'normalized_directory':
        issues.append({'error': "manifest input_storage must be 'normalized_directory'"})
    if manifest.get('dataset') != 'NExT-QA' or manifest.get('split') != 'val':
        issues.append({'error': 'manifest must describe NExT-QA validation'})
    if not isinstance(videos, list):
        issues.append({'error': 'manifest videos must be a list'})
        videos = []
    if manifest.get('video_count') != len(videos):
        issues.append({'error': 'manifest video_count does not match its rows'})
    try:
        local_ids = [int(video['local_sequential_id']) for video in videos]
        video_ids = [str(video['official_video_id']) for video in videos]
        if sorted(local_ids) != list(range(1, expected_counts['videos'] + 1)):
            issues.append({'error': 'local sequential IDs do not cover the expected unique range'})
        if len(video_ids) != len(set(video_ids)):
            issues.append({'error': 'duplicate official video IDs'})
    except (KeyError, TypeError, ValueError) as exc:
        issues.append({'error': f'invalid video IDs: {exc}'})
    for video in videos:
        video_id = str(video.get('official_video_id', '<missing>'))
        try:
            loaded = load_normalized_video_input(
                video, repository_root=root, verify_frame_hashes=verify_image_sha256,
            )
            fingerprints = {
                name: _fingerprint(loaded.input_dir / name) for name in SOURCE_FILES
            }
            if baseline_report is not None:
                if video_id not in baseline:
                    raise ValueError('video missing from the baseline report')
                if fingerprints != baseline[video_id]['json_fingerprints']:
                    raise ValueError('materialized JSON fingerprints differ from the baseline')
            totals['videos'] += 1
            totals['sampled_frames'] += loaded.sampled_frame_count
            totals['completed_hsgg_frames'] += loaded.completed_frame_count
            totals['skipped_frames'] += loaded.skipped_frame_count
            totals['materialized_images'] += loaded.materialized_frame_count
            rows.append({
                'official_video_id': video_id,
                'status': 'passed',
                'normalized_input_dir': str(loaded.input_dir),
                'frames_dir': str(loaded.materialized_frame_root),
                'sampled_frames': loaded.sampled_frame_count,
                'completed_hsgg_frames': loaded.completed_frame_count,
                'skipped_frames': loaded.skipped_frame_count,
                'materialized_images': loaded.materialized_frame_count,
                'json_fingerprints': fingerprints,
            })
        except (KeyError, TypeError, ValueError, OSError) as exc:
            issue = {'official_video_id': video_id, 'error': str(exc)}
            issues.append(issue)
            rows.append({'official_video_id': video_id, 'status': 'failed', 'error': str(exc)})
    for key, expected in expected_counts.items():
        if totals[key] != expected:
            issues.append({'error': f'{key} coverage mismatch', 'actual': totals[key], 'expected': expected})
    return {
        'schema_version': 'nextqa_materialized_source_audit_v1',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'passed' if not issues else 'failed',
        'manifest_path': str(manifest_path),
        'manifest_fingerprint': _fingerprint(manifest_path),
        'input_storage': manifest.get('input_storage'),
        'counts': totals,
        'expected_counts': dict(expected_counts),
        'provenance_validation': {
            'hsgg_member_sha256': 'recorded provenance checked; original archive bytes not read',
            'image_sha256': 'recomputed and compared' if verify_image_sha256 else 'recorded hashes and file sizes checked',
            'json_sha256': 'recomputed for all six source JSON files per video',
            'baseline_report': str(Path(baseline_report).resolve()) if baseline_report else None,
        },
        'operations': {
            'archives_opened': False,
            'images_copied': False,
            'source_files_modified': False,
            'model_api_called': False,
        },
        'videos': rows,
        'issues': issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=None)
    parser.add_argument('--output', type=Path, help='Write a new report; existing reports are never overwritten.')
    parser.add_argument('--verify-image-sha256', action='store_true')
    parser.add_argument('--baseline-report', type=Path, help='Require all six JSON fingerprints to match a prior report.')
    args = parser.parse_args()
    if args.manifest is None:
        from vlkg.core.assets.asset_registry import asset_file
        manifest_path = asset_file('nextqa', 'source', 'manifest', _BOOTSTRAP_ROOT)
    else:
        manifest_path = args.manifest
    report = verify_materialized_source(
        manifest_path,
        verify_image_sha256=args.verify_image_sha256,
        baseline_report=args.baseline_report,
    )
    serialized = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output is not None:
        target = args.output.resolve()
        for row in report['videos']:
            source = row.get('normalized_input_dir')
            if source and target.is_relative_to(Path(source)):
                raise ValueError('report output must not be inside the materialized source directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(serialized)
        print(json.dumps({'status': report['status'], 'counts': report['counts'], 'report': str(target)}))
    else:
        print(serialized, end='')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
