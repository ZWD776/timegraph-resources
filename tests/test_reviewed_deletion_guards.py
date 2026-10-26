"""Safety checks use disposable toy files only, never project experiment assets."""
import hashlib
from pathlib import Path

import pytest

from scripts.datasets.execute_reviewed_deletion import checked_candidate, remove_verified_file, prior_journal
from scripts.datasets.check_predeletion_readiness import NoArchiveGuard, ROOT


def row_for(path, root, content):
    return {'path': path.relative_to(root).as_posix(), 'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest(), 'decision': 'delete_candidate', 'delete_now': False}


def test_removal_only_touches_the_verified_toy_candidate(tmp_path):
    folder = tmp_path / 'archive/legacy_20260928/toy'
    folder.mkdir(parents=True)
    candidate, retained = folder / 'old.bin', folder / 'retained.txt'
    candidate.write_bytes(b'old toy data')
    retained.write_bytes(b'keep this')
    journal = []
    remove_verified_file(row_for(candidate, tmp_path, b'old toy data'), journal.append, tmp_path)
    assert not candidate.exists()
    assert retained.read_bytes() == b'keep this'
    assert [row['status'] for row in journal] == ['hash_verified_before_removal', 'removed']


def test_same_size_changed_content_blocks_removal(tmp_path):
    p = tmp_path / 'archive/legacy_20260928/toy/old.bin'
    p.parent.mkdir(parents=True)
    p.write_bytes(b'BBBB')
    journal = []
    with pytest.raises(ValueError, match='SHA256'):
        remove_verified_file(row_for(p, tmp_path, b'AAAA'), journal.append, tmp_path)
    assert p.read_bytes() == b'BBBB'
    assert journal == []


@pytest.mark.parametrize('value', ['datasets/navqa/current.bin', 'archive/legacy_20260928/../escape.bin',
                                  'D:/outside.bin', 'archive/legacy_20260928/toy\\old.bin'])
def test_paths_outside_reviewed_boundary_are_rejected(tmp_path, value):
    row = {'path': value, 'bytes': 4, 'sha256': '0' * 64, 'decision': 'delete_candidate', 'delete_now': False}
    with pytest.raises(ValueError, match='archive'):
        checked_candidate(row, tmp_path)


def test_retained_row_cannot_be_removed(tmp_path):
    row = {'path': 'archive/legacy_20260928/report.txt', 'decision': 'retain_sealed_information', 'delete_now': False}
    with pytest.raises(ValueError, match='retained'):
        checked_candidate(row, tmp_path)


def test_archive_guard_also_rejects_windows_extended_paths():
    guard = NoArchiveGuard()
    path = ROOT / 'archive/legacy_20260928/nonexistent.bin'
    with pytest.raises(PermissionError):
        guard.check(path)
    if __import__('os').name == 'nt':
        with pytest.raises(PermissionError):
            guard.check('\\\\?\\' + str(path))
    guard.check(ROOT / 'datasets/navqa/source/nonexistent.bin')


def test_resume_refuses_an_unverified_removal_record(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_text('{"path":"old.bin","status":"removed","bytes":4}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        prior_journal(path)


def test_resume_stops_on_a_truncated_journal(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_text('{"path":"old.bin"', encoding='utf-8')
    with pytest.raises(ValueError, match='truncated'):
        prior_journal(path)
