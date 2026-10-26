from pathlib import Path
import hashlib
import json

import pytest

from scripts.datasets.materialize_nextqa_index import materialize


def setup_bundle(root: Path):
    source = root / '.scratch/nextqa/visual_indexes/old'
    source.mkdir(parents=True)
    (source / 'base.json').write_bytes(b'{}')
    (source / 'frame.faiss').write_bytes(b'unchanged-index-bytes')
    (source / 'frame.json').write_bytes(b'[{"candidate_id":"one"}]')
    manifest = {'schema_version': 'nextqa_visual_index_manifest_v2',
                'base_visual_index_manifest_path': str(source / 'base.json'),
                'base_visual_index_manifest_sha256': hashlib.sha256(b'{}').hexdigest(),
                'videos': [{'video_id': '100', 'shards': {'frame': {
                    'count': 1, 'index_path': str(source / 'frame.faiss'),
                    'metadata_path': str(source / 'frame.json')}}}]}
    path = source / 'manifest.json'
    path.write_text(json.dumps(manifest))
    target = root / 'datasets/nextqa/indexes/selected/payload'
    return path, target, root / '.scratch/nextqa/visual_indexes'


def test_copy_is_idempotent_and_preserves_originals(tmp_path):
    source, target, allowed = setup_bundle(tmp_path)
    before = source.read_bytes()
    report = materialize(source, target, allowed_source=allowed, root=tmp_path)
    assert report['files'] == 2
    output = json.loads((target / 'visual_index_manifest.json').read_text())
    shard = output['videos'][0]['shards']['frame']
    assert Path(shard['index_path']).read_bytes() == b'unchanged-index-bytes'
    assert source.read_bytes() == before
    assert materialize(source, target, allowed_source=allowed, root=tmp_path) == report


def test_different_destination_is_not_overwritten(tmp_path):
    source, target, allowed = setup_bundle(tmp_path)
    conflict = target / 'videos/100/frame.faiss'
    conflict.parent.mkdir(parents=True)
    conflict.write_bytes(b'unrelated')
    with pytest.raises(RuntimeError, match='hash mismatch'):
        materialize(source, target, allowed_source=allowed, root=tmp_path)
    assert conflict.read_bytes() == b'unrelated'
    assert not (target / 'visual_index_manifest.json').exists()


def test_source_escape_is_rejected_before_copy(tmp_path):
    source, target, allowed = setup_bundle(tmp_path)
    data = json.loads(source.read_text())
    outside = tmp_path / 'elsewhere.faiss'
    outside.write_bytes(b'unrelated')
    data['videos'][0]['shards']['frame']['index_path'] = str(outside)
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='outside source tree'):
        materialize(source, target, allowed_source=allowed, root=tmp_path)
    assert not target.exists()
