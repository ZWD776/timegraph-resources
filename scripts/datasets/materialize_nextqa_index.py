"""Copy selected visual shards byte-for-byte into one new index bundle.

Never rebuilds embeddings, deletes sources, or overwrites different destination
bytes. The published manifest is written only after all shards verify.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
SAFE = re.compile(r'^[A-Za-z0-9_-]+$')


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f'refusing to replace different bytes: {path}')
        return
    with path.open('xb') as stream:
        stream.write(content)


def encode(payload) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def materialize(source: Path, target: Path, *, allowed_source: Path, root: Path) -> dict:
    root, source, target = root.resolve(), source.resolve(), target.resolve()
    allowed_source = allowed_source.resolve()
    if not target.is_relative_to(root / 'datasets/nextqa/indexes'):
        raise ValueError('target must be inside the parallel NExT-QA indexes tree')
    if target.is_relative_to(allowed_source) or allowed_source.is_relative_to(target):
        raise ValueError('source and destination trees overlap')
    source_bytes = source.read_bytes()
    original = json.loads(source_bytes)
    if original.get('schema_version') != 'nextqa_visual_index_manifest_v2':
        raise ValueError('expected the selected v2 visual index manifest')
    output = copy.deepcopy(original)
    jobs = []
    destinations = set()
    for video in output['videos']:
        video_id = str(video['video_id'])
        if not SAFE.fullmatch(video_id):
            raise ValueError('unsafe video identifier')
        for kind, shard in video['shards'].items():
            if not SAFE.fullmatch(kind):
                raise ValueError('unsafe shard kind')
            for field, suffix in [('index_path', '.faiss'), ('metadata_path', '.metadata.json')]:
                value = shard.get(field)
                if value is None:
                    if field != 'index_path' or int(shard.get('count', 0)) != 0:
                        raise ValueError('null path permitted only for empty index shards')
                    continue
                src = Path(value).resolve()
                if not src.is_relative_to(allowed_source) or not src.is_file():
                    raise ValueError(f'input shard outside source tree or missing: {src}')
                dst = target / 'videos' / video_id / (kind + suffix)
                if dst in destinations:
                    raise ValueError(f'duplicate target: {dst}')
                destinations.add(dst)
                jobs.append((src, dst))
                shard[field] = str(dst)
    if len({str(v['video_id']) for v in output['videos']}) != len(output['videos']):
        raise ValueError('duplicate video records')
    # Freeze provenance manifests inside the bundle so no shard-manifest dependency
    # remains on the v1 index directory. Graph/source references remain external.
    provenance = target / 'provenance'
    immutable_bytes(provenance / 'source_v2_manifest.json', source_bytes)
    base = Path(original['base_visual_index_manifest_path']).resolve()
    if not base.is_relative_to(allowed_source):
        raise ValueError('base manifest outside source tree')
    base_bytes = base.read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != original['base_visual_index_manifest_sha256']:
        raise ValueError('base manifest hash no longer matches v2 provenance')
    immutable_bytes(provenance / 'source_v1_manifest.json', base_bytes)
    output['base_visual_index_manifest_path'] = str(provenance / 'source_v1_manifest.json')
    ledger = []
    for i, (src, dst) in enumerate(jobs, 1):
        before = src.stat()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            with src.open('rb') as inp, dst.open('xb') as out:
                shutil.copyfileobj(inp, out, 1024 * 1024)
        src_hash, dst_hash = digest(src), digest(dst)
        after = src.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'source changed during copy: {src}')
        if src_hash != dst_hash:
            raise RuntimeError(f'copy hash mismatch; destination retained for inspection: {dst}')
        ledger.append({'source': src.relative_to(root).as_posix(),
                       'target': dst.relative_to(root).as_posix(),
                       'bytes': after.st_size, 'sha256': src_hash})
        if i % 500 == 0 or i == len(jobs):
            print(f'verified {i}/{len(jobs)}', flush=True)
    if source.read_bytes() != source_bytes:
        raise RuntimeError('source manifest changed during copy')
    immutable_bytes(target / 'file_hashes.json', encode(ledger))
    immutable_bytes(target / 'visual_index_manifest.json', encode(output))
    return {'contract': 'nextqa_index_materialization_v1', 'status': 'copied_and_hash_verified',
            'source_manifest': source.relative_to(root).as_posix(),
            'source_manifest_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'target_manifest': (target / 'visual_index_manifest.json').relative_to(root).as_posix(),
            'target_manifest_sha256': digest(target / 'visual_index_manifest.json'),
            'file_hashes': (target / 'file_hashes.json').relative_to(root).as_posix(),
            'file_hashes_sha256': digest(target / 'file_hashes.json'),
            'files': len(jobs), 'bytes': sum(r['bytes'] for r in ledger),
            'videos': len(output['videos']), 'source_deleted': False,
            'graph_and_image_dependencies': 'external_unchanged'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.source, args.target,
                         allowed_source=ROOT / '.scratch/nextqa/visual_indexes', root=ROOT)
    print(json.dumps(result, indent=2), flush=True)
    immutable_bytes(args.target.resolve() / 'materialization_summary.json', encode(result))


if __name__ == '__main__':
    main()
