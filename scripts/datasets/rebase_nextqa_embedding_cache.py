"""Relocate cache keys in a new SQLite snapshot without re-encoding vectors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.datasets.rebase_nextqa_runtime import Relocator  # noqa: E402


def content_hash(connection, mapper=None):
    """Hash stable logical rows; vector bytes and dimension must remain identical."""
    digest = hashlib.sha256()
    count = 0
    # Ordering by unchanged trailing filename is equivalent after common-root relocation.
    for path, bbox, embedding, dim in connection.execute(
        'SELECT image_path,bbox_key,embedding,dim FROM embedding_cache ORDER BY image_path,bbox_key'
    ):
        if mapper:
            path = str(Path(mapper.path(path)))
        for value in (path.encode(), bbox.encode(), embedding, str(dim).encode()):
            digest.update(len(value).to_bytes(8, 'big'))
            digest.update(value)
        count += 1
    return count, digest.hexdigest()


def migrate(source: Path, target: Path, mapper: Relocator):
    if target.exists():
        raise FileExistsError(f'cache destination already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f'file:{source.resolve().as_posix()}?mode=ro', uri=True)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
        paths = [row[0] for row in src.execute('SELECT DISTINCT image_path FROM embedding_cache')]
        mapping = [(str(Path(mapper.path(p))), p) for p in paths]
        if len({p[0] for p in mapping}) != len(mapping):
            raise ValueError('cache key relocation is not one-to-one')
        if any(a == b for a, b in mapping):
            raise ValueError('cache contains unmapped legacy image paths')
        if any(not Path(a).is_file() for a, _ in mapping):
            raise FileNotFoundError('relocated cache references missing image')
        with dst:
            for i,(new,old) in enumerate(mapping,1):
                dst.execute('UPDATE embedding_cache SET image_path=? WHERE image_path=?', (new,old))
                if i % 5000 == 0:
                    print(f'cache paths updated {i}/{len(mapping)}', flush=True)
        before = content_hash(src, mapper)
        after = content_hash(dst)
        if before != after:
            raise AssertionError('cache vectors/keys/dimensions differ beyond path relocation')
        if dst.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise AssertionError('relocated cache failed SQLite quick_check')
        return {'status':'passed','rows':before[0],'unique_image_paths':len(paths),
                'logical_row_sha256':after[1],'embedding_bytes_and_dimensions_unchanged':True,
                'sqlite_quick_check':'ok','source_modified':False}
    finally:
        dst.close()
        src.close()


if __name__=='__main__':
    base = ROOT / 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload'
    source = base / 'nextqa_cg6_human_v2_full570_20260812a/cache/siglip_object_crop.sqlite'
    target = base / 'runtime_paths_v2/cache/siglip_object_crop.sqlite'
    mapper = Relocator({'.scratch/nextqa/graphs/nextqa_cg6_human_v2_full570_20260812a':
                        (base / 'nextqa_cg6_human_v2_full570_20260812a').relative_to(ROOT).as_posix()}, ROOT)
    result = migrate(source, target, mapper)
    result.update(source=source.relative_to(ROOT).as_posix(), target=target.relative_to(ROOT).as_posix())
    with target.open('rb') as stream:
        result['target_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    (ROOT / 'registry/migrations/nextqa_cache_path_rebase.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2),flush=True)
