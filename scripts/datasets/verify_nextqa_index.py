"""Offline old/new VisualShardStore comparison, without models or API calls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def verify(source: Path, target: Path) -> dict:
    import numpy as np
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import VisualShardStore

    old, new = VisualShardStore(source), VisualShardStore(target)
    if old.video_ids != new.video_ids:
        raise AssertionError('video coverage differs')
    ids = sorted(old.video_ids)
    selected = sorted({ids[0], ids[len(ids) // 2], ids[-1]})
    cases = []
    for vid in selected:
        for kind in old.video_manifests[vid]['shards']:
            index, _ = old._load(vid, kind)
            query = np.ones(old.embedding_dimension, dtype=np.float32)
            if index is not None and index.ntotal:
                query = np.asarray(index.reconstruct(0), dtype=np.float32)
            expected = old.search(video_id=vid, kind=kind, query_vector=query, top_k=8)
            actual = new.search(video_id=vid, kind=kind, query_vector=query, top_k=8)
            if expected != actual:
                raise AssertionError(f'search output differs: {vid}/{kind}')
            cases.append({'video_id': vid, 'kind': kind, 'hits': len(actual), 'equal': True})
        old.release_video(vid)
        new.release_video(vid)
    return {'status': 'passed', 'video_coverage': len(ids), 'query_cases': cases,
            'comparison': 'exact existing VisualShardStore result dictionaries including scores and metadata',
            'scope': 'sampled offline search parity; not full QA replay'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.target), indent=2))
