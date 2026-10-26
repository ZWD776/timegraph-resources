"""Exercise relocated resources with old dataset roots and network denied."""
from __future__ import annotations

import argparse
import contextlib
from collections import Counter
from dataclasses import asdict, is_dataclass
import gc
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
_ACTIVE_GUARD = None


@contextlib.contextmanager
def historical_evidence_access():
    """Allow archive only while opening an explicit immutable parity reference."""
    _ACTIVE_GUARD.historical_evidence_mode = True
    try:
        yield
    finally:
        _ACTIVE_GUARD.historical_evidence_mode = False


class Guard:
    def __init__(self):
        self.denied = []
        self.network = 0
        self.reads = Counter()
        self.historical_evidence_mode = False
        self.historical_evidence_reads = Counter()

    def check(self, value, event):
        if isinstance(value, int) or value is None:
            return
        raw = os.fsdecode(value)
        if raw.startswith('file:'):
            raw = raw[5:].split('?', 1)[0]
        path = Path(raw).resolve()
        if not path.is_relative_to(ROOT):
            return
        relative = path.relative_to(ROOT).as_posix()
        normalized = relative.casefold()
        if normalized.startswith('archive/'):
            if self.historical_evidence_mode:
                self.historical_evidence_reads[relative] += 1
                return
            self.denied.append({'event': event, 'path': relative})
            raise PermissionError('active runtime may not read archive: ' + relative)
        if normalized.startswith(('.scratch/navqa', '.scratch/openeqa', 'data/openeqa/',
                                  'data/workdir/openeqa/', 'data/cache/', 'graphiti-main/')):
            self.denied.append({'event': event, 'path': relative})
            raise PermissionError('retired data or optional backend read denied: ' + relative)
        self.reads[relative] += 1

    def hook(self, event, args):
        if event in ('open', 'sqlite3.connect'):
            self.check(args[0], event)
        elif event == 'socket.connect':
            self.network += 1
            raise PermissionError('network disabled for migration verification')


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()
                if not any(term in key.lower() for term in ('path', 'source_dir'))
                and key not in ('temporal_graph_json', 'canonical_graph_json', 'frame_time_index_json')}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, str) and value.startswith(('{', '[')):
        try:
            return normalize(json.loads(value))
        except json.JSONDecodeError:
            pass
    return value


def navqa_graph_probe():
    from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
    from vlkg.core.graph_retrieval import canonical_hybrid_retrieval as retrieval
    from vlkg.core.memory.types import MemoryQuery
    base = ROOT / 'datasets/navqa/graphs/navqa-cg6-time-fact-core-v1/payload'
    cache = ROOT / 'datasets/navqa/retrieval/navqa-cg6-retrieval-v1/payload/seq4_cam0_every10'
    cases = [('original', 'projection_cache.json.gz'), ('runtime', 'runtime_projection_cache.json.gz')]
    signatures, output, counts = {}, {}, {}
    for part, cache_name in cases:
        path = base / part / 'time_fact_core/seq4/canonical_temporal_memory_graph.json'
        graph = CanonicalTemporalMemoryGraph.model_validate_json(path.read_bytes())
        fingerprint = retrieval._projection_cache_fingerprint(graph, temporal_document=None,
                                                               include_base_delta_projection=False)
        session = retrieval.CanonicalHybridRetrievalSession(graph, projection_cache_path=cache / cache_name,
                                                             domain_profile='generic')
        results = []
        for text in ('Did a person interact with a car?', 'When was a building visible?'):
            result = session.retrieve(MemoryQuery(query_text=text, limit=8), mode='temporal', limit=8)
            results.append(normalize(asdict(result) if is_dataclass(result) else result.model_dump(mode='json')))
        signatures[part], output[part] = fingerprint, results
        counts[part] = {'nodes': graph.node_count, 'edges': graph.edge_count}
        retrieval._PROJECTION_CACHE.clear()
        del session, graph
        gc.collect()
    if signatures['original'] != signatures['runtime'] or output['original'] != output['runtime']:
        raise AssertionError('NAVQA graph signature or retrieval behavior changed')
    return {'seq_id': '4', 'graph_counts': counts, 'projection_fingerprint_equal': True,
            'two_query_outputs_equal_except_path_fields': True, 'calls_model': False}


def navqa_visual_probe():
    import numpy as np
    from PIL import Image
    from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import VisualFaissGrounder
    current = ROOT / 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime'
    original = ROOT / 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/visual_faiss'
    new = VisualFaissGrounder(index_dir=current)
    from vlkg.core.assets.asset_provenance import relocated_provenance
    with historical_evidence_access():
        original_metadata = relocated_provenance(original / 'object_siglip_flat_ip.metadata.sqlite', ROOT)
        old = sqlite3.connect('file:' + str(original_metadata) + '?mode=ro', uri=True)
    old.row_factory = sqlite3.Row
    try:
        probes = []
        for seq in ('0', '16', '22'):
            first = new.connection.execute('SELECT faiss_id,image_path FROM object_metadata WHERE seq_id=? LIMIT 1', (seq,)).fetchone()
            vector = np.asarray(new.index.reconstruct(first['faiss_id']), dtype=np.float32)
            hits = new._search_by_vector(vector=vector, seq_id=seq, top_k=8, search_k=256,
                                        same_sequence_only=True, min_score=0.0)
            for hit in hits:
                # Verify original object identity and nonpath metadata directly.
                old_row = old.execute('SELECT * FROM object_metadata WHERE faiss_id=?', (hit['faiss_id'],)).fetchone()
                new_row = new.connection.execute('SELECT * FROM object_metadata WHERE faiss_id=?', (hit['faiss_id'],)).fetchone()
                if normalize(dict(old_row)) != normalize(dict(new_row)):
                    raise AssertionError('NAVQA visual object metadata changed')
            with Image.open(first['image_path']) as image:
                image.load()
            probes.append({'seq_id': seq, 'hits': len(hits), 'image_read': True})
        evidence = ROOT / 'datasets/navqa/indexes/navqa-cg6-retrieval-v1/payload/runtime/evidence_layer.sqlite'
        with sqlite3.connect('file:' + str(evidence) + '?mode=ro', uri=True) as conn:
            refs = conn.execute('SELECT cg_id,count(*) FROM cg_graph_refs GROUP BY cg_id').fetchall()
            if refs != [('CG6-clean', 7)]:
                raise AssertionError('evidence index still contains non-mainline graphs')
            for value, in conn.execute('SELECT resource_json FROM sequences'):
                if any(g['cg_id'] != 'CG6-clean' for g in json.loads(value)['cg_graphs']):
                    raise AssertionError('resource metadata still contains non-mainline graphs')
        return {'probes': probes, 'graph_refs': refs, 'calls_model': False}
    finally:
        old.close(); new.close()


def openeqa_probe():
    import numpy as np
    from PIL import Image
    from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
    from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex
    root = ROOT / 'datasets/openeqa'
    probes = []
    for shard in ('hm3d', 'scannet'):
        graph_base = root / 'graphs/openeqa-time-fact-core-v1/payload'
        manifest = json.loads((graph_base / 'runtime' / shard / 'openeqa_hm3d_graph_build_manifest.json').read_text(encoding='utf-8'))
        index_base = root / 'indexes/openeqa-v5-full-visual-indexes/payload'
        search = ObjectVisualSearchIndex(index_dir=index_base / 'runtime' / shard)
        from vlkg.core.assets.asset_provenance import relocated_provenance
        with historical_evidence_access():
            original_metadata = relocated_provenance(index_base / 'original' / shard / 'openeqa_object_siglip_flat_ip.metadata.sqlite', ROOT)
            before = sqlite3.connect('file:' + str(original_metadata) + '?mode=ro', uri=True)
        before.row_factory = sqlite3.Row
        try:
            for position in (0, len(manifest['episodes']) // 2, len(manifest['episodes']) - 1):
                episode = manifest['episodes'][position]
                graph_path = Path(episode['canonical_graph_json'])
                graph = CanonicalTemporalMemoryGraph.model_validate_json(graph_path.read_bytes())
                row = search.connection.execute('SELECT faiss_id,image_path FROM object_metadata WHERE episode_history=? LIMIT 1',
                                                 (episode['episode_history'],)).fetchone()
                if row is None:
                    raise AssertionError('episode has no indexed objects')
                vector = np.asarray(search.index.reconstruct(row['faiss_id']), dtype=np.float32)
                hits = search.search_vector(episode_history=episode['episode_history'], vector=vector, top_k=8, min_score=0)
                for hit in hits:
                    old_row = before.execute('SELECT * FROM object_metadata WHERE faiss_id=?', (hit['faiss_id'],)).fetchone()
                    new_row = search.connection.execute('SELECT * FROM object_metadata WHERE faiss_id=?', (hit['faiss_id'],)).fetchone()
                    if normalize(dict(old_row)) != normalize(dict(new_row)):
                        raise AssertionError('OpenEQA visual identity/score input changed')
                with Image.open(row['image_path']) as image:
                    image.load()
                probes.append({'shard': shard, 'episode': episode['episode_history'],
                               'nodes': graph.node_count, 'edges': graph.edge_count, 'hits': len(hits), 'image_read': True})
                del graph
                gc.collect()
        finally:
            before.close(); search.close()
    return {'probes': probes, 'calls_model': False}


def main():
    global _ACTIVE_GUARD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=('navqa_graph', 'navqa_visual', 'openeqa_resources'))
    args = parser.parse_args()
    guard = Guard()
    _ACTIVE_GUARD = guard
    sys.addaudithook(guard.hook)
    import faiss
    original = faiss.read_index
    def checked(path, *positional, **kwargs):
        guard.check(path, 'faiss.read_index')
        return original(path, *positional, **kwargs)
    faiss.read_index = checked
    error = None
    detail = {}
    try:
        detail = {'navqa_graph': navqa_graph_probe, 'navqa_visual': navqa_visual_probe,
                  'openeqa_resources': openeqa_probe}[args.case]()
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)}
    result = {'case': args.case, 'status': 'passed' if error is None and not guard.denied and not guard.network else 'failed',
              'detail': detail, 'error': error, 'denied_accesses': guard.denied, 'network_attempts': guard.network,
              'historical_evidence_reads': dict(guard.historical_evidence_reads),
              'limits': ['Only executed branches are covered.', 'No complete OS-level tracing of native libraries.',
                         'Original sealed or dataset-owned views are read solely as parity evidence.']}
    out = ROOT / 'registry/audits/relocated_mainlines' / (args.case + '.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
