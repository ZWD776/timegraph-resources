"""Verify path-only differences and unchanged search/rendered evidence."""
from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.datasets.rebase_nextqa_runtime import Relocator  # noqa: E402

PATH_FIELDS = {'canonical_image_path', 'source_graph_path', 'image_path', 'image_paths',
               'source_dir', 'hsgg_archive', 'frame_archive', 'graph_dir',
               'temporal_graph_path', 'canonical_graph_path', 'standard_json_dir',
               'frame_time_index_json', 'frames_dir', 'input_dir', 'output_root',
               'config_path', 'source_manifest_path', 'run_root', 'path'}


def pathless(value, field=''):
    """Independent semantic comparison: mask explicit path fields, not prose/IDs."""
    if isinstance(value, dict):
        return {k: pathless(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [pathless(v, field) for v in value]
    if isinstance(value, str) and field in PATH_FIELDS:
        return '<asset-path>' if value else value
    return value


def run():
    import numpy as np
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import VisualShardStore
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.rendering import qa_safe_hit, render_hit
    from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
    from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession
    from vlkg.core.graph_retrieval.canonical_record_vector_recall import record_semantic_text
    from vlkg.datasets.nextqa.retrieval.graph_only_retrieval import _record_tokens, render_record_evidence

    report = json.loads((ROOT / 'registry/migrations/nextqa_runtime_path_rebase.json').read_text(encoding='utf-8'))
    if report['status'] != 'passed':
        raise RuntimeError('path rebase has unresolved references')
    unknown = set(report['changed_fields']) - PATH_FIELDS
    if unknown:
        raise RuntimeError(f'unreviewed field changes: {sorted(unknown)}')
    graph = json.loads((ROOT / report['graph_manifest']).read_text(encoding='utf-8'))
    original_graph_root = ROOT / 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/nextqa_cg6_human_v2_full570_20260812a'
    samples = sorted({0, len(graph['videos'])//2, len(graph['videos'])-1})
    graph_checks = []
    projection_records_checked = 0
    for idx in samples:
        row = graph['videos'][idx]
        for field in ('canonical_graph_path', 'temporal_graph_path'):
            target = Path(row[field])
            source = original_graph_root / 'graphs' / target.parent.name / target.name
            before = json.loads(source.read_text(encoding='utf-8'))
            after = json.loads(target.read_text(encoding='utf-8'))
            if pathless(before) != pathless(after):
                raise AssertionError(f'non-path graph value differs: {target}')
            if field == 'canonical_graph_path':
                old_projection = CanonicalHybridRetrievalSession(CanonicalTemporalMemoryGraph.model_validate(before)).projection
                new_projection = CanonicalHybridRetrievalSession(CanonicalTemporalMemoryGraph.model_validate(after)).projection
                if old_projection.records.keys() != new_projection.records.keys():
                    raise AssertionError('canonical retrieval record IDs changed')
                for key, record in old_projection.records.items():
                    other = new_projection.records[key]
                    if record_semantic_text(record) != record_semantic_text(other):
                        raise AssertionError(f'vector embedding input changed: {key}')
                    if _record_tokens(record) != _record_tokens(other):
                        raise AssertionError(f'lexical retrieval tokens changed: {key}')
                    if render_record_evidence(record) != render_record_evidence(other):
                        raise AssertionError(f'graph QA evidence changed: {key}')
                    projection_records_checked += 1
            graph_checks.append(target.relative_to(ROOT).as_posix())
    old = VisualShardStore(ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/visual_index_manifest.json')
    new = VisualShardStore(ROOT / report['index_manifest'])
    if old.video_ids != new.video_ids:
        raise AssertionError('video coverage differs')
    ids = sorted(old.video_ids)
    chosen = sorted({ids[0], ids[len(ids)//2], ids[-1]})
    relocator = Relocator(report['mapping'], ROOT)
    queries = []
    for vid in chosen:
        for kind in old.video_manifests[vid]['shards']:
            index, _ = old._load(vid, kind)
            query = np.ones(old.embedding_dimension, dtype=np.float32)
            if index is not None and index.ntotal:
                query = np.asarray(index.reconstruct(0), dtype=np.float32)
            before = old.search(video_id=vid, kind=kind, query_vector=query, top_k=8)
            after = new.search(video_id=vid, kind=kind, query_vector=query, top_k=8)
            if relocator.transform(deepcopy(before)) != after:
                raise AssertionError(f'search differs beyond rebased paths: {vid}/{kind}')
            if [qa_safe_hit(x) for x in before] != [qa_safe_hit(x) for x in after]:
                raise AssertionError(f'QA evidence differs: {vid}/{kind}')
            for route in ('temporal_next', 'temporal_previous', 'causal_why'):
                if [render_hit(route, 'query', x) for x in before] != [render_hit(route, 'query', x) for x in after]:
                    raise AssertionError(f'rendered evidence differs: {vid}/{kind}/{route}')
            queries.append({'video_id':vid,'kind':kind,'hits':len(after),'equal_except_paths':True,
                            'qa_safe_evidence_equal':True,'rendered_text_equal':True})
        old.release_video(vid)
        new.release_video(vid)
    # All manifest shards must still refer to the identical FAISS files.
    old_index = json.loads((ROOT / 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/visual_index_manifest.json').read_text(encoding='utf-8'))
    new_index = json.loads((ROOT / report['index_manifest']).read_text(encoding='utf-8'))
    for a,b in zip(old_index['videos'],new_index['videos'],strict=True):
        for kind in a['shards']:
            for key in ('index_path','count','dimension','index_bytes'):
                if a['shards'][kind].get(key)!=b['shards'][kind].get(key):
                    raise AssertionError(f'vector contract changed: {a["video_id"]}/{kind}/{key}')
    result={'status':'passed','graph_samples_non_path_equal':graph_checks,
            'projection_records_checked':projection_records_checked,
            'sampled_graph_vector_text_lexical_tokens_evidence_equal':True,
            'index_video_coverage':len(ids),'query_checks':queries,
            'all_vector_file_references_unchanged':True,
            'scope':'Graph sample semantics plus sampled search/QA rendering parity; full JSON roundtrip checks were done during derivation.'}
    (ROOT / 'registry/migrations/nextqa_runtime_path_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'passed','graph_samples':len(graph_checks),'queries':len(queries)},indent=2))


if __name__=='__main__':
    run()
