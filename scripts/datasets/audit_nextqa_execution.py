"""Trace offline entrypoints while denying old NExT-QA data and network access.

Run each case in a fresh Python process: audit hooks cannot be uninstalled.
The guard covers Python file opens, sqlite connects and FAISS read_index. It is
diagnostic instrumentation, not a complete OS sandbox or proof of every branch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import json
import os
from pathlib import Path
import runpy
import sys
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]


class AccessGuard:
    def __init__(self, root: Path):
        self.root = os.path.normcase(os.path.abspath(root))
        self.denied_roots = [os.path.join(self.root, '.scratch', 'nextqa'),
                             os.path.join(self.root, 'data', 'next-qa'),
                             os.path.join(self.root, 'archive')]
        for relative in (
            'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/nextqa_cg6_human_v2_full570_20260812a',
            'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/runtime',
            'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/runtime_paths_v2',
            'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/runtime_paths_v2',
        ):
            self.denied_roots.append(os.path.normcase(os.path.join(self.root, *relative.split('/'))))
        self.accesses = Counter()
        self.denied = []
        self.network_attempts = 0

    def check_path(self, value, operation):
        if isinstance(value, int) or value is None:
            return
        value = os.fsdecode(value)
        if value.startswith('file:'):
            value = unquote(value[5:].split('?', 1)[0])
        path = os.path.normcase(os.path.abspath(value))
        if any(path == base or path.startswith(base + os.sep) for base in self.denied_roots):
            self.denied.append({'operation': operation, 'path': path})
            raise PermissionError(f'audit denied legacy NExT-QA asset: {path}')
        if path.startswith(self.root + os.sep):
            self.accesses[os.path.relpath(path, self.root).replace('\\', '/')] += 1

    def hook(self, event, args):
        if event in {'open', 'sqlite3.connect'}:
            self.check_path(args[0], event)
        elif event == 'socket.connect':
            self.network_attempts += 1
            raise PermissionError('network disabled for dependency audit')


def resource_probe():
    import numpy as np
    from PIL import Image
    from vlkg.core.assets.asset_registry import asset_file
    from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import VisualShardStore
    from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import build_object_candidates, open_embedding_cache_read_only
    manifest_path = asset_file('nextqa', 'graphs', 'manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    store = VisualShardStore(asset_file('nextqa', 'indexes', 'manifest'))
    results = []
    cache = open_embedding_cache_read_only(asset_file('nextqa', 'cache', 'sqlite'))
    try:
        for offset in (0, len(manifest['videos'])//2, len(manifest['videos'])-1):
            row = manifest['videos'][offset]
            graph = json.loads(Path(row['temporal_graph_path']).read_text(encoding='utf-8'))
            vid = row['official_video_id']
            observation = graph['observations'][0]
            with Image.open(observation['image_path']) as image:
                image.load()
            metadata, vectors, misses = build_object_candidates(graph, cache)
            if misses:
                raise AssertionError(f'object-cache misses for {vid}: {misses}')
            count = 0
            for kind in store.video_manifests[vid]['shards']:
                index, _ = store._load(vid, kind)
                query = (np.asarray(index.reconstruct(0), dtype=np.float32)
                         if index is not None and index.ntotal else np.ones(store.embedding_dimension,dtype=np.float32))
                count += len(store.search(video_id=vid, kind=kind, query_vector=query, top_k=8))
            results.append({'video_id':vid,'object_candidates':len(metadata),'cache_misses':misses,'search_hits':count})
            store.release_video(vid)
    finally:
        cache.close()
    return {'probes':results,'models_called':False}


def run_case(case: str, run_id: str = 'prearchive_guard_mock_20260930'):
    from scripts.datasets.nextqa_mainline import replay, stage_command
    if case == 'replay':
        return replay()
    if case == 'resources':
        return resource_probe()
    if case == 'graph_mock':
        output = ROOT / 'datasets/nextqa/runs/new_runs/payload/qa' / run_id
        if output.exists():
            raise FileExistsError(f'audit run output already exists: {output}')
        command = stage_command('graph-qa', run_id, answer_mode='mock') + ['--checkpoint-every', '4996']
        saved = sys.argv
        try:
            sys.argv = command[1:]
            try:
                runpy.run_path(command[1], run_name='__main__')
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
        finally:
            sys.argv = saved
        status = json.loads((output/'run_status.json').read_text(encoding='utf-8'))
        return {'output':output.relative_to(ROOT).as_posix(),'run_status':status}
    if case in {'index_defaults','pair_defaults'}:
        import importlib
        script = 'build_visual_indexes' if case=='index_defaults' else 'build_visual_pair_index_v2'
        builder = importlib.import_module('scripts.nextqa.'+script)
        args = builder.build_parser().parse_args(['--run-id','audit-do-not-build'])
        audit_root = ROOT/'datasets/nextqa/runs/new_runs/payload/visual_indexes/audit-do-not-build'
        if case == 'index_defaults':
            inputs = builder.resolve_inputs(args, audit_root)
            path = inputs['graph_root']/'graph_manifest.json'
            if not path.is_file():
                path = inputs['graph_root']/'nextqa_graph_build_manifest.json'
            json.loads(inputs['selection'].read_text(encoding='utf-8'))
            json.loads(inputs['config'].read_text(encoding='utf-8'))
        else:
            path = builder.resolve_base_manifest(args, audit_root)
        json.loads(path.read_text(encoding='utf-8'))
        return {'default_read':str(path)}
    if case == 'causal_inputs':
        from vlkg.core.assets.asset_registry import asset_file
        source=asset_file('nextqa','retrieval','causal_evidence',ROOT)
        path = ROOT / 'datasets/nextqa/runs/nextqa-cg6-qa-v3-luna-full4996/payload/nextqa_causal_why_coherent_chain_v2_qa_full1927_20260829a/run_config.json'
        contract=json.loads(path.read_text(encoding='utf-8'))['contract']
        with source.open('rb') as stream:
            stream.read(1)
        import hashlib
        if hashlib.sha256(source.read_bytes()).hexdigest() != contract['visual_evidence']['sha256']:
            raise AssertionError('registered causal evidence differs from frozen QA contract')
        return {'registered_evidence_read':str(source),'frozen_contract_hash_matches':True}
    raise ValueError(case)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',choices=('replay','resources','graph_mock','index_defaults','pair_defaults','causal_inputs'))
    parser.add_argument('--run-id', default='prearchive_guard_mock_20260930')
    parser.add_argument('--deny-graphiti-source', action='store_true')
    args=parser.parse_args()
    os.environ['VLKG_NEXTQA_RUN_ROOT']=str(ROOT/'datasets/nextqa/runs/new_runs/payload')
    sys.path[:0]=[str(ROOT),str(ROOT/'src')]
    guard=AccessGuard(ROOT)
    if args.deny_graphiti_source:
        guard.denied_roots.append(os.path.join(guard.root, 'graphiti-main'))
    sys.addaudithook(guard.hook)
    # FAISS C++ file opens are not covered by Python's open audit event.
    import faiss
    original=faiss.read_index
    def checked_read_index(path,*positional,**kwargs):
        guard.check_path(path,'faiss.read_index')
        return original(path,*positional,**kwargs)
    faiss.read_index=checked_read_index
    detail={};error=None
    log=ROOT/'datasets/nextqa/scratch'/('prearchive_'+args.case+'.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    try:
        with log.open('w',encoding='utf-8') as stream, contextlib.redirect_stdout(stream):
            detail=run_case(args.case, args.run_id)
    except (Exception,SystemExit) as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    report={'case':args.case,'status':'passed' if error is None and not guard.denied and not guard.network_attempts else 'blocked',
            'graphiti_source_denied':args.deny_graphiti_source,
            'denied_legacy_accesses':guard.denied,'network_attempts':guard.network_attempts,
            'workspace_accesses':dict(sorted(guard.accesses.items())), 'error':error,'detail':detail,
            'limits':['No OS-level tracing of arbitrary native file access.','Only executed branches are covered.','FAISS read_index explicitly instrumented; directory stat/enumeration not blocked.']}
    out=ROOT/'registry/audits/nextqa_prearchive'/('execution_'+args.case+'.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in report.items() if k!='workspace_accesses'},ensure_ascii=False,indent=2))
    return 0 if report['status']=='passed' else 2


if __name__=='__main__':
    raise SystemExit(main())
