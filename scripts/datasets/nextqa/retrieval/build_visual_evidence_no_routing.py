"""Build the uniform visual-retrieval evidence for the NExT-QA ablation."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
from vlkg.core.assets.asset_registry import asset_file
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from vlkg.datasets.nextqa.runtime.io_utils import atomic_write_json, require_within_nextqa_scratch
from vlkg.datasets.nextqa.runtime.run_snapshot import collect_imported_repo_sources, fingerprint_file, persist_drift_report, persist_run_config, persist_source_snapshot
from vlkg.datasets.nextqa.runtime.visual_runtime_no_routing import ABLATION_LABEL, EVIDENCE_LIMIT, TOP_K_PER_KIND, render_neutral_hit, retrieve_uniform_candidates
from vlkg.datasets.nextqa.runtime.visual_runtime_v1.indexing import CANDIDATE_KINDS
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.indexing import VisualShardStore
from vlkg.core.indexing.vision import VisionFeatureExtractor
DEFAULT_IR = (REPO_ROOT / 'datasets/nextqa/ir/nextqa-ir-v3-luna-full4996/payload/nextqa-ir-v3-luna-full4996-20260817b/question_ir.retrieval.json')
DEFAULT_GRAPH = (REPO_ROOT / 'datasets/nextqa/retrieval/nextqa-cg6-retrieval-v3-route-policy-v3/payload/nextqa_visual_graph_baseline_v2_irv3_full4996_20260818a/frozen_graph_evidence.json')
# Resolve selected payloads only when running, so --help works without downloads.
DEFAULT_INDEX = None
DEFAULT_CONFIG = (REPO_ROOT / 'configs/datasets/nextqa/visual_runtime.json')
DEFAULT_PILOT = (REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'dry_run' / 'pilot5' / 'nextqa_val_pilot5_manifest.json'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build isolated NExT-QA visual evidence without routing.')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--question-ir', type=Path, default=DEFAULT_IR)
    parser.add_argument('--frozen-graph-evidence', type=Path, default=DEFAULT_GRAPH)
    parser.add_argument('--visual-index-manifest', type=Path, default=DEFAULT_INDEX)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--pilot-manifest', type=Path, default=DEFAULT_PILOT)
    parser.add_argument('--scope', choices=('pilot', 'full'), default='pilot')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--resume', action='store_true')
    return parser

def _load_rows(path: Path, name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    if not isinstance(payload, list) or not all((isinstance(row, dict) for row in payload)):
        raise ValueError(f'{name} must be a JSON list of objects')
    return payload

def _index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id or question_id in result:
            raise ValueError(f'{name} contains an empty or duplicate question ID: {question_id}')
        result[question_id] = row
    return result

def _pilot_videos(path: Path) -> set[str]:
    payload = json.loads(path.resolve().read_text(encoding='utf-8'))
    ids = {str(item['official_video_id']) for item in payload.get('videos', [])}
    if len(ids) != 5:
        raise ValueError('pilot manifest must contain five videos')
    return ids

def _run_root(run_id: str, resume: bool) -> Path:
    root = require_within_nextqa_scratch((REPO_ROOT / 'datasets/nextqa/runs/new_runs/payload') / 'ablation' / run_id)
    if root.exists() and (not resume):
        raise FileExistsError(f'run already exists; use --resume explicitly: {root}')
    if not root.exists() and resume:
        raise FileNotFoundError(f'cannot resume missing run: {root}')
    root.mkdir(parents=True, exist_ok=resume)
    return root

def main() -> int:
    args = build_parser().parse_args()
    question_path = args.question_ir.resolve()
    graph_path = args.frozen_graph_evidence.resolve()
    index_path = args.visual_index_manifest.resolve() if args.visual_index_manifest else asset_file('nextqa', 'indexes', 'manifest', REPO_ROOT)
    config_path = args.config.resolve()
    pilot_path = args.pilot_manifest.resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('schema_version') != 'nextqa_visual_runtime_config_v2':
        raise ValueError('no-routing builder requires the audited visual index config')
    questions = _index(_load_rows(question_path, 'question IR'), 'question IR')
    graphs = _index(_load_rows(graph_path, 'frozen graph evidence'), 'frozen graph evidence')
    if set(questions) != set(graphs):
        raise ValueError('question IR and frozen graph evidence IDs do not align')
    selected_ids = set(questions)
    if args.scope == 'pilot':
        selected_ids = {qid for qid in selected_ids if str(questions[qid]['video_id']) in _pilot_videos(pilot_path)}
    question_ids = sorted(selected_ids, key=lambda qid: (str(questions[qid]['video_id']), qid))
    expected = 39 if args.scope == 'pilot' else 4996
    if len(question_ids) != expected:
        raise ValueError(f'{args.scope} scope requires {expected} questions, found {len(question_ids)}')
    run_root = _run_root(args.run_id, args.resume)
    contract = {'dataset': 'NExT-QA', 'split': 'val', 'stage': 'visual_evidence_no_routing_v1', 'scope': args.scope, 'question_ir': fingerprint_file(question_path), 'frozen_graph_evidence': fingerprint_file(graph_path), 'visual_index_manifest': fingerprint_file(index_path), 'config': fingerprint_file(config_path), 'pilot_manifest': fingerprint_file(pilot_path) if args.scope == 'pilot' else None, 'question_count': len(question_ids), 'candidate_kinds': list(CANDIDATE_KINDS), 'top_k_per_kind': TOP_K_PER_KIND, 'visual_evidence_limit': EVIDENCE_LIMIT, 'query_method': 'normalized_question_siglip_text_v1', 'sort_rule': 'similarity_desc_candidate_id_asc', 'ablation_label': ABLATION_LABEL, 'uses_route': False, 'uses_handler': False, 'uses_intent': False, 'uses_visual_evidence_demand': False, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_gold_answer': False, 'changes_graph_retrieval': False, 'calls_api': False, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'device': args.device}
    persist_run_config(run_root / 'run_config.json', contract=contract, resume=args.resume, schema_version='nextqa_visual_no_routing_run_config_v1')
    sources = collect_imported_repo_sources(extra_paths=[Path(__file__)])
    snapshot = persist_source_snapshot(run_root / 'source_snapshot.json', source_paths=sources, resume=args.resume)
    store = VisualShardStore(index_path)
    extractor = VisionFeatureExtractor(model_name=str(config['embedding_model']), device=None if args.device == 'auto' else args.device, local_files_only=True)
    vectors = extractor.encode_text_batch([str(questions[qid].get('normalized_question') or '').strip() for qid in question_ids])
    records: list[dict[str, Any]] = []
    preview: list[dict[str, Any]] = []
    kind_counts: Counter[str] = Counter()
    for qid, vector in zip(question_ids, vectors, strict=True):
        question = questions[qid]
        video_id = str(question['video_id'])
        graph = graphs[qid]
        hits = retrieve_uniform_candidates(store, video_id=video_id, query_vector=vector)
        evidence: list[dict[str, Any]] = []
        used_ids: list[str] = []
        for rank, hit in enumerate(hits, start=1):
            text = render_neutral_hit(hit)
            if not text:
                continue
            candidate_id = str(hit['candidate_id'])
            evidence.append({'rank': len(evidence) + 1, 'candidate_id': candidate_id, 'candidate_kind': str(hit['candidate_kind']), 'similarity': float(hit.get('similarity') or 0.0), 'text': text})
            used_ids.append(candidate_id)
            kind_counts[str(hit['candidate_kind'])] += 1
        record = {'schema_version': 'nextqa_visual_no_routing_evidence_record_v1', 'question_id': qid, 'video_id': video_id, 'ablation_label': ABLATION_LABEL, 'query_method': 'normalized_question_siglip_text_v1', 'candidate_kinds': list(CANDIDATE_KINDS), 'top_k_per_kind': TOP_K_PER_KIND, 'graph_retrieval_status': str(graph.get('status') or ''), 'graph_retrieval_profile': str(graph.get('graph_retrieval_profile') or ''), 'graph_record_ids': list(graph.get('graph_record_ids', [])), 'graph_record_ids_sha256': str(graph.get('graph_record_ids_sha256') or ''), 'graph_evidence_count': len(graph.get('evidence', [])), 'visual_evidence': evidence, 'visual_candidate_ids': used_ids, 'visual_candidate_count': len(evidence), 'uses_gold_answer': False, 'uses_official_type': False, 'uses_answer_choices': False, 'uses_route': False, 'uses_handler': False, 'uses_intent': False, 'uses_visual_evidence_demand': False, 'changes_graph_retrieval': False, 'offline_reads_visual_index': True, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'calls_api': False}
        records.append(record)
        preview.append({'question_id': qid, 'video_id': video_id, 'question': str(question.get('normalized_question') or ''), 'graph_evidence': [{'rank': item.get('rank'), 'text': item.get('evidence_text') or ''} for item in graph.get('evidence', [])], 'visual_evidence': [{'rank': item['rank'], 'candidate_kind': item['candidate_kind'], 'text': item['text']} for item in evidence]})
        store.release_video(video_id)
    serialized = json.dumps([row['visual_evidence'] for row in records], ensure_ascii=False).lower()
    checks = {'question_count_exact': len(records) == expected, 'question_ids_unique': len({row['question_id'] for row in records}) == len(records), 'graph_record_ids_preserved': all((row['graph_record_ids'] == graphs[row['question_id']].get('graph_record_ids', []) and row['graph_record_ids_sha256'] == graphs[row['question_id']].get('graph_record_ids_sha256') for row in records)), 'same_video_only': all((str(row['video_id']) == str(graphs[row['question_id']].get('video_id')) for row in records)), 'fixed_no_routing_label': all((row.get('ablation_label') == ABLATION_LABEL for row in records)), 'fixed_candidate_inventory': all((tuple(row.get('candidate_kinds', [])) == tuple(CANDIDATE_KINDS) and row.get('top_k_per_kind') == TOP_K_PER_KIND for row in records)), 'visual_evidence_nonempty': all((bool(row.get('visual_evidence')) for row in records)), 'visual_evidence_budget': all((0 < len(row.get('visual_evidence', [])) <= EVIDENCE_LIMIT for row in records)), 'no_route_fields': all(('route_id' not in row and 'handler_id' not in row for row in records)), 'no_forbidden_transport_fields': not any((token in serialized for token in ('image_path', 'embedding', '.png', '.jpg', '.jpeg', 'official_type', 'gold_choice'))), 'purity_flags': all((row.get(key) is False for row in records for key in ('uses_gold_answer', 'uses_official_type', 'uses_answer_choices', 'uses_route', 'uses_handler', 'uses_intent', 'uses_visual_evidence_demand', 'qa_receives_images', 'qa_receives_image_paths', 'qa_receives_visual_embeddings')))}
    audit = {'schema_version': 'nextqa_visual_no_routing_evidence_audit_v1', 'run_id': args.run_id, 'question_count': len(records), 'video_count': len({row['video_id'] for row in records}), 'selected_candidate_kind_distribution': dict(sorted(kind_counts.items())), 'checks': checks, 'failed_checks': [name for name, passed in checks.items() if not passed], 'calls_api': False, 'status': 'passed' if all(checks.values()) else 'failed'}
    atomic_write_json(run_root / 'visual_evidence_records.json', records)
    atomic_write_json(run_root / 'qa_text_input_preview.json', preview)
    atomic_write_json(run_root / 'audit.json', audit)
    atomic_write_json(run_root / 'visual_evidence_manifest.json', {'schema_version': 'nextqa_visual_no_routing_evidence_manifest_v1', 'run_id': args.run_id, 'visual_evidence_records_path': str((run_root / 'visual_evidence_records.json').resolve()), 'question_count': len(records), 'ablation_label': ABLATION_LABEL})
    drift = persist_drift_report(run_root / 'source_drift_report.json', snapshot=snapshot, current_source_paths=collect_imported_repo_sources(extra_paths=[Path(__file__)]))
    if drift['code_drift_detected']:
        audit['status'] = 'invalid_code_drift'
        atomic_write_json(run_root / 'audit.json', audit)
    atomic_write_json(run_root / 'run_status.json', {'status': audit['status'], 'question_count': len(records), 'calls_api': False, 'code_drift_detected': drift['code_drift_detected']})
    if audit['status'] != 'passed':
        raise SystemExit(f"no-routing visual evidence audit failed: {audit['failed_checks']}")
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
