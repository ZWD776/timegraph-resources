"""Run the OpenEQA visual-without-routing ablation.

This runner deliberately shares the canonical graph retrieval path with the
matched graph-only and routed visual controls, but uses one route-neutral
question-text visual query and a generic renderer.  It never reads semantic
Question IR routing fields, official category, or gold answers to construct
visual evidence.
"""
from __future__ import annotations

import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next(p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback
from typing import Any
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import scripts.datasets.openeqa.qa.run_hm3d_graph_only_qa as graph_only
from scripts.datasets.openeqa._shared import load_open_eqa_rows, repo_path, write_json_array_atomic, write_json_atomic
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.qa.canonical_graph_qa_context import retrieve_canonical_graph_context
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession, unified_graph_retrieval_question_ir
from vlkg.datasets.openeqa.qa.graph_qa_pipeline import answer_via_mode
from vlkg.datasets.openeqa.retrieval.visual_evidence import fuse_graph_visual_evidence
from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex
from vlkg.core.memory.types import MemoryQuery
NO_ROUTING_CONTRACT = 'openeqa_no_routing_visual_control.v1'
NO_ROUTING_LABEL = 'no_routing'
DEFAULT_PROFILE = 'canonical_vector_evidence_v5'

def build_parser() -> argparse.ArgumentParser:
    parser = graph_only.build_parser()
    parser.description = 'Run the OpenEQA visual retrieval without routing ablation. Graph retrieval is shared with the matched graph-only control; visual search uses only the raw question text.'
    parser.set_defaults(graph_retrieval_profile=DEFAULT_PROFILE)
    parser.add_argument('--visual-index-dir', type=Path, required=True)
    parser.add_argument('--visual-index-name', default='openeqa_object_siglip_flat_ip')
    parser.add_argument('--visual-top-k', type=int, default=8)
    parser.add_argument('--visual-min-score', type=float, default=0.0)
    parser.add_argument('--visual-max-candidate-packs', type=int, default=8)
    parser.add_argument('--visual-max-evidence-lines', type=int, default=8)
    return parser

def _neutral_visual_payload(visual_index: ObjectVisualSearchIndex, *, episode_history: str, question: str, top_k: int, min_score: float) -> dict[str, Any]:
    normalized_question = ' '.join(str(question or '').split())
    vectors = visual_index.encode_texts([normalized_question])
    hits = visual_index.search_vector(episode_history=episode_history, vector=vectors[0], top_k=max(1, int(top_k)), min_score=float(min_score)) if vectors else []
    intent = {'text': normalized_question, 'role': 'context', 'evidence_unit': 'object_crop', 'source': 'raw_question_no_routing', 'target_terms': [], 'anchor_terms': [], 'uses_question_type': False, 'uses_route': False, 'uses_official_category': False, 'uses_gold_answer': False}
    return {'contract': NO_ROUTING_CONTRACT, 'control_label': NO_ROUTING_LABEL, 'available': bool(hits), 'episode_history': episode_history, 'uses_gold_answer': False, 'uses_question_type': False, 'uses_route': False, 'uses_handler': False, 'uses_primary_need': False, 'uses_evidence_demand': False, 'uses_official_question_category': False, 'episode_scoped_search': True, 'global_topk_then_filter': False, 'episode_candidate_count': visual_index.episode_size(episode_history), 'allows_unbound_visual_observations': True, 'unbound_observations_are_graph_facts': False, 'query_intent_count': 1, 'query_text_source': 'normalized_question_only', 'groups': [{'query_intent': intent, 'rendered_query': normalized_question, 'retrieved_unit': 'object_crop', 'episode_candidate_count': visual_index.episode_size(episode_history), 'score_threshold_count': len(hits), 'hits': hits}], 'retrieved_unit_group_counts': {'object_crop': 1} if hits else {}}

def _neutral_visual_demand() -> dict[str, Any]:
    return {'contract': NO_ROUTING_CONTRACT, 'control_label': NO_ROUTING_LABEL, 'primary_need': '', 'route_id': '', 'handler_id': '', 'strategy_variant': '', 'visual_query_builder_id': '', 'target_terms': [], 'anchor_terms': [], 'attribute_dimensions': [], 'state_terms': [], 'constraint_terms': [], 'spatial_relations': [], 'spatial_constraints': [], 'query_intents': [], 'uses_gold_answer': False, 'uses_question_type': False, 'uses_route': False, 'uses_handler': False, 'uses_primary_need': False, 'uses_evidence_demand': False, 'uses_official_question_category': False, 'source': 'raw_question_only'}

def _audit_record(*, row: dict[str, Any], graph_path: Path, retrieval_mode: str, graph_retrieval_profile: str, question_ir: dict[str, Any] | None, graph_result: dict[str, Any], fused_result: dict[str, Any], llm_answer: dict[str, Any], visual_payload: dict[str, Any], elapsed_sec: float) -> dict[str, Any]:
    graph_matches = [item for item in graph_result.get('matches', []) or [] if isinstance(item, dict)]
    graph_bundle = graph_result.get('llm_ready_bundle') or {}
    visual_evidence = fused_result.get('visual_evidence') or {}
    visual_stats = visual_evidence.get('stats') or {}
    structured = llm_answer.get('structured_answer') or {}
    prediction = str(llm_answer.get('answer_text') or '').strip()
    if not prediction:
        prediction = str(graph_result.get('answer_brief') or '').strip()
    return {'question_id': str(row.get('question_id') or ''), 'episode_history': str(row.get('episode_history') or ''), 'category': str(row.get('category') or ''), 'question': str(row.get('question') or ''), 'prediction': prediction, 'retrieval_mode': retrieval_mode, 'graph_retrieval_profile': graph_retrieval_profile, 'question_ir': question_ir or {}, 'question_type': (question_ir or {}).get('question_type'), 'question_route_id': (question_ir or {}).get('question_route_id'), 'question_route': (question_ir or {}).get('question_route') or {}, 'question_route_decision_source': 'not_used_no_routing_control', 'runtime_route_recompiled': False, 'graph_path': str(graph_path), 'graph_retrieval_source_record_ids': [str(item.get('source_record_id') or '') for item in graph_matches], 'graph_retrieval_source_kinds': [str(item.get('source_kind') or '') for item in graph_matches], 'graph_rag_status': llm_answer.get('graph_rag_status'), 'provider': llm_answer.get('provider'), 'model': llm_answer.get('model'), 'match_count': len(graph_matches), 'evidence_line_count': llm_answer.get('graph_rag_metadata', {}).get('evidence_line_count', 0), 'used_focus_frames': structured.get('used_focus_frames', []), 'used_evidence_lines': structured.get('used_evidence_lines', []), 'answer_brief': graph_result.get('answer_brief'), 'usage': llm_answer.get('response_metadata', {}).get('usage') or {}, 'elapsed_sec': round(elapsed_sec, 3), 'error': llm_answer.get('error'), 'visual_backbone': 'openeqa_no_routing_visual_control', 'visual_runtime_contract': NO_ROUTING_CONTRACT, 'selected_visual_builder': NO_ROUTING_LABEL, 'visual_route_id': NO_ROUTING_LABEL, 'route_strategy_handler': NO_ROUTING_LABEL, 'route_strategy_applied': False, 'strategy_variant': NO_ROUTING_LABEL, 'visual_retrieval_contract': visual_payload.get('contract'), 'visual_retrieval_episode_history': visual_payload.get('episode_history'), 'visual_retrieval_episode_scoped': visual_payload.get('episode_scoped_search'), 'visual_retrieval_global_topk_then_filter': visual_payload.get('global_topk_then_filter'), 'visual_retrieval_uses_gold_answer': False, 'visual_query_source': visual_payload.get('query_text_source'), 'visual_retrieved_units': [str(group.get('retrieved_unit') or '') for group in visual_payload.get('groups', []) or [] if isinstance(group, dict)], 'visual_retrieved_unit_group_counts': visual_payload.get('retrieved_unit_group_counts', {}), 'visual_candidate_packs': visual_evidence.get('candidate_packs', []), 'visual_evidence_lines': visual_evidence.get('evidence_lines', []), 'visual_stats': visual_stats, 'no_routing_contract': {'contract': NO_ROUTING_CONTRACT, 'query_text_source': 'normalized_question_only', 'uses_question_type': False, 'uses_question_route': False, 'uses_route_id': False, 'uses_handler_id': False, 'uses_primary_need': False, 'uses_evidence_demand': False, 'uses_official_question_category': False, 'uses_gold_answer': False, 'route_neutral_renderer': True, 'episode_scoped_search': True, 'graph_retrieval_shared_with_matched_control': True, 'raw_pixels_sent_to_qa': False}}

def _error_record(*, row: dict[str, Any], graph_path: Path | None, error: str) -> dict[str, Any]:
    return {'question_id': str(row.get('question_id') or ''), 'episode_history': str(row.get('episode_history') or ''), 'category': str(row.get('category') or ''), 'question': str(row.get('question') or ''), 'prediction': '', 'graph_path': str(graph_path) if graph_path else None, 'graph_rag_status': 'error', 'error': error, 'visual_backbone': 'openeqa_no_routing_visual_control', 'visual_runtime_contract': NO_ROUTING_CONTRACT, 'selected_visual_builder': NO_ROUTING_LABEL, 'visual_route_id': NO_ROUTING_LABEL, 'route_strategy_handler': NO_ROUTING_LABEL, 'strategy_variant': NO_ROUTING_LABEL, 'question_route_decision_source': 'not_used_no_routing_control', 'runtime_route_recompiled': False, 'no_routing_contract': {'contract': NO_ROUTING_CONTRACT, 'uses_question_type': False, 'uses_question_route': False, 'uses_route_id': False, 'uses_handler_id': False, 'uses_primary_need': False, 'uses_evidence_demand': False, 'uses_official_question_category': False, 'uses_gold_answer': False}}

def _summary(*, run_name: str, output_dir: Path, selected_rows: list[dict[str, Any]], audit: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    usage = Counter()
    for item in audit:
        item_usage = item.get('usage') or {}
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            usage[key] += int(item_usage.get(key) or 0)
    visual = Counter()
    for item in audit:
        for key, value in (item.get('visual_stats') or {}).items():
            if isinstance(value, (int, float)) and (not isinstance(value, bool)):
                visual[key] += value
    statuses = Counter((str(item.get('graph_rag_status') or 'unknown') for item in audit))
    return {'contract': 'openeqa_no_routing_visual_qa_summary.v1', 'run_name': run_name, 'output_dir': str(output_dir), 'generated_at': datetime.now(timezone.utc).isoformat(), 'data_source': args.data_source, 'open_eqa_json': str(repo_path(args.open_eqa_json)), 'graph_manifest': str(repo_path(args.graph_manifest)), 'graph_manifest_run_id': manifest.get('run_id'), 'visual_index_dir': str(repo_path(args.visual_index_dir)), 'visual_index_name': args.visual_index_name, 'visual_backbone': 'openeqa_no_routing_visual_control', 'visual_runtime_contract': NO_ROUTING_CONTRACT, 'answer_mode': args.answer_mode, 'qa_workers': args.qa_workers, 'graph_retrieval_profile': args.graph_retrieval_profile, 'graph_retrieval_contract': graph_only.graph_retrieval_contract(args.graph_retrieval_profile), 'retrieval_mode': args.retrieval_mode, 'retrieval_limit': args.retrieval_limit, 'visual_top_k': args.visual_top_k, 'visual_min_score': args.visual_min_score, 'visual_max_candidate_packs': args.visual_max_candidate_packs, 'visual_max_evidence_lines': args.visual_max_evidence_lines, 'selected_question_count': len(selected_rows), 'completed_question_count': len(audit), 'prediction_count': sum((bool(item.get('question_id')) for item in audit)), 'graph_rag_status_counts': dict(sorted(statuses.items())), 'visual_totals': dict(sorted(visual.items())), 'token_usage': {'prompt_tokens': int(usage.get('prompt_tokens', 0)), 'completion_tokens': int(usage.get('completion_tokens', 0)), 'total_tokens': int(usage.get('total_tokens', 0))}, 'error_count': sum((bool(item.get('error')) for item in audit)), 'uses_gold_answer': False, 'uses_official_question_category_for_routing': False, 'route_purity': {'control_label': NO_ROUTING_LABEL, 'question_type_branches': False}, 'visual_retrieval_episode_scope': 'episode_scoped_exact', 'graph_source_ids_available': True, 'uses_oracle_splice': False, 'no_routing_contract': {'contract': NO_ROUTING_CONTRACT, 'query_text_source': 'normalized_question_only', 'uses_question_type': False, 'uses_question_route': False, 'uses_route_id': False, 'uses_handler_id': False, 'uses_primary_need': False, 'uses_evidence_demand': False, 'uses_official_question_category': False, 'uses_gold_answer': False, 'raw_pixels_sent_to_qa': False}}

def _persist(paths: dict[str, Path], audit: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_json_array_atomic(paths['audit'], audit)
    write_json_array_atomic(paths['predictions'], ({'question_id': str(item.get('question_id') or ''), 'answer': str(item.get('prediction') or '')} for item in audit if item.get('question_id')))
    write_json_atomic(paths['summary_json'], summary)
    paths['summary_md'].write_text('\n'.join([f"# OpenEQA no-routing visual QA: {summary['run_name']}", '', f"- selected/completed: `{summary['selected_question_count']}/{summary['completed_question_count']}`", f"- graph retrieval profile: `{summary['graph_retrieval_profile']}`", f"- visual query: `{summary['no_routing_contract']['query_text_source']}`", f"- errors: `{summary['error_count']}`", '- route-specific handlers: `disabled`']) + '\n', encoding='utf-8')

def main() -> None:
    args = build_parser().parse_args()
    graph_only.pin_qa_api_key(args.config, args.qa_api_key_index)
    run_name = args.run_name or datetime.now().strftime('no_routing_visual_%Y%m%d_%H%M%S')
    output_dir = repo_path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {'run_config': output_dir / 'run_config.json', 'predictions': output_dir / 'predictions.json', 'audit': output_dir / 'audit.json', 'summary_json': output_dir / 'summary.json', 'summary_md': output_dir / 'summary.md'}
    dataset_path = repo_path(args.open_eqa_json)
    graph_manifest_path = repo_path(args.graph_manifest)
    visual_index_dir = repo_path(args.visual_index_dir)
    rows = load_open_eqa_rows(dataset_path)
    manifest = graph_only.load_json(graph_manifest_path)
    graph_index = graph_only.graph_by_episode(manifest)
    question_ir_records = {} if args.skip_question_ir else graph_only.load_question_ir_records(args.question_ir_json)
    selected_rows = graph_only.select_rows(rows, data_source=args.data_source, categories=args.category, episode_histories=args.episode_history, question_ids=[*(args.question_id or []), *graph_only.load_question_id_jsons(args.question_id_json)], offset=args.offset, limit=args.limit)
    issues = graph_only.preflight(args=args, manifest=manifest, selected_rows=selected_rows, graph_index=graph_index, question_ir_records=question_ir_records)
    if args.visual_top_k < 1 or args.visual_max_candidate_packs < 1 or args.visual_max_evidence_lines < 1:
        issues.append('visual budgets must be >= 1')
    if args.graph_retrieval_profile not in graph_only.UNIFIED_GRAPH_RETRIEVAL_PROFILES:
        issues.append('no-routing control requires a unified route-independent graph retrieval profile')
    run_config = {'contract': NO_ROUTING_CONTRACT, 'run_name': run_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'data_source': args.data_source, 'open_eqa_json': str(dataset_path), 'graph_manifest': str(graph_manifest_path), 'question_ir_json': '' if args.skip_question_ir else str(repo_path(args.question_ir_json)), 'visual_index_dir': str(visual_index_dir), 'visual_index_name': args.visual_index_name, 'visual_backbone': 'openeqa_no_routing_visual_control', 'visual_runtime_contract': NO_ROUTING_CONTRACT, 'graph_retrieval_profile': args.graph_retrieval_profile, 'retrieval_mode': args.retrieval_mode, 'retrieval_limit': args.retrieval_limit, 'visual_top_k': args.visual_top_k, 'visual_min_score': args.visual_min_score, 'visual_max_candidate_packs': args.visual_max_candidate_packs, 'visual_max_evidence_lines': args.visual_max_evidence_lines, 'answer_mode': args.answer_mode, 'qa_config': str(args.config), 'qa_workers': args.qa_workers, 'qa_temperature': args.qa_temperature, 'qa_api_key_index': args.qa_api_key_index, 'selected_question_count': len(selected_rows), 'uses_gold_answer': False, 'uses_official_question_category_for_routing': False, 'no_routing_contract': _neutral_visual_demand()}
    write_json_atomic(paths['run_config'], run_config)
    print(f'run_name={run_name}')
    print(f'selected_questions={len(selected_rows)}')
    print(f'graph_retrieval_profile={args.graph_retrieval_profile}')
    print(f'visual_index_dir={visual_index_dir}')
    if issues:
        print('preflight_status=failed')
        for issue in issues:
            print(f'- {issue}')
        raise SystemExit(2)
    print('preflight_status=passed')
    if not args.run:
        print('Use --run to start QA.')
        return
    existing_audit = graph_only.load_existing_audit(paths['audit']) if args.resume else []
    if args.force:
        existing_audit = []
    completed_ids = {str(item.get('question_id') or '') for item in existing_audit if item.get('question_id') and (not item.get('error'))}
    audit_by_id = {str(item.get('question_id') or ''): item for item in existing_audit if item.get('question_id')}
    visual_index = ObjectVisualSearchIndex(index_dir=visual_index_dir, index_name=args.visual_index_name)
    neutral_demand = _neutral_visual_demand()

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        question_id = str(row.get('question_id') or '')
        episode_history = str(row.get('episode_history') or '')
        question = str(row.get('question') or '')
        question_ir = None if args.skip_question_ir else question_ir_records.get(question_id)
        retrieval_ir = unified_graph_retrieval_question_ir(question_ir, graph_retrieval_profile=args.graph_retrieval_profile)
        retrieval_mode = graph_only.resolve_retrieval_mode('', args.retrieval_mode, question_ir=retrieval_ir, graph_retrieval_profile=args.graph_retrieval_profile)
        graph_path = graph_index.get(episode_history)
        if graph_path is None:
            return _error_record(row=row, graph_path=None, error=f'Missing graph for episode_history={episode_history}')
        started = datetime.now()
        try:
            graph = graph_only.load_graph(graph_path)
            session_kwargs: dict[str, Any] = {}
            if args.graph_retrieval_profile in graph_only.CANONICAL_VECTOR_PROFILES:
                session_kwargs = {'vector_recall_cache_path': graph_only.vector_cache_path_for_graph(graph_path), 'vector_recall_model_name': args.vector_model}
            session = CanonicalHybridRetrievalSession(graph, **session_kwargs)
            query = MemoryQuery(query_text=graph_only.query_text_from_question_ir(question, retrieval_ir), metadata={'dataset': 'OpenEQA', 'data_source': args.data_source, 'question_id': question_id, 'episode_history': episode_history, 'question_ir': retrieval_ir or {}, 'graph_retrieval_profile': args.graph_retrieval_profile, 'raw_question': question}, limit=args.retrieval_limit)
            graph_result = retrieve_canonical_graph_context(graph, query, retrieval_mode=retrieval_mode, limit=args.retrieval_limit, retrieval_session=session)
            visual_payload = _neutral_visual_payload(visual_index, episode_history=episode_history, question=question, top_k=args.visual_top_k, min_score=args.visual_min_score)
            fused = fuse_graph_visual_evidence(graph_result, visual_payload, graph=graph, evidence_demand=neutral_demand, max_candidate_packs=args.visual_max_candidate_packs, max_visual_lines=args.visual_max_evidence_lines)
            llm_answer = answer_via_mode(answer_mode=args.answer_mode, config_path=args.config, question=question, graph_result=fused, graph_qa_config_overrides=graph_only.graph_qa_overrides(args))
            return _audit_record(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, graph_result=graph_result, fused_result=fused, llm_answer=llm_answer, visual_payload=visual_payload, elapsed_sec=(datetime.now() - started).total_seconds())
        except Exception as exc:
            return _error_record(row=row, graph_path=graph_path, error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')
    pending = [row for row in selected_rows if str(row.get('question_id') or '') not in completed_ids]
    completed_since_start = 0
    try:
        with ThreadPoolExecutor(max_workers=args.qa_workers) as executor:
            futures = {executor.submit(run_one, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                record = future.result()
                question_id = str(record.get('question_id') or row.get('question_id') or '')
                audit_by_id[question_id] = record
                completed_since_start += 1
                audit = [audit_by_id[str(item.get('question_id') or '')] for item in selected_rows if str(item.get('question_id') or '') in audit_by_id]
                print(f"[{completed_since_start}/{len(selected_rows)}] completed {question_id} status={record.get('graph_rag_status') or 'error'}", flush=True)
                if completed_since_start % args.checkpoint_every == 0:
                    _persist(paths, audit, _summary(run_name=run_name, output_dir=output_dir, selected_rows=selected_rows, audit=audit, args=args, manifest=manifest))
        audit = [audit_by_id[str(item.get('question_id') or '')] for item in selected_rows if str(item.get('question_id') or '') in audit_by_id]
        _persist(paths, audit, _summary(run_name=run_name, output_dir=output_dir, selected_rows=selected_rows, audit=audit, args=args, manifest=manifest))
    finally:
        visual_index.close()
    print(f"predictions={paths['predictions']}")
    print(f"audit={paths['audit']}")
    print(f"summary={paths['summary_json']}")
if __name__ == '__main__':
    main()
