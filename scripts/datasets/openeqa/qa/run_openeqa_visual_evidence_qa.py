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
from vlkg.datasets.openeqa.retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession, unified_graph_retrieval_question_ir
from vlkg.datasets.openeqa.retrieval.evidence_router import IR_FIRST_ROUTING_PROFILES, open_eqa_v0_route_taxonomy, question_route_decision_issues
from vlkg.datasets.openeqa.runtime.execution_contract import ir_first_execution_config_issues, question_ir_routing_profile
from vlkg.datasets.openeqa.runtime.visual_object_index import ObjectVisualSearchIndex, validate_object_visual_index_manifest
from vlkg.datasets.openeqa.retrieval.route_subset import load_route_subset_manifest, validate_route_subset_manifest
from vlkg.datasets.openeqa.qa.visual_qa_pipeline import VISUAL_BACKBONE_LEGACY_V4, VISUAL_BACKBONE_NAVQA_EQUIVALENT, VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, VISUAL_BACKBONE_OPENEQA_QR_V2, VISUAL_BACKBONE_OPENEQA_QR_V3, VISUAL_BACKBONE_OPENEQA_QR_V4, VISUAL_BACKBONE_OPENEQA_QR_V5, VISUAL_BACKBONE_ROUTE_HYBRID_V2, VISUAL_BACKBONES, answer_visual_enhanced_canonical_query
from vlkg.datasets.openeqa.runtime.visual_runtime_v1 import OPENEQA_VISUAL_RUNTIME_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import OPENEQA_VISUAL_RUNTIME_CONTRACT as OPENEQA_VISUAL_RUNTIME_V2_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v3 import OPENEQA_VISUAL_RUNTIME_CONTRACT as OPENEQA_VISUAL_RUNTIME_V3_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v4 import OPENEQA_VISUAL_RUNTIME_CONTRACT as OPENEQA_VISUAL_RUNTIME_V4_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import OPENEQA_VISUAL_RUNTIME_CONTRACT as OPENEQA_VISUAL_RUNTIME_V5_CONTRACT, ROUTE_IDS as V5_ROUTE_IDS, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, route_handler_contract_issues as single_route_handler_contract_issues, single_route_decision_issues
from vlkg.core.memory.types import MemoryQuery
DEFAULT_VISUAL_PROFILE = graph_only.CANONICAL_VECTOR_EVIDENCE_PROFILE
DEFAULT_VISUAL_INDEX_NAME = 'openeqa_object_siglip_flat_ip'

def ir_first_shard_config_issues(*, selected_rows: list[dict[str, Any]], question_ir_records: dict[str, dict[str, Any]], visual_backbone: str, graph_retrieval_profile: str) -> list[str]:
    routing_profiles = {question_ir_routing_profile(question_ir_records.get(str(row.get('question_id') or ''))) for row in selected_rows}
    issues: list[str] = []
    relevant_profiles = routing_profiles & IR_FIRST_ROUTING_PROFILES
    if ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5 in routing_profiles:
        relevant_profiles.add(ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5)
    for routing_profile in sorted(relevant_profiles):
        issues.extend(ir_first_execution_config_issues(routing_profile=routing_profile, visual_backbone=visual_backbone, graph_retrieval_profile=graph_retrieval_profile))
    return issues

def select_v5_question_type_rows(rows: list[dict[str, Any]], *, question_ir_records: dict[str, dict[str, Any]], question_types: list[str] | None, per_type_limit: int | None) -> tuple[list[dict[str, Any]], list[str]]:
    requested = {str(value).strip() for value in question_types or [] if str(value).strip()}
    issues: list[str] = []
    unknown = requested - set(V5_ROUTE_IDS)
    if unknown:
        issues.append(f'unknown V5 question types: {sorted(unknown)}')
    if per_type_limit is not None and per_type_limit < 1:
        issues.append('question-type-limit must be at least 1')
    if per_type_limit is not None and (not requested):
        issues.append('question-type-limit requires question-type')
    if not requested or issues:
        return (list(rows), issues)
    counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row.get('question_id') or '')
        ir = question_ir_records.get(question_id)
        question_type = str((ir or {}).get('question_type') if isinstance(ir, dict) else '').strip()
        if question_type not in requested:
            continue
        if per_type_limit is not None and counts[question_type] >= per_type_limit:
            continue
        counts[question_type] += 1
        selected.append(row)
    missing_types = sorted(requested - set(counts))
    if missing_types:
        issues.append(f'selected source has no rows for question types: {missing_types}')
    return (selected, issues)

def visual_runtime_contract(visual_backbone: str) -> str:
    if visual_backbone in {VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, VISUAL_BACKBONE_NAVQA_EQUIVALENT}:
        return OPENEQA_VISUAL_RUNTIME_CONTRACT
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V2:
        return OPENEQA_VISUAL_RUNTIME_V2_CONTRACT
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V3:
        return OPENEQA_VISUAL_RUNTIME_V3_CONTRACT
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V4:
        return OPENEQA_VISUAL_RUNTIME_V4_CONTRACT
    if visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
        return OPENEQA_VISUAL_RUNTIME_V5_CONTRACT
    if visual_backbone == VISUAL_BACKBONE_ROUTE_HYBRID_V2:
        return 'openeqa_route_hybrid_v2_mixed_runtime'
    return ''

def build_parser() -> argparse.ArgumentParser:
    parser = graph_only.build_parser()
    parser.description = 'Run OpenEQA graph plus EvidenceDemand-routed visual retrieval QA. Graph retrieval remains the unified graph-only path; routing changes only visual retrieval and evidence organization.'
    parser.set_defaults(graph_retrieval_profile=DEFAULT_VISUAL_PROFILE)
    parser.add_argument('--visual-index-dir', type=Path, required=True)
    parser.add_argument('--visual-index-name', default=DEFAULT_VISUAL_INDEX_NAME)
    parser.add_argument('--visual-top-k', type=int, default=8)
    parser.add_argument('--visual-min-score', type=float, default=0.0)
    parser.add_argument('--visual-max-candidate-packs', type=int, default=8)
    parser.add_argument('--visual-max-evidence-lines', type=int, default=8)
    parser.add_argument('--route-subset-manifest', type=Path, default=None, help='Immutable question-ID manifest for exactly one persisted EvidenceDemand.')
    parser.add_argument('--strategy-variant', action='append', default=None, help='Optional persisted strategy_variant filter within the route manifest.')
    parser.add_argument('--question-type', action='append', default=None, help='V5 diagnostic filter over persisted semantic IR question_type.')
    parser.add_argument('--question-type-limit', type=int, default=None, help='Optional deterministic per-question-type limit for V5 pilots.')
    parser.add_argument('--visual-backbone', choices=sorted(VISUAL_BACKBONES), default=VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1)
    return parser

def output_paths(output_dir: Path) -> dict[str, Path]:
    return {'run_config': output_dir / 'run_config.json', 'predictions': output_dir / 'predictions.json', 'audit': output_dir / 'audit.json', 'summary_json': output_dir / 'summary.json', 'summary_md': output_dir / 'summary.md'}

def _audit_record(*, row: dict[str, Any], graph_path: Path, retrieval_mode: str, graph_retrieval_profile: str, question_ir: dict[str, Any] | None, result: dict[str, Any], elapsed_sec: float) -> dict[str, Any]:
    base = graph_only.audit_record_from_result(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=graph_retrieval_profile, question_ir=question_ir, result=result, elapsed_sec=elapsed_sec)
    graph_answer = result.get('graph_answer') or {}
    visual_evidence = graph_answer.get('visual_evidence') or {}
    visual_retrieval = result.get('visual_retrieval') or {}
    llm_ready_bundle = graph_answer.get('llm_ready_bundle') or {}
    selected_visual_builder = str(result.get('selected_visual_builder') or '')
    route_decision_source = str(result.get('question_route_decision_source') or '')
    base.pop('graph_only_contract', None)
    base.update({'pipeline_contract': result.get('contract'), 'visual_backbone': result.get('visual_backbone'), 'selected_visual_builder': selected_visual_builder, 'question_route_decision_source': route_decision_source, 'runtime_route_recompiled': route_decision_source != 'question_ir.question_route', 'evidence_demand': result.get('evidence_demand') or {}, 'visual_retrieval_contract': visual_retrieval.get('contract'), 'visual_retrieval_episode_history': visual_retrieval.get('episode_history'), 'visual_retrieval_episode_scoped': visual_retrieval.get('episode_scoped_search'), 'visual_retrieval_global_topk_then_filter': visual_retrieval.get('global_topk_then_filter'), 'visual_retrieval_uses_gold_answer': visual_retrieval.get('uses_gold_answer'), 'visual_retrieved_unit_group_counts': visual_retrieval.get('retrieved_unit_group_counts') or {}, 'visual_retrieved_units': [str(group.get('retrieved_unit') or '') for group in visual_retrieval.get('groups', []) or [] if isinstance(group, dict)], 'visual_evidence_contract': visual_evidence.get('contract'), 'visual_runtime_contract': visual_evidence.get('visual_runtime_contract'), 'visual_route_strategy_contract': visual_evidence.get('route_strategy_contract'), 'route_strategy_handler': visual_evidence.get('route_strategy_handler') or visual_evidence.get('handler_id'), 'route_handler_id': visual_evidence.get('handler_id') or visual_evidence.get('route_strategy_handler'), 'strategy_variant': visual_evidence.get('strategy_variant'), 'route_strategy_applied': visual_evidence.get('route_strategy_applied'), 'frozen_v3_delegation': visual_evidence.get('frozen_v3_delegation'), 'target_location_contract': visual_evidence.get('target_location_contract'), 'relation_bound_identity_contract': visual_evidence.get('relation_bound_identity_contract'), 'relation_bound_identity_verification': visual_evidence.get('relation_bound_identity_verification') or {}, 'relation_bound_identity_candidates': [{'candidate_id': item.get('candidate_id'), 'entity_label': item.get('entity_label'), 'canonical_entity_id': item.get('canonical_entity_id'), 'relation': item.get('relation'), 'anchor_label': item.get('anchor_label'), 'anchor_entity_id': item.get('anchor_entity_id'), 'ranking_score': item.get('ranking_score'), 'modifier_overlap': item.get('modifier_overlap'), 'constraint_support_count': item.get('constraint_support_count'), 'anchor_component_penalized': item.get('anchor_component_penalized')} for item in visual_evidence.get('object_identity_relation_candidates', []) or [] if isinstance(item, dict)], 'visual_route_id': visual_evidence.get('navqa_route_id'), 'visual_route_policy': visual_evidence.get('route_policy') or {}, 'visual_stats': visual_evidence.get('stats') or {}, 'visual_candidate_packs': visual_evidence.get('candidate_packs') or [], 'visual_spatial_pairs': visual_evidence.get('spatial_pairs') or [], 'visual_evidence_lines': visual_evidence.get('evidence_lines') or [], 'visual_override_contract': llm_ready_bundle.get('visual_override_contract') or {}, 'legacy_or_guarded_visual_builder_used': selected_visual_builder != VISUAL_BACKBONE_NAVQA_EQUIVALENT, 'guarded_visual_authority_present': bool(llm_ready_bundle.get('visual_override_contract')), 'visual_enhanced_contract': {'uses_hsgg_derived_graph': True, 'graph_retrieval_is_route_conditioned': False, 'visual_retrieval_is_evidence_demand_conditioned': True, 'visual_search_is_episode_scoped': True, 'uses_object_crop_embeddings': True, 'unbound_visual_observations_are_graph_facts': False, 'uses_gold_answer_for_routing_retrieval_or_prediction': False}})
    return base

def _error_record(*, row: dict[str, Any], graph_path: Path | None, retrieval_mode: str | None, graph_retrieval_profile: str, question_ir: dict[str, Any] | None, error: str) -> dict[str, Any]:
    base = graph_only.error_audit_record(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=graph_retrieval_profile, question_ir=question_ir, error=error)
    base.pop('graph_only_contract', None)
    base['visual_enhanced_contract'] = {'uses_gold_answer_for_routing_retrieval_or_prediction': False, 'visual_search_is_episode_scoped': True}
    return base

def _summary(*, run_name: str, output_dir: Path, selected_rows: list[dict[str, Any]], audit: list[dict[str, Any]], args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    statuses = Counter((str(item.get('graph_rag_status') or 'unknown') for item in audit))
    needs = Counter((str((item.get('evidence_demand') or {}).get('primary_need') or 'unknown') for item in audit))
    categories = Counter((str(item.get('category') or 'unknown') for item in audit))
    usage = Counter()
    visual = Counter()
    selected_builders = Counter()
    route_strategy_handlers = Counter()
    strategy_variants = Counter()
    route_strategy_applied = Counter()
    frozen_v3_delegation = Counter()
    for item in audit:
        usage.update(item.get('usage') or {})
        visual.update(item.get('visual_stats') or {})
        selected_builders.update([str(item.get('selected_visual_builder') or 'unknown')])
        route_strategy_handlers.update([str(item.get('route_strategy_handler') or 'unknown')])
        strategy_variants.update([str(item.get('strategy_variant') or 'unknown')])
        route_strategy_applied.update([str(item.get('route_strategy_applied'))])
        frozen_v3_delegation.update([str(item.get('frozen_v3_delegation'))])
    return {'contract': 'openeqa_evidence_demand_visual_qa_summary_v1', 'run_name': run_name, 'output_dir': str(output_dir), 'generated_at': datetime.now(timezone.utc).isoformat(), 'data_source': args.data_source, 'open_eqa_json': str(repo_path(args.open_eqa_json)), 'graph_manifest': str(repo_path(args.graph_manifest)), 'graph_manifest_run_id': manifest.get('run_id'), 'visual_index_dir': str(repo_path(args.visual_index_dir)), 'visual_index_name': args.visual_index_name, 'visual_backbone': args.visual_backbone, 'visual_runtime_contract': visual_runtime_contract(args.visual_backbone), 'answer_mode': args.answer_mode, 'qa_workers': args.qa_workers, 'graph_retrieval_profile': args.graph_retrieval_profile, 'graph_retrieval_contract': graph_only.graph_retrieval_contract(args.graph_retrieval_profile), 'evidence_router_contract': 'openeqa_evidence_demand_router.v1', 'evidence_route_taxonomy': open_eqa_v0_route_taxonomy(), 'temporal_visual_route_enabled': False, 'selected_question_count': len(selected_rows), 'completed_question_count': len(audit), 'prediction_count': len(graph_only.predictions_from_audit(audit)), 'category_counts': dict(sorted(categories.items())), 'evidence_need_counts': dict(sorted(needs.items())), 'graph_rag_status_counts': dict(sorted(statuses.items())), 'visual_totals': dict(sorted(visual.items())), 'selected_visual_builder_counts': dict(sorted(selected_builders.items())), 'route_strategy_handler_counts': dict(sorted(route_strategy_handlers.items())), 'strategy_variant_counts': dict(sorted(strategy_variants.items())), 'route_strategy_applied_counts': dict(sorted(route_strategy_applied.items())), 'frozen_v3_delegation_counts': dict(sorted(frozen_v3_delegation.items())), 'token_usage': {'prompt_tokens': int(usage.get('prompt_tokens', 0)), 'completion_tokens': int(usage.get('completion_tokens', 0)), 'total_tokens': int(usage.get('total_tokens', 0))}, 'error_count': sum((1 for item in audit if item.get('error'))), 'contract_checks': {'graph_retrieval_is_route_conditioned': False, 'visual_retrieval_is_evidence_demand_conditioned': True, 'visual_search_is_episode_scoped': True, 'uses_gold_answer': False, 'navqa_strategy_runtime_imported': False, 'openeqa_visual_runtime_isolated': True}}

def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"# OpenEQA Visual Evidence QA: {summary['run_name']}", '', f"- data source: `{summary['data_source']}`", f"- selected/completed: `{summary['selected_question_count']}/{summary['completed_question_count']}`", f"- answer mode: `{summary['answer_mode']}`", f"- graph retrieval profile: `{summary['graph_retrieval_profile']}`", f"- temporal visual route enabled: `{summary['temporal_visual_route_enabled']}`", f"- errors: `{summary['error_count']}`", '', '## Evidence Needs', '']
    for need, count in summary['evidence_need_counts'].items():
        lines.append(f'- `{need}`: `{count}`')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def _persist(paths: dict[str, Path], audit: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    write_json_array_atomic(paths['audit'], audit)
    write_json_array_atomic(paths['predictions'], ({'question_id': str(item.get('question_id') or ''), 'answer': str(item.get('prediction') or '')} for item in audit if item.get('question_id')))
    write_json_atomic(paths['summary_json'], summary)
    _write_summary_md(paths['summary_md'], summary)

def main() -> None:
    args = build_parser().parse_args()
    graph_only.pin_qa_api_key(args.config, args.qa_api_key_index)
    source_slug = args.data_source.replace('-v0', '').replace('-', '_')
    run_name = args.run_name or datetime.now().strftime(f'{source_slug}_visual_evidence_%Y%m%d_%H%M%S')
    output_dir = repo_path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    open_eqa_json = repo_path(args.open_eqa_json)
    graph_manifest_path = repo_path(args.graph_manifest)
    visual_index_dir = repo_path(args.visual_index_dir)
    rows = load_open_eqa_rows(open_eqa_json)
    manifest = graph_only.load_json(graph_manifest_path)
    graph_index = graph_only.graph_by_episode(manifest)
    question_ir_records = {} if args.skip_question_ir else graph_only.load_question_ir_records(args.question_ir_json)
    selected_question_ids = [*(args.question_id or []), *graph_only.load_question_id_jsons(args.question_id_json)]
    route_subset_path = repo_path(args.route_subset_manifest) if args.route_subset_manifest else None
    route_subset_payload: dict[str, Any] | None = None
    route_subset_ids: list[str] | None = None
    route_subset_issues: list[str] = []
    if route_subset_path is not None:
        if args.visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
            route_subset_issues.append('V5 single-route execution does not accept route subsets')
            if args.strategy_variant:
                route_subset_issues.append('V5 single-route execution does not accept strategy_variant filters')
        route_subset_payload = load_route_subset_manifest(route_subset_path)
        route_subset_ids, validation_issues = validate_route_subset_manifest(route_subset_payload, data_source=args.data_source, question_ir_records=question_ir_records, strategy_variants=args.strategy_variant)
        route_subset_issues.extend(validation_issues)
        conflicting_filters = {'category': args.category, 'episode_history': args.episode_history, 'question_id': selected_question_ids, 'question_id_json': args.question_id_json, 'offset': args.offset, 'limit': args.limit}
        for name, value in conflicting_filters.items():
            if value not in (None, 0, [], ()):
                route_subset_issues.append(f"--route-subset-manifest cannot be combined with --{name.replace('_', '-')}")
    elif args.strategy_variant:
        route_subset_issues.append('--strategy-variant requires --route-subset-manifest')
        if args.visual_backbone == VISUAL_BACKBONE_OPENEQA_QR_V5:
            route_subset_issues.append('V5 single-route execution does not accept strategy_variant filters')
    selected_rows = graph_only.select_rows(rows, data_source=args.data_source, categories=args.category, episode_histories=args.episode_history, question_ids=route_subset_ids if route_subset_path else selected_question_ids, offset=args.offset, limit=args.limit)
    if args.question_type and args.visual_backbone != VISUAL_BACKBONE_OPENEQA_QR_V5:
        route_subset_issues.append('question-type filtering requires visual_backbone=openeqa_question_router_v5')
    selected_rows, question_type_issues = select_v5_question_type_rows(selected_rows, question_ir_records=question_ir_records, question_types=args.question_type, per_type_limit=args.question_type_limit)
    route_subset_issues.extend(question_type_issues)
    issues = graph_only.preflight(args=args, manifest=manifest, selected_rows=selected_rows, graph_index=graph_index, question_ir_records=question_ir_records)
    issues.extend(route_subset_issues)
    if route_subset_path is not None and route_subset_ids is not None:
        selected_ids = {str(row.get('question_id') or '').strip() for row in selected_rows}
        missing_ids = sorted(set(route_subset_ids) - selected_ids)
        extra_ids = sorted(selected_ids - set(route_subset_ids))
        if missing_ids:
            issues.append(f'route subset is missing {len(missing_ids)} dataset rows; sample={missing_ids[:5]}')
        if extra_ids:
            issues.append(f'route subset selected {len(extra_ids)} unexpected rows; sample={extra_ids[:5]}')
    if args.checkpoint_every < 1:
        issues.append('checkpoint-every must be at least 1')
    if args.visual_backbone in {VISUAL_BACKBONE_OPENEQA_NAVQA_QR_V1, VISUAL_BACKBONE_OPENEQA_QR_V2, VISUAL_BACKBONE_OPENEQA_QR_V3, VISUAL_BACKBONE_OPENEQA_QR_V4, VISUAL_BACKBONE_OPENEQA_QR_V5}:
        if args.skip_question_ir:
            issues.append('OpenEQA mainline does not allow --skip-question-ir')
        for row in selected_rows:
            question_id = str(row.get('question_id') or '')
            question_ir = question_ir_records.get(question_id)
            routing_profile = question_ir_routing_profile(question_ir)
            route_issues = single_route_decision_issues(question_ir) if routing_profile == ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5 else question_route_decision_issues(question_ir)
            if route_issues:
                issues.append(f'{question_id}: {route_issues[0]}')
                if len(issues) >= 16:
                    break
            if routing_profile == ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5:
                route = (question_ir or {}).get('question_route')
                if isinstance(route, dict):
                    issues.extend((f'{question_id}: {issue}' for issue in single_route_handler_contract_issues(route)))
    issues.extend(ir_first_shard_config_issues(selected_rows=selected_rows, question_ir_records=question_ir_records, visual_backbone=args.visual_backbone, graph_retrieval_profile=args.graph_retrieval_profile))
    for suffix in ('faiss', 'metadata.sqlite', 'manifest.json'):
        path = visual_index_dir / f'{args.visual_index_name}.{suffix}'
        if not path.is_file():
            issues.append(f'Missing visual index file: {path}')
    visual_manifest_path = visual_index_dir / f'{args.visual_index_name}.manifest.json'
    if visual_manifest_path.is_file():
        visual_manifest = graph_only.load_json(visual_manifest_path)
        issues.extend(validate_object_visual_index_manifest(visual_manifest, expected_data_source=args.data_source, expected_graph_manifest_run_id=manifest.get('run_id'), require_full_coverage=True, forbidden_image_root_markers=('uniform32',)))
    run_config = {'contract': 'openeqa_evidence_demand_visual_qa_run_config_v1', 'run_name': run_name, 'created_at': datetime.now(timezone.utc).isoformat(), 'data_source': args.data_source, 'open_eqa_json': str(open_eqa_json), 'graph_manifest': str(graph_manifest_path), 'question_ir_json': '' if args.skip_question_ir else str(repo_path(args.question_ir_json)), 'route_subset_manifest': str(route_subset_path or ''), 'route_subset_primary_need': str((route_subset_payload or {}).get('primary_need') or ''), 'route_subset_strategy_variants': sorted(args.strategy_variant or []), 'question_type_filters': sorted(args.question_type or []), 'question_type_limit': args.question_type_limit, 'question_id_jsons': [str(repo_path(path)) for path in args.question_id_json or []], 'visual_index_dir': str(visual_index_dir), 'visual_index_name': args.visual_index_name, 'visual_backbone': args.visual_backbone, 'visual_runtime_contract': visual_runtime_contract(args.visual_backbone), 'output_dir': str(output_dir), 'answer_mode': args.answer_mode, 'qa_config': str(args.config), 'qa_max_tokens': args.max_tokens, 'qa_temperature': args.qa_temperature, 'qa_api_key_index': args.qa_api_key_index, 'qa_workers': args.qa_workers, 'retrieval_mode': args.retrieval_mode, 'retrieval_limit': args.retrieval_limit, 'graph_retrieval_profile': args.graph_retrieval_profile, 'graph_retrieval_contract': graph_only.graph_retrieval_contract(args.graph_retrieval_profile), 'vector_model': args.vector_model, 'evidence_route_taxonomy': open_eqa_v0_route_taxonomy(), 'visual_top_k': args.visual_top_k, 'visual_min_score': args.visual_min_score, 'visual_max_candidate_packs': args.visual_max_candidate_packs, 'visual_max_evidence_lines': args.visual_max_evidence_lines, 'checkpoint_every': args.checkpoint_every, 'selected_question_count': len(selected_rows), 'uses_gold_answer': False}
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
    elif args.resume:
        existing_audit = [item for item in existing_audit if not item.get('error')]
    completed_ids = {str(item.get('question_id') or '') for item in existing_audit if item.get('question_id') and (not item.get('error'))}
    audit_by_id = {str(item.get('question_id') or ''): item for item in existing_audit if item.get('question_id')}
    visual_index = ObjectVisualSearchIndex(index_dir=visual_index_dir, index_name=args.visual_index_name)
    pending_checkpoint = False
    latest_summary: dict[str, Any] | None = None
    try:
        pending_rows = [(ordinal, row) for ordinal, row in enumerate(selected_rows, start=1) if str(row.get('question_id') or '') not in completed_ids]

        def run_one(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
            ordinal, row = item
            question_id = str(row.get('question_id') or '')
            episode_history = str(row.get('episode_history') or '')
            category = str(row.get('category') or '')
            question = str(row.get('question') or '')
            question_ir = None if args.skip_question_ir else question_ir_records.get(question_id)
            retrieval_ir = unified_graph_retrieval_question_ir(question_ir, graph_retrieval_profile=args.graph_retrieval_profile)
            retrieval_mode = graph_only.resolve_retrieval_mode(category, args.retrieval_mode, question_ir=retrieval_ir, graph_retrieval_profile=args.graph_retrieval_profile)
            graph_path = graph_index.get(episode_history)
            if graph_path is None:
                return _error_record(row=row, graph_path=None, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, error=f'Missing graph for episode_history={episode_history}')
            started = datetime.now()
            try:
                graph = graph_only.load_graph(graph_path)
                session_kwargs: dict[str, Any] = {}
                if args.graph_retrieval_profile in graph_only.CANONICAL_VECTOR_PROFILES:
                    session_kwargs = {'vector_recall_cache_path': graph_only.vector_cache_path_for_graph(graph_path), 'vector_recall_model_name': args.vector_model}
                retrieval_session = CanonicalHybridRetrievalSession(graph, **session_kwargs)
                query = MemoryQuery(query_text=graph_only.query_text_from_question_ir(question, retrieval_ir), metadata={'dataset': 'OpenEQA', 'data_source': args.data_source, 'question_id': question_id, 'episode_history': episode_history, 'question_ir': retrieval_ir or {}, 'graph_retrieval_profile': args.graph_retrieval_profile, 'raw_question': question}, limit=args.retrieval_limit)
                result = answer_visual_enhanced_canonical_query(graph, query, question=question, episode_history=episode_history, category=category, question_ir=question_ir, visual_index=visual_index, retrieval_mode=retrieval_mode, answer_mode=args.answer_mode, config_path=args.config, limit=args.retrieval_limit, visual_top_k=args.visual_top_k, visual_min_score=args.visual_min_score, visual_max_candidate_packs=args.visual_max_candidate_packs, visual_max_evidence_lines=args.visual_max_evidence_lines, retrieval_session=retrieval_session, graph_qa_config_overrides=graph_only.graph_qa_overrides(args), visual_backbone=args.visual_backbone)
                return _audit_record(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, result=result, elapsed_sec=(datetime.now() - started).total_seconds())
            except Exception as exc:
                return _error_record(row=row, graph_path=graph_path, retrieval_mode=retrieval_mode, graph_retrieval_profile=args.graph_retrieval_profile, question_ir=question_ir, error=f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}')

        def ordered_audit() -> list[dict[str, Any]]:
            return [audit_by_id[str(row.get('question_id') or '')] for row in selected_rows if str(row.get('question_id') or '') in audit_by_id]
        completed_since_start = 0
        with ThreadPoolExecutor(max_workers=args.qa_workers) as executor:
            futures = {executor.submit(run_one, item): item for item in pending_rows}
            for future in as_completed(futures):
                ordinal, row = futures[future]
                record = future.result()
                question_id = str(record.get('question_id') or row.get('question_id') or '')
                audit_by_id[question_id] = record
                completed_since_start += 1
                audit = ordered_audit()
                print(f"[{ordinal}/{len(selected_rows)}] completed {question_id} status={record.get('graph_rag_status') or 'error'}", flush=True)
                latest_summary = _summary(run_name=run_name, output_dir=output_dir, selected_rows=selected_rows, audit=audit, args=args, manifest=manifest)
                pending_checkpoint = True
                if completed_since_start % args.checkpoint_every == 0:
                    _persist(paths, audit, latest_summary)
                    pending_checkpoint = False
        if pending_checkpoint and latest_summary is not None:
            _persist(paths, ordered_audit(), latest_summary)
    finally:
        visual_index.close()
    print(f"predictions={paths['predictions']}")
    print(f"audit={paths['audit']}")
    print(f"summary={paths['summary_json']}")
if __name__ == '__main__':
    main()
