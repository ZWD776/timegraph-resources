"""Persist OpenEQA single-route decisions already made by semantic IR.

This builder intentionally does not classify questions.  The source semantic
IR must contain one explicit ``question_type``; missing or unknown values are
contract errors rather than opportunities for heuristic fallback.
"""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import load_open_eqa_rows, repo_path, write_json_atomic
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_CONTRACT, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, route_handler_id, route_handler_contract_issues, route_visual_query_builder_id, route_visual_strategy_contract, single_route_decision_issues
OUTPUT_CONTRACT = 'openeqa_unified_question_route_ir.v1'

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Persist OpenEQA route decisions from an explicit semantic IR question_type without runtime reclassification.')
    parser.add_argument('--open-eqa-json', type=Path, default=(REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--source-question-ir', type=Path, action='append', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    return parser

def load_source_records(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
        rows = payload.get('records', []) if isinstance(payload, dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get('question_id') or row.get('uuid') or '').strip()
            if not question_id:
                continue
            if question_id in result:
                raise ValueError(f'Duplicate source semantic IR ID: {question_id}')
            result[question_id] = row
    return result

def normalize_ir(source_record: dict[str, Any], *, question: str, question_id: str, data_source: str, episode_history: str) -> dict[str, Any]:
    raw_ir = source_record.get('llm_question_ir') or source_record.get('question_ir')
    ir = dict(raw_ir) if isinstance(raw_ir, dict) else {}
    question_type = str(ir.get('question_type') or ir.get('openeqa_question_type') or '').strip()
    if not question_type:
        raise ValueError(f'{question_id}: semantic IR must provide question_type; single-route builder will not infer it')
    ir = normalize_strategy_parameters(ir, question_type=question_type)
    ir.update({'schema_version': 'question_ir.v3', 'dataset': 'OpenEQA', 'data_source': data_source, 'question_id': question_id, 'episode_history': episode_history, 'question_type': question_type, 'uses_gold_answer': False, 'route_decision_is_persisted': True, 'route_decision_runtime_recompile_allowed': False})
    ir.pop('category', None)
    route = {'contract': ROUTE_CONTRACT, 'route_id': question_type, 'question_type': question_type, 'route_source': 'semantic_question_ir.question_type', 'handler_id': '', 'visual_query_builder_id': '', 'visual_strategy_contract': '', 'uses_gold_answer': False, 'uses_official_question_category': False, 'runtime_route_recompile_allowed': False, 'metadata': {'routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, 'official_category_used_for_routing': False, 'routing_input_contract': 'semantic_question_ir.question_type_only'}}
    route['handler_id'] = route_handler_id(route)
    route['visual_query_builder_id'] = route_visual_query_builder_id(route)
    route['visual_strategy_contract'] = route_visual_strategy_contract(route)
    ir['question_route'] = route
    ir['question_route_id'] = question_type
    issues = single_route_decision_issues(ir)
    if issues:
        raise ValueError(f'{question_id}: {issues[0]}')
    handler_issues = route_handler_contract_issues(route)
    if handler_issues:
        raise ValueError(f'{question_id}: {handler_issues[0]}')
    return ir

def normalize_strategy_parameters(ir: dict[str, Any], *, question_type: str) -> dict[str, Any]:
    normalized = dict(ir)
    if question_type == 'relation_bound_identity':
        return normalized
    hard_constraints = [dict(item) for item in normalized.get('hard_constraints', []) or [] if isinstance(item, dict)]
    constraint_text = ' '.join((str(item.get('text') or '').strip() for item in hard_constraints if str(item.get('text') or '').strip())).strip()
    targets = [str(value).strip() for value in normalized.get('target_entities', []) or [] if str(value).strip()]
    if question_type == 'surface_capacity':
        activity_match = re.search('\\b(?:space available|available space|room)\\s+to\\s+(.+)$', constraint_text, re.IGNORECASE)
        activity = str(activity_match.group(1) if activity_match else '').strip()
        if activity:
            normalized['target_state'] = f'has room to {activity}'
            if targets:
                normalized['hard_constraints'] = [{'kind': 'state', 'text': f'room on the {targets[0]} to {activity}', 'role': 'target_modifier', 'target_entities': [targets[0]], 'must_bind_to': ['answer_entity']}]
        elif not str(normalized.get('target_state') or '').strip() and constraint_text:
            normalized['target_state'] = constraint_text
        if not str(normalized.get('attribute_dimension') or '').strip():
            normalized['attribute_dimension'] = 'state'
        if not normalized.get('spatial_constraints') and targets:
            normalized['spatial_constraints'] = [{'relation': 'on', 'target': targets[0]}]
    elif question_type == 'entity_state':
        normalized['attribute_dimension'] = str(normalized.get('attribute_dimension') or '').strip() or 'state'
        if normalized.get('operation') == 'state_query':
            normalized['operation'] = 'attribute_query'
    elif question_type == 'grounded_world_knowledge':
        normalized['target_entities'] = [target for target in targets if target.casefold() not in {'context', 'scene', 'setting'}] or targets
        if not str(normalized.get('target_state') or '').strip() and constraint_text:
            normalized['target_state'] = constraint_text
    elif question_type in {'content_description', 'scene_description'}:
        scene_scope = {'area', 'home', 'house', 'place', 'room', 'scene', 'space'}
        content_carriers = {'art', 'artwork', 'image', 'painting', 'photo', 'picture', 'poster', 'screen', 'tapestry'}
        target_tokens = {token for target in targets for token in target.casefold().replace('-', ' ').split()}
        if question_type == 'content_description':
            if not target_tokens & content_carriers:
                normalized['operation'] = 'identity_query'
        elif not target_tokens & scene_scope:
            normalized['operation'] = 'attribute_query'
    elif question_type == 'text_reading':
        if not any((target.casefold() == 'written text' for target in targets)):
            normalized['target_entities'] = ['written text', *targets]
    else:
        return normalized
    normalized['strategy_parameter_normalization'] = {'contract': 'openeqa_v5_strategy_parameter_normalization.v1', 'source': 'persisted_question_type_and_semantic_ir_parameters', 'runtime_reclassification': False}
    return normalized

def build_payload(*, open_eqa_json: Path, source_ir_paths: list[Path], rows: list[dict[str, Any]], source_records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    route_counts: Counter[str] = Counter()
    handler_counts: Counter[str] = Counter()
    missing: list[str] = []
    for index, row in enumerate(rows, start=1):
        question_id = str(row.get('question_id') or '').strip()
        source = source_records.get(question_id)
        if source is None:
            missing.append(question_id)
            continue
        episode_history = str(row.get('episode_history') or '').strip()
        data_source = episode_history.partition('/')[0]
        ir = normalize_ir(source, question=str(row.get('question') or '').strip(), question_id=question_id, data_source=data_source, episode_history=episode_history)
        route = ir['question_route']
        route_counts.update([str(route['route_id'])])
        handler_counts.update([str(route['handler_id'])])
        records.append({'index': index, 'question_id': question_id, 'uuid': question_id, 'question': str(row.get('question') or '').strip(), 'question_category': str(row.get('category') or '').strip(), 'episode_history': episode_history, 'data_source': data_source, 'llm_question_ir': ir})
    if missing:
        raise ValueError(f'Missing {len(missing)} source IR records: {missing[:8]}')
    return {'contract': OUTPUT_CONTRACT, 'route_contract': ROUTE_CONTRACT, 'routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, 'generated_at': datetime.now(timezone.utc).isoformat(), 'dataset': 'OpenEQA', 'open_eqa_json': str(open_eqa_json), 'source_question_irs': [str(path) for path in source_ir_paths], 'question_count': len(records), 'uses_gold_answer': False, 'uses_official_question_category_for_routing': False, 'routing_inputs': ['semantic_question_ir.question_type'], 'runtime_route_recompile_allowed': False, 'persists_route_strategy_handler': True, 'persists_visual_strategy_policy': True, 'summary': {'route_counts': dict(sorted(route_counts.items())), 'handler_counts': dict(sorted(handler_counts.items())), 'route_contract_issue_count': 0}, 'records': records}

def main() -> None:
    args = build_parser().parse_args()
    open_eqa_json = repo_path(args.open_eqa_json)
    source_ir_paths = [repo_path(path) for path in args.source_question_ir]
    output_json = repo_path(args.output_json)
    if output_json.exists():
        raise FileExistsError(f'Refusing to overwrite route manifest: {output_json}')
    rows = load_open_eqa_rows(open_eqa_json)
    payload = build_payload(open_eqa_json=open_eqa_json, source_ir_paths=source_ir_paths, rows=rows, source_records=load_source_records(source_ir_paths))
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_json, payload)
    print(json.dumps(payload['summary'], ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    main()
