"""Compile raw OpenEQA questions into the frozen V5 14-class semantic IR."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
SRC_ROOT = REPO_ROOT / 'src'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from scripts.datasets.openeqa._shared import load_open_eqa_rows, repo_path, write_json_atomic
from vlkg.datasets.openeqa.runtime.ir_transport import _openeqa_post_payload
from vlkg.core.config.config import load_domain_config
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_DEFINITIONS, ROUTE_IDS, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, TAXONOMY_CONTRACT
OUTPUT_CONTRACT = 'openeqa_semantic_question_ir.v5'
IR_SCHEMA = 'openeqa_semantic_ir.v5_14class'
ALLOWED_ANSWER_TYPES = frozenset({'binary', 'count', 'location_text', 'text'})
ALLOWED_OPERATIONS = frozenset({'attribute_query', 'count_query', 'descriptive_query', 'existence_query', 'functional_query', 'identity_query', 'location_query', 'spatial_relation', 'state_query', 'world_knowledge_query'})
FORBIDDEN_IR_FIELDS = frozenset({'category', 'evidence_demand', 'primary_need', 'question_route', 'question_route_id', 'route_id', 'strategy_variant'})

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compile raw OpenEQA questions into the frozen 14-class V5 semantic IR without answers, graph evidence, visual evidence, or dataset labels.')
    parser.add_argument('--open-eqa-json', type=Path, default=(REPO_ROOT / 'datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--data-source', choices=('hm3d-v0', 'scannet-v0'), required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    parser.add_argument('--config', default='datasets/openeqa/qa.yaml')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--question-id', action='append', default=None)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=6000)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--timeout-sec', type=float, default=None)
    parser.add_argument('--semantic-retries', type=int, default=2)
    parser.add_argument('--api-key-index', type=int, default=None, help='Pin this process to one zero-based key-pool slot without logging it.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    return parser

def _text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = _text(item)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result

def _object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]

def pin_api_key(config: Any, index: int | None) -> None:
    if index is None:
        return
    if index < 0:
        raise ValueError('api-key-index must be non-negative')
    pool_var = str(config.api_key_env_var)
    if not pool_var.endswith('_KEYS'):
        pool_var = f'{pool_var}S'
    raw_pool = os.getenv(pool_var, '')
    try:
        parsed = json.loads(raw_pool)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, list):
        pool = [_text(value) for value in parsed if _text(value)]
    else:
        import re
        pool = [value.strip() for value in re.split('[,;\\r\\n]+', raw_pool) if value.strip()]
    if index >= len(pool):
        raise ValueError(f'api-key-index {index} is outside the configured pool size {len(pool)}')
    os.environ[str(config.api_key_env_var)] = pool[index]
    os.environ.pop(pool_var, None)

def select_rows(rows: list[dict[str, Any]], *, data_source: str, offset: int=0, limit: int | None=None, question_ids: list[str] | None=None) -> list[dict[str, Any]]:
    selected = [row for row in rows if _text(row.get('episode_history')).partition('/')[0] == data_source]
    requested_ids = {_text(value) for value in question_ids or [] if _text(value)}
    if requested_ids:
        selected = [row for row in selected if _text(row.get('question_id') or row.get('uuid')) in requested_ids]
    if offset:
        selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    return selected

def build_prompt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taxonomy = [{'question_type': route_id, 'definition': ROUTE_DEFINITIONS[route_id]} for route_id in sorted(ROUTE_IDS)]
    return {'task': 'compile_raw_openeqa_questions_to_semantic_ir', 'do_not_answer_questions': True, 'input_contract': {'allowed_fields': ['uuid', 'question'], 'metadata_labels_are_not_provided': True, 'answers_are_not_provided': True, 'graph_evidence_is_not_provided': True, 'visual_evidence_is_not_provided': True}, 'question_type_taxonomy': taxonomy, 'decision_rules': ['Choose exactly one question_type from the taxonomy.', 'relation_bound_identity identifies an unknown entity from anchors or relations; relation_query describes or verifies a relation between known entities.', 'direct_identity identifies a specified entity without requiring an anchor relation.', 'entity_state covers open, closed, powered, clean, full, empty, and other object states; keep the concrete state in target_state.', 'entity_existence is only for whether a target exists; surface_capacity is only for available surface space or activity capacity.', 'target_location asks where a known target is; relation_bound_identity asks what an unknown relation-bound target is.', 'entity_count covers all counting; keep filters in spatial_constraints, hard_constraints, attribute_dimension, or target_state.', 'text_reading covers visible words, dates, signs, labels, and logos; keep ownership or location constraints in IR parameters.', 'functional_affordance asks about use, purpose, possible action, or recommendation.', 'grounded_world_knowledge requires contextual commonsense but is not a direct use or purpose question.', 'scene_description classifies or describes a room or whole scene; content_description describes depicted content in a painting, image, poster, or screen.', 'Parameters refine evidence retrieval but never create another question type.', 'If no taxonomy value is defensible, omit question_type; never invent a fallback type.'], 'operation_contract': sorted(ALLOWED_OPERATIONS), 'answer_type_contract': sorted(ALLOWED_ANSWER_TYPES), 'output_contract': {'top_level': {'items': 'one item per input, same order'}, 'item': {'uuid': 'copy exactly', 'question_type': 'exactly one taxonomy value', 'operation': 'one operation_contract value', 'answer_type': 'one answer_type_contract value', 'target_entities': ['short open-vocabulary entity names'], 'attribute_dimension': 'requested attribute or empty string', 'target_state': 'requested state or empty string', 'spatial_constraints': [{'relation': 'open vocabulary', 'target': 'entity name'}], 'hard_constraints': [{'kind': 'constraint kind', 'text': 'normalized constraint'}], 'aggregation': 'single_best|count|all|yes_no', 'rendering': 'yes_no|number|location_phrase|short_phrase|free_text', 'confidence': 'number from 0 to 1', 'rationale': 'one short sentence about the semantic parse'}, 'no_extra_text': True}, 'questions': [{'uuid': _text(row.get('question_id') or row.get('uuid')), 'question': _text(row.get('question'))} for row in rows]}

def build_api_payload(rows: list[dict[str, Any]], *, model: str, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {'model': model, 'temperature': temperature, 'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': 'You are the OpenEQA semantic IR compiler. Compile the raw question only. Do not answer it. Return exactly one JSON object with key items and no extra text.'}, {'role': 'user', 'content': json.dumps(build_prompt(rows), ensure_ascii=False)}]}

def _response_content(response: dict[str, Any]) -> str:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ValueError('IR response has no choices')
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get('message')
    message = message if isinstance(message, dict) else {}
    content = message.get('content')
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return ''.join((str(item.get('text') or '') for item in content if isinstance(item, dict))).strip()
    raise ValueError('IR response content is not text')

def parse_response_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    content = _response_content(response)
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or not isinstance(parsed.get('items'), list):
        raise ValueError('IR response must be an object with an items list')
    return [dict(item) for item in parsed['items'] if isinstance(item, dict)]

def align_response_items(items: list[dict[str, Any]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected_ids = [_text(row.get('question_id') or row.get('uuid')) for row in rows]
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        question_id = _text(item.get('uuid'))
        if not question_id:
            raise ValueError('IR response item has no uuid')
        if question_id in by_id:
            raise ValueError(f'IR response has duplicate uuid: {question_id}')
        by_id[question_id] = item
    expected_set = set(expected_ids)
    actual_set = set(by_id)
    if actual_set != expected_set:
        raise ValueError(f'IR response uuid coverage mismatch missing={sorted(expected_set - actual_set)} extra={sorted(actual_set - expected_set)}')
    return [by_id[question_id] for question_id in expected_ids]

def normalize_ir_item(raw: dict[str, Any], *, row: dict[str, Any], data_source: str) -> dict[str, Any]:
    question_id = _text(row.get('question_id') or row.get('uuid'))
    if _text(raw.get('uuid')) != question_id:
        raise ValueError(f'{question_id}: response uuid mismatch')
    question_type = _text(raw.get('question_type'))
    if question_type not in ROUTE_IDS:
        raise ValueError(f'{question_id}: invalid question_type={question_type!r}')
    operation = _text(raw.get('operation'))
    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f'{question_id}: invalid operation={operation!r}')
    answer_type = _text(raw.get('answer_type'))
    if answer_type not in ALLOWED_ANSWER_TYPES:
        raise ValueError(f'{question_id}: invalid answer_type={answer_type!r}')
    leaked = sorted(FORBIDDEN_IR_FIELDS & set(raw))
    if leaked:
        raise ValueError(f'{question_id}: forbidden IR fields={leaked}')
    try:
        confidence = float(raw.get('confidence', 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {'schema_version': IR_SCHEMA, 'dataset': 'OpenEQA', 'data_source': data_source, 'question_id': question_id, 'episode_history': _text(row.get('episode_history')), 'question_type': question_type, 'operation': operation, 'answer_type': answer_type, 'target_entities': _string_list(raw.get('target_entities')), 'attribute_dimension': _text(raw.get('attribute_dimension')), 'target_state': _text(raw.get('target_state')), 'spatial_constraints': _object_list(raw.get('spatial_constraints')), 'hard_constraints': _object_list(raw.get('hard_constraints')), 'aggregation': _text(raw.get('aggregation')) or 'single_best', 'rendering': _text(raw.get('rendering')) or 'short_phrase', 'confidence': max(0.0, min(1.0, confidence)), 'rationale': _text(raw.get('rationale')), 'compiler_inputs': ['raw_question'], 'uses_gold_answer': False, 'uses_graph_evidence': False, 'uses_visual_evidence': False, 'route_decision_is_persisted': False}

def compile_batch(rows: list[dict[str, Any]], *, config: Any, max_tokens: int, temperature: float, semantic_retries: int=2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    api_key = os.getenv(config.api_key_env_var)
    if not api_key:
        raise RuntimeError(f'Missing environment variable: {config.api_key_env_var}')
    payload = build_api_payload(rows, model=config.model, max_tokens=max_tokens, temperature=temperature)
    last_error: Exception | None = None
    items: list[dict[str, Any]] = []
    response_attempt = 0
    for response_attempt in range(max(0, semantic_retries) + 1):
        response = _openeqa_post_payload(payload, api_key=api_key, config=config)
        try:
            items = align_response_items(parse_response_items(response), rows)
            break
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if response_attempt >= max(0, semantic_retries):
                raise
    if not items:
        raise ValueError(f'IR response remained invalid: {last_error}')
    return (items, {'question_count': len(rows), 'response_item_count': len(items), 'model': config.model, 'semantic_response_attempts': response_attempt + 1})

def _counter(records: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    counts = Counter((_text((record.get('llm_question_ir') or {}).get(field)) or '<empty>' for record in records))
    return dict(sorted(counts.items()))

def build_output(*, args: argparse.Namespace, input_path: Path, model: str, records: list[dict[str, Any]], batch_logs: list[dict[str, Any]], status: str, error: str='') -> dict[str, Any]:
    records = sorted(records, key=lambda record: int(record.get('index') or 0))
    ids = [_text(record.get('question_id')) for record in records]
    return {'contract': OUTPUT_CONTRACT, 'semantic_ir_schema': IR_SCHEMA, 'taxonomy_contract': TAXONOMY_CONTRACT, 'target_routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, 'generated_at': datetime.now(timezone.utc).isoformat(), 'status': status, 'error': error, 'dataset': 'OpenEQA', 'data_source': args.data_source, 'model': model, 'temperature': args.temperature, 'batch_size': args.batch_size, 'open_eqa_json': str(input_path), 'question_count': len(records), 'compiler_input_fields': ['uuid', 'question'], 'official_question_category_provided': False, 'uses_gold_answer': False, 'uses_graph_evidence': False, 'uses_visual_evidence': False, 'persists_question_route': False, 'summary': {'question_type_counts': _counter(records, 'question_type'), 'operation_counts': _counter(records, 'operation'), 'duplicate_question_id_count': len(ids) - len(set(ids))}, 'batch_logs': batch_logs, 'records': records}

def write_outputs(json_path: Path, md_path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(json_path, payload)
    summary = payload.get('summary') or {}
    lines = ['# OpenEQA V5 Semantic IR', '', f"- status: `{payload.get('status')}`", f"- data source: `{payload.get('data_source')}`", f"- model: `{payload.get('model')}`", f"- questions: `{payload.get('question_count')}`", '- compiler inputs: `uuid`, `question`', '- official question category provided: `False`', '- gold/graph/visual evidence provided: `False`', '', '## Question Types', '']
    for question_type, count in (summary.get('question_type_counts') or {}).items():
        lines.append(f'- `{question_type}`: `{count}`')
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    args = build_parser().parse_args()
    if args.force and args.resume:
        raise SystemExit('--force and --resume cannot be combined')
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be at least 1')
    input_path = repo_path(args.open_eqa_json)
    output_json = repo_path(args.output_json)
    output_md = repo_path(args.output_md)
    if output_json.exists() and (not args.force) and (not args.resume):
        raise SystemExit(f'Output exists: {output_json}; use --resume or --force')
    rows = select_rows(load_open_eqa_rows(input_path), data_source=args.data_source, offset=args.offset, limit=args.limit, question_ids=args.question_id)
    if not rows:
        raise SystemExit('No OpenEQA questions selected')
    domain = load_domain_config(args.config)
    config = domain.graph_qa.model_copy(update={'enabled': True, 'temperature': args.temperature, 'max_tokens': args.max_tokens, **({'timeout_sec': args.timeout_sec} if args.timeout_sec is not None else {})})
    pin_api_key(config, args.api_key_index)
    existing: dict[str, Any] = {}
    if args.resume and output_json.exists():
        loaded = json.loads(output_json.read_text(encoding='utf-8-sig'))
        existing = loaded if isinstance(loaded, dict) else {}
        if existing.get('contract') != OUTPUT_CONTRACT:
            raise SystemExit('Cannot resume a different output contract')
        if existing.get('data_source') != args.data_source:
            raise SystemExit('Cannot resume a different data source')
    records = [record for record in existing.get('records', []) if isinstance(record, dict)]
    batch_logs = [log for log in existing.get('batch_logs', []) if isinstance(log, dict)]
    done = {_text(record.get('question_id')) for record in records}
    pending = [(index, row) for index, row in enumerate(rows, start=1) if _text(row.get('question_id') or row.get('uuid')) not in done]
    print(f'data_source={args.data_source} selected={len(rows)} completed={len(done)} pending={len(pending)} model={config.model}', flush=True)
    try:
        for start in range(0, len(pending), args.batch_size):
            indexed_batch = pending[start:start + args.batch_size]
            batch_rows = [row for _, row in indexed_batch]
            raw_items, batch_log = compile_batch(batch_rows, config=config, max_tokens=args.max_tokens, temperature=args.temperature, semantic_retries=args.semantic_retries)
            for (index, row), raw in zip(indexed_batch, raw_items, strict=True):
                ir = normalize_ir_item(raw, row=row, data_source=args.data_source)
                records.append({'index': index, 'uuid': ir['question_id'], 'question_id': ir['question_id'], 'question': _text(row.get('question')), 'episode_history': ir['episode_history'], 'data_source': args.data_source, 'llm_question_ir': ir})
            batch_log['completed_question_count'] = len(records)
            batch_logs.append(batch_log)
            partial = build_output(args=args, input_path=input_path, model=config.model, records=records, batch_logs=batch_logs, status='partial')
            write_outputs(output_json, output_md, partial)
            print(f'completed={len(records)}/{len(rows)}', flush=True)
    except Exception as exc:
        failed = build_output(args=args, input_path=input_path, model=config.model, records=records, batch_logs=batch_logs, status=f'failed:{type(exc).__name__}', error=str(exc))
        write_outputs(output_json, output_md, failed)
        raise
    complete = build_output(args=args, input_path=input_path, model=config.model, records=records, batch_logs=batch_logs, status='complete')
    write_outputs(output_json, output_md, complete)
    print(json.dumps(complete['summary'], ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    main()
