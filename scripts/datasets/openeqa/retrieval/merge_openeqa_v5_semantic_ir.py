"""Validate and merge complete OpenEQA V5 semantic IR shards."""
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
from scripts.datasets.openeqa.retrieval.run_openeqa_v5_semantic_ir import FORBIDDEN_IR_FIELDS, IR_SCHEMA, OUTPUT_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v5 import ROUTE_IDS, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, TAXONOMY_CONTRACT

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Merge complete, disjoint V5 semantic IR shards.')
    parser.add_argument('--open-eqa-json', type=Path, default=Path('datasets/openeqa/source/openeqa-em-eqa-hm3d-scannet-v1/payload/official/open-eqa-v0.json'))
    parser.add_argument('--data-source', choices=('hm3d-v0', 'scannet-v0'), required=True)
    parser.add_argument('--input-json', type=Path, action='append', required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    return parser

def _text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def merge_records(payloads: list[dict[str, Any]], *, dataset_rows: list[dict[str, Any]], data_source: str) -> list[dict[str, Any]]:
    source_rows = [row for row in dataset_rows if _text(row.get('episode_history')).partition('/')[0] == data_source]
    expected_ids = [_text(row.get('question_id') or row.get('uuid')) for row in source_rows]
    expected_set = set(expected_ids)
    by_id: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for payload in payloads:
        if payload.get('contract') != OUTPUT_CONTRACT:
            issues.append('input has the wrong semantic IR contract')
        if payload.get('semantic_ir_schema') != IR_SCHEMA:
            issues.append('input has the wrong semantic IR schema')
        if payload.get('taxonomy_contract') != TAXONOMY_CONTRACT:
            issues.append('input has the wrong taxonomy contract')
        if payload.get('data_source') != data_source:
            issues.append('input data source mismatch')
        if payload.get('status') != 'complete':
            issues.append('input semantic IR shard is not complete')
        for record in payload.get('records', []) or []:
            if not isinstance(record, dict):
                issues.append('input contains a non-object record')
                continue
            question_id = _text(record.get('question_id') or record.get('uuid'))
            if not question_id:
                issues.append('input contains a record without question_id')
                continue
            if question_id in by_id:
                issues.append(f'duplicate question_id across shards: {question_id}')
                continue
            ir = record.get('llm_question_ir')
            if not isinstance(ir, dict):
                issues.append(f'{question_id}: missing semantic IR')
                continue
            if ir.get('question_type') not in ROUTE_IDS:
                issues.append(f'{question_id}: invalid question_type')
            leaked = sorted(FORBIDDEN_IR_FIELDS & (set(record) | set(ir)))
            if leaked:
                issues.append(f'{question_id}: forbidden fields {leaked}')
            if ir.get('compiler_inputs') != ['raw_question']:
                issues.append(f'{question_id}: compiler input drift')
            if any((ir.get(field) is not False for field in ('uses_gold_answer', 'uses_graph_evidence', 'uses_visual_evidence', 'route_decision_is_persisted'))):
                issues.append(f'{question_id}: semantic IR provenance drift')
            by_id[question_id] = dict(record)
    actual_set = set(by_id)
    if actual_set != expected_set:
        issues.append(f'source coverage mismatch missing={len(expected_set - actual_set)} extra={len(actual_set - expected_set)}')
    if issues:
        raise ValueError('; '.join(issues[:20]))
    merged: list[dict[str, Any]] = []
    for index, question_id in enumerate(expected_ids, start=1):
        record = dict(by_id[question_id])
        record['index'] = index
        merged.append(record)
    return merged

def build_merged_payload(*, records: list[dict[str, Any]], data_source: str, input_path: Path, input_jsons: list[Path], model: str) -> dict[str, Any]:
    question_types = Counter((_text((record.get('llm_question_ir') or {}).get('question_type')) for record in records))
    operations = Counter((_text((record.get('llm_question_ir') or {}).get('operation')) for record in records))
    return {'contract': OUTPUT_CONTRACT, 'semantic_ir_schema': IR_SCHEMA, 'taxonomy_contract': TAXONOMY_CONTRACT, 'target_routing_profile': ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, 'generated_at': datetime.now(timezone.utc).isoformat(), 'status': 'complete', 'error': '', 'dataset': 'OpenEQA', 'data_source': data_source, 'model': model, 'temperature': 0.0, 'open_eqa_json': str(input_path), 'source_semantic_ir_shards': [str(path) for path in input_jsons], 'question_count': len(records), 'compiler_input_fields': ['uuid', 'question'], 'official_question_category_provided': False, 'uses_gold_answer': False, 'uses_graph_evidence': False, 'uses_visual_evidence': False, 'persists_question_route': False, 'summary': {'question_type_counts': dict(sorted(question_types.items())), 'operation_counts': dict(sorted(operations.items())), 'duplicate_question_id_count': 0}, 'batch_logs': [], 'records': records}

def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ['# OpenEQA V5 Semantic IR', '', '- status: `complete`', f"- data source: `{payload['data_source']}`", f"- model: `{payload['model']}`", f"- questions: `{payload['question_count']}`", '- compiler inputs: `uuid`, `question`', '- official question category provided: `False`', '- gold/graph/visual evidence provided: `False`', '', '## Question Types', '']
    for question_type, count in payload['summary']['question_type_counts'].items():
        lines.append(f'- `{question_type}`: `{count}`')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    args = build_parser().parse_args()
    input_path = repo_path(args.open_eqa_json)
    input_jsons = [repo_path(path) for path in args.input_json]
    payloads = [json.loads(path.read_text(encoding='utf-8-sig')) for path in input_jsons]
    records = merge_records(payloads, dataset_rows=load_open_eqa_rows(input_path), data_source=args.data_source)
    models = {_text(payload.get('model')) for payload in payloads}
    if len(models) != 1:
        raise ValueError(f'semantic IR shard model mismatch: {sorted(models)}')
    merged = build_merged_payload(records=records, data_source=args.data_source, input_path=input_path, input_jsons=input_jsons, model=next(iter(models)))
    output_json = repo_path(args.output_json)
    output_md = repo_path(args.output_md)
    write_json_atomic(output_json, merged)
    write_markdown(output_md, merged)
    print(json.dumps(merged['summary'], ensure_ascii=False, sort_keys=True))
if __name__ == '__main__':
    main()
