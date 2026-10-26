"""Freeze the route-neutral graph evidence used by the visual mainline."""
from __future__ import annotations
import hashlib
import json
from typing import Any, Iterable, Mapping

def graph_record_ids_sha256(record_ids: Iterable[str]) -> str:
    encoded = json.dumps([str(value) for value in record_ids], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def compact_graph_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    question_id = str(row.get('question_id') or '').strip()
    video_id = str(row.get('video_id') or '').strip()
    if not question_id or not video_id:
        raise ValueError('retrieval record is missing question_id or video_id')
    status = str(row.get('status') or 'error')
    record_ids = [str(value) for value in row.get('graph_record_ids', [])]
    evidence: list[dict[str, Any]] = []
    for item in row.get('evidence', []):
        if not isinstance(item, Mapping):
            continue
        evidence.append({'rank': int(item.get('rank') or len(evidence) + 1), 'source_record_id': str(item.get('source_record_id') or ''), 'source_kind': str(item.get('source_kind') or ''), 'evidence_text': str(item.get('evidence_text') or ''), 'entity_ids': [str(value) for value in item.get('entity_ids', [])], 'time_range': dict(item.get('time_range') or {}), 'score': float(item.get('score') or 0.0)})
    if status == 'completed' and record_ids != [item['source_record_id'] for item in evidence]:
        raise ValueError(f'graph evidence ID order mismatch: {question_id}')
    return {'schema_version': 'nextqa_frozen_graph_evidence_record_v1', 'question_id': question_id, 'video_id': video_id, 'status': status, 'graph_retrieval_profile': str(row.get('graph_retrieval_profile') or ''), 'graph_record_ids': record_ids, 'graph_record_ids_sha256': graph_record_ids_sha256(record_ids), 'evidence': evidence, 'error_type': row.get('error_type'), 'error_message': row.get('error_message'), 'uses_gold_answer': False, 'uses_official_type': False, 'uses_visual_index': False}

def freeze_graph_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frozen = [compact_graph_evidence(row) for row in rows]
    question_ids = [str(row['question_id']) for row in frozen]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError('retrieval records contain duplicate question IDs')
    return frozen

def compare_prediction_record_ids(frozen: Iterable[Mapping[str, Any]], predictions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    prediction_by_id = {str(row.get('question_id')): row for row in predictions}
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    matched = 0
    error_aligned = 0
    frozen_rows = list(frozen)
    for row in frozen_rows:
        question_id = str(row['question_id'])
        prediction = prediction_by_id.get(question_id)
        if prediction is None:
            missing.append(question_id)
            continue
        if str(row.get('status')) != 'completed':
            if str(prediction.get('status')) == 'error':
                error_aligned += 1
            else:
                mismatches.append({'question_id': question_id, 'reason': 'retrieval_error_but_prediction_not_error'})
            continue
        expected = [str(value) for value in row.get('graph_record_ids', [])]
        actual = [str(value) for value in prediction.get('graph_record_ids', [])]
        if expected == actual:
            matched += 1
        else:
            mismatches.append({'question_id': question_id, 'reason': 'graph_record_ids_mismatch', 'expected_sha256': graph_record_ids_sha256(expected), 'actual_sha256': graph_record_ids_sha256(actual)})
    unexpected = sorted(set(prediction_by_id) - {str(row['question_id']) for row in frozen_rows})
    return {'completed_record_id_matches': matched, 'retrieval_error_prediction_error_matches': error_aligned, 'mismatch_count': len(mismatches), 'mismatches': mismatches, 'missing_prediction_question_ids': sorted(missing), 'unexpected_prediction_question_ids': unexpected, 'passed': not mismatches and (not missing) and (not unexpected)}
