"""Paired NExT-QA diagnostics across question IR, retrieval, and graph QA."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
IR_SEQUENCE_FIELDS = ('target_entities', 'actors', 'actions', 'objects')
IR_SCALAR_FIELDS = ('normalized_question', 'temporal_direction', 'temporal_anchor', 'referenced_event', 'causal_query', 'count_query', 'location_query', 'attribute_query', 'answer_format')
IR_COMPARE_FIELDS = IR_SEQUENCE_FIELDS + IR_SCALAR_FIELDS

def _index(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id:
            raise ValueError(f'{name} contains an empty question_id')
        if question_id in indexed:
            raise ValueError(f'{name} contains duplicate question_id: {question_id}')
        indexed[question_id] = row
    return indexed

def _normalized_text(value: Any) -> str:
    return ' '.join(str(value or '').strip().casefold().split())

def _normalized_ir_value(field: str, value: Any) -> Any:
    if field in IR_SEQUENCE_FIELDS:
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(sorted({_normalized_text(item) for item in value if _normalized_text(item)}))
    if field in {'count_query', 'location_query'}:
        return bool(value)
    return _normalized_text(value)

def _ir_delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    changed_fields = [field for field in IR_COMPARE_FIELDS if _normalized_ir_value(field, old.get(field)) != _normalized_ir_value(field, new.get(field))]
    additions: dict[str, list[str]] = {}
    removals: dict[str, list[str]] = {}
    for field in IR_SEQUENCE_FIELDS:
        old_values = set(_normalized_ir_value(field, old.get(field)))
        new_values = set(_normalized_ir_value(field, new.get(field)))
        if new_values - old_values:
            additions[field] = sorted(new_values - old_values)
        if old_values - new_values:
            removals[field] = sorted(old_values - new_values)
    return {'changed': bool(changed_fields), 'changed_fields': changed_fields, 'term_additions': additions, 'term_removals': removals, 'normalized_question_changed': 'normalized_question' in changed_fields}

def _evidence_payload(retrieval: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    evidence = retrieval.get('evidence')
    if not isinstance(evidence, list):
        return result
    for row in evidence:
        if not isinstance(row, Mapping):
            continue
        time_range = row.get('time_range')
        if not isinstance(time_range, Mapping):
            time_range = {}
        result.append({'rank': row.get('rank'), 'source_record_id': str(row.get('source_record_id') or ''), 'source_kind': str(row.get('source_kind') or ''), 'start_ts': time_range.get('start_ts'), 'end_ts': time_range.get('end_ts'), 'evidence_text': str(row.get('evidence_text') or '')})
    return result

def _retrieval_delta(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
    old_ids = [str(value) for value in old.get('graph_record_ids') or []]
    new_ids = [str(value) for value in new.get('graph_record_ids') or []]
    old_set = set(old_ids)
    new_set = set(new_ids)
    union = old_set | new_set
    intersection = old_set & new_set
    old_payload = _evidence_payload(old)
    new_payload = _evidence_payload(new)
    return {'status_changed': old.get('status') != new.get('status'), 'ordered_record_ids_equal': old_ids == new_ids, 'record_id_sets_equal': old_set == new_set, 'evidence_payload_equal': old_payload == new_payload, 'old_evidence_count': len(old_ids), 'new_evidence_count': len(new_ids), 'record_id_overlap_count': len(intersection), 'record_id_jaccard': len(intersection) / len(union) if union else 1.0, 'top5_overlap_count': len(set(old_ids[:5]) & set(new_ids[:5]))}

def _paired_outcome(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    old_correct = bool(old.get('correct'))
    new_correct = bool(new.get('correct'))
    if not old_correct and new_correct:
        return 'improved'
    if old_correct and (not new_correct):
        return 'regressed'
    if old_correct:
        return 'unchanged_correct'
    return 'unchanged_incorrect'

def _mechanism(*, answer_changed: bool, request_hash_equal: bool, ir_delta: Mapping[str, Any], retrieval_delta: Mapping[str, Any], old_status: str, new_status: str) -> str:
    if old_status != new_status:
        return 'status_changed'
    if not answer_changed:
        if request_hash_equal:
            return 'same_request_same_answer'
        if retrieval_delta['evidence_payload_equal']:
            return 'changed_request_same_evidence_same_answer'
        return 'changed_evidence_same_answer'
    if request_hash_equal:
        return 'same_request_model_variance'
    if not retrieval_delta['evidence_payload_equal']:
        if ir_delta['changed']:
            return 'ir_and_retrieval_changed'
        return 'retrieval_changed_without_ir_change'
    if ir_delta['normalized_question_changed']:
        return 'question_text_changed_same_evidence'
    if ir_delta['changed']:
        return 'semantic_ir_changed_same_evidence'
    return 'changed_request_without_ir_or_evidence_change'

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0

def compare_paired_runs(*, old_ir: Sequence[Mapping[str, Any]], new_ir: Sequence[Mapping[str, Any]], old_retrieval: Sequence[Mapping[str, Any]], new_retrieval: Sequence[Mapping[str, Any]], old_predictions: Sequence[Mapping[str, Any]], new_predictions: Sequence[Mapping[str, Any]], old_evaluation: Sequence[Mapping[str, Any]], new_evaluation: Sequence[Mapping[str, Any]], repaired_video_ids: set[str] | None=None) -> dict[str, Any]:
    sources = {'old IR': _index(old_ir, name='old IR'), 'new IR': _index(new_ir, name='new IR'), 'old retrieval': _index(old_retrieval, name='old retrieval'), 'new retrieval': _index(new_retrieval, name='new retrieval'), 'old predictions': _index(old_predictions, name='old predictions'), 'new predictions': _index(new_predictions, name='new predictions'), 'old evaluation': _index(old_evaluation, name='old evaluation'), 'new evaluation': _index(new_evaluation, name='new evaluation')}
    question_ids = set(next(iter(sources.values())))
    for name, indexed in sources.items():
        if set(indexed) != question_ids:
            raise ValueError(f'paired question ID sets do not align: {name}')
    repaired_video_ids = repaired_video_ids or set()
    rows: list[dict[str, Any]] = []
    for question_id in sorted(question_ids):
        old_ir_row = sources['old IR'][question_id]
        new_ir_row = sources['new IR'][question_id]
        old_retrieval_row = sources['old retrieval'][question_id]
        new_retrieval_row = sources['new retrieval'][question_id]
        old_prediction = sources['old predictions'][question_id]
        new_prediction = sources['new predictions'][question_id]
        old_result = sources['old evaluation'][question_id]
        new_result = sources['new evaluation'][question_id]
        if old_result.get('gold_choice_index') != new_result.get('gold_choice_index'):
            raise ValueError(f'paired gold labels do not align: {question_id}')
        ir_change = _ir_delta(old_ir_row, new_ir_row)
        retrieval_change = _retrieval_delta(old_retrieval_row, new_retrieval_row)
        old_request_hash = str(old_prediction.get('request_sha256') or '')
        new_request_hash = str(new_prediction.get('request_sha256') or '')
        request_hash_equal = bool(old_request_hash and new_request_hash and (old_request_hash == new_request_hash))
        answer_changed = old_result.get('predicted_choice_index') != new_result.get('predicted_choice_index')
        old_status = str(old_result.get('status') or 'error')
        new_status = str(new_result.get('status') or 'error')
        outcome = _paired_outcome(old_result, new_result)
        mechanism = _mechanism(answer_changed=answer_changed, request_hash_equal=request_hash_equal, ir_delta=ir_change, retrieval_delta=retrieval_change, old_status=old_status, new_status=new_status)
        video_id = str(new_result.get('video_id') or new_ir_row.get('video_id') or '')
        candidate_ir_effect_path = 'none'
        if outcome in {'improved', 'regressed'}:
            if mechanism == 'ir_and_retrieval_changed':
                candidate_ir_effect_path = 'direct_prompt_and_retrieval'
            elif mechanism in {'semantic_ir_changed_same_evidence', 'question_text_changed_same_evidence'}:
                candidate_ir_effect_path = 'direct_prompt_only'
        candidate_ir_effect = candidate_ir_effect_path != 'none'
        rows.append({'question_id': question_id, 'video_id': video_id, 'video_cohort': 'repaired_graph_29' if video_id in repaired_video_ids else 'unchanged_graph_541', 'official_type': str(new_result.get('official_type') or ''), 'outcome': outcome, 'old_correct': bool(old_result.get('correct')), 'new_correct': bool(new_result.get('correct')), 'old_status': old_status, 'new_status': new_status, 'old_predicted_choice_index': old_result.get('predicted_choice_index'), 'new_predicted_choice_index': new_result.get('predicted_choice_index'), 'gold_choice_index': new_result.get('gold_choice_index'), 'answer_changed': answer_changed, 'request_hash_equal': request_hash_equal, 'ir': ir_change, 'retrieval': retrieval_change, 'mechanism': mechanism, 'candidate_ir_effect': candidate_ir_effect, 'candidate_ir_effect_path': candidate_ir_effect_path, 'old_evidence_sufficiency': old_prediction.get('evidence_sufficiency'), 'new_evidence_sufficiency': new_prediction.get('evidence_sufficiency')})
    outcome_counts = Counter((row['outcome'] for row in rows))
    mechanism_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    changed_fields_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        mechanism_by_outcome[row['outcome']][row['mechanism']] += 1
        for field in row['ir']['changed_fields']:
            changed_fields_by_outcome[row['outcome']][field] += 1

    def summarize_subset(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(subset)
        old_correct = sum((bool(row['old_correct']) for row in subset))
        new_correct = sum((bool(row['new_correct']) for row in subset))
        return {'count': count, 'old_correct': old_correct, 'new_correct': new_correct, 'old_accuracy': old_correct / count if count else 0.0, 'new_accuracy': new_correct / count if count else 0.0, 'accuracy_delta': (new_correct - old_correct) / count if count else 0.0, 'improved': sum((row['outcome'] == 'improved' for row in subset)), 'regressed': sum((row['outcome'] == 'regressed' for row in subset)), 'regressed_with_new_incomplete_evidence': sum((row['outcome'] == 'regressed' and row['new_evidence_sufficiency'] != 'sufficient' for row in subset)), 'regressed_with_new_sufficient_evidence': sum((row['outcome'] == 'regressed' and row['new_evidence_sufficiency'] == 'sufficient' for row in subset)), 'answer_changed': sum((bool(row['answer_changed']) for row in subset)), 'ir_changed': sum((bool(row['ir']['changed']) for row in subset)), 'evidence_payload_changed': sum((not bool(row['retrieval']['evidence_payload_equal']) for row in subset)), 'same_request_answer_changed': sum((bool(row['request_hash_equal'] and row['answer_changed']) for row in subset)), 'same_request_score_flip': sum((bool(row['request_hash_equal'] and row['outcome'] in {'improved', 'regressed'}) for row in subset)), 'candidate_ir_effect': sum((bool(row['candidate_ir_effect']) for row in subset)), 'candidate_ir_direct_prompt_only': sum((row['candidate_ir_effect_path'] == 'direct_prompt_only' for row in subset)), 'candidate_ir_direct_prompt_and_retrieval': sum((row['candidate_ir_effect_path'] == 'direct_prompt_and_retrieval' for row in subset)), 'mean_evidence_record_jaccard': _mean([float(row['retrieval']['record_id_jaccard']) for row in subset])}
    type_codes = sorted({str(row['official_type']) for row in rows})
    cohorts = sorted({str(row['video_cohort']) for row in rows})
    summary = {'schema_version': 'nextqa_ir_retrieval_qa_paired_diagnostic_v1', 'question_count': len(rows), 'paired_outcomes': dict(sorted(outcome_counts.items())), 'overall': summarize_subset(rows), 'by_official_type': {type_code: summarize_subset([row for row in rows if row['official_type'] == type_code]) for type_code in type_codes}, 'by_video_cohort': {cohort: summarize_subset([row for row in rows if row['video_cohort'] == cohort]) for cohort in cohorts}, 'mechanism_by_outcome': {outcome: dict(sorted(counter.items())) for outcome, counter in sorted(mechanism_by_outcome.items())}, 'ir_changed_fields_by_outcome': {outcome: dict(sorted(counter.items())) for outcome, counter in sorted(changed_fields_by_outcome.items())}, 'interpretation_limits': ['candidate_ir_effect means the persisted QA request changed through direct IR prompt fields and optionally retrieval evidence; it is association, not proof of causality.', 'same_request_model_variance isolates answer changes where the persisted QA request hash is identical.', 'The repaired_graph_29 cohort is reported separately because its canonical graph changed between mainline stages.']}
    return {'summary': summary, 'question_results': rows}
