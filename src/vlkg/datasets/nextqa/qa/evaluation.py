"""Evaluator-only exact-choice metrics for NExT-QA graph QA."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
TYPE_GROUPS = {'CH': 'causal', 'CW': 'causal', 'TC': 'temporal', 'TN': 'temporal', 'TP': 'temporal', 'DC': 'descriptive', 'DL': 'descriptive', 'DO': 'descriptive'}

def _unique_index(rows: Sequence[Mapping[str, Any]], *, name: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_id = str(row.get('question_id') or '')
        if not question_id:
            raise ValueError(f'{name} contains an empty question_id')
        if question_id in result:
            raise ValueError(f'{name} contains duplicate question_id: {question_id}')
        result[question_id] = row
    return result

def _provisional_error_category(prediction: Mapping[str, Any], *, official_type: str) -> str:
    if prediction.get('status') != 'completed':
        if prediction.get('error_stage') == 'graph_retrieval' or prediction.get('graph_rag_status') == 'graph_only_retrieval_error':
            return 'candidate_recall_gap'
        return 'QA_generation_or_parse_gap'
    if not prediction.get('graph_record_ids'):
        return 'candidate_recall_gap'
    if prediction.get('evidence_sufficiency') == 'insufficient':
        return 'candidate_recall_gap'
    if official_type in {'TN', 'TP', 'TC'}:
        return 'temporal_order_ranking_gap'
    if official_type in {'CH', 'CW'}:
        return 'causal_reasoning_gap'
    return 'answer_choice_disambiguation_gap'

def evaluate_exact_choice_accuracy(predictions: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], runtime_questions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    prediction_by_id = _unique_index(predictions, name='predictions')
    label_by_id = _unique_index(labels, name='evaluator labels')
    runtime_by_id = _unique_index(runtime_questions, name='runtime questions')
    if set(prediction_by_id) != set(label_by_id) or set(prediction_by_id) != set(runtime_by_id):
        raise ValueError('prediction, label, and runtime question ID sets must match exactly')
    type_counts: Counter[str] = Counter()
    type_correct: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    group_correct: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    attribution: Counter[str] = Counter()
    question_results: list[dict[str, Any]] = []
    correct_count = 0
    completed_count = 0
    error_count = 0
    videos_with_completed: set[str] = set()
    all_videos: set[str] = set()
    evidence_present_count = 0
    retrieved_evidence_present_wrong_reasoning_count = 0
    for question_id in sorted(prediction_by_id):
        prediction = prediction_by_id[question_id]
        label = label_by_id[question_id]
        runtime = runtime_by_id[question_id]
        official_type = str(runtime.get('official_type') or '')
        if official_type not in TYPE_GROUPS:
            raise ValueError(f'unmapped official question type: {official_type!r}')
        group = TYPE_GROUPS[official_type]
        gold = label.get('gold_choice_index')
        if isinstance(gold, bool) or not isinstance(gold, int) or gold < 0 or (gold > 4):
            raise ValueError(f'invalid evaluator gold index: {question_id}')
        choices = runtime.get('answer_choices')
        if not isinstance(choices, list) or len(choices) != 5:
            raise ValueError(f'runtime question does not have five choices: {question_id}')
        status = str(prediction.get('status') or 'error')
        predicted = prediction.get('predicted_choice_index')
        valid_prediction = status == 'completed' and isinstance(predicted, int) and (not isinstance(predicted, bool)) and (0 <= predicted <= 4)
        if valid_prediction:
            completed_count += 1
            videos_with_completed.add(str(runtime.get('video_id')))
            canonical_text = str(choices[predicted])
            if prediction.get('predicted_choice_text') != canonical_text:
                raise ValueError(f'prediction choice text/index mismatch: {question_id}')
            confusion[str(gold)][str(predicted)] += 1
        else:
            error_count += 1
            predicted = None
            confusion[str(gold)]['error'] += 1
        correct = bool(valid_prediction and predicted == gold)
        correct_count += int(correct)
        type_counts[official_type] += 1
        type_correct[official_type] += int(correct)
        group_counts[group] += 1
        group_correct[group] += int(correct)
        all_videos.add(str(runtime.get('video_id')))
        evidence_present = bool(prediction.get('graph_record_ids'))
        evidence_present_count += int(evidence_present)
        category = None
        if not correct:
            category = _provisional_error_category(prediction, official_type=official_type)
            attribution[category] += 1
            if valid_prediction and evidence_present and (prediction.get('evidence_sufficiency') != 'insufficient'):
                retrieved_evidence_present_wrong_reasoning_count += 1
        question_results.append({'question_id': question_id, 'video_id': str(runtime.get('video_id')), 'official_type': official_type, 'type_group': group, 'status': status, 'predicted_choice_index': predicted, 'gold_choice_index': gold, 'correct': correct, 'provisional_error_category': category})
    total = len(prediction_by_id)
    return {'schema_version': 'nextqa_exact_choice_evaluation_v1', 'question_count': total, 'completed_question_count': completed_count, 'error_question_count': error_count, 'correct_question_count': correct_count, 'overall_accuracy': correct_count / total if total else 0.0, 'completed_only_accuracy': correct_count / completed_count if completed_count else 0.0, 'official_type_accuracy': {type_code: {'correct': type_correct[type_code], 'count': type_counts[type_code], 'accuracy': type_correct[type_code] / type_counts[type_code]} for type_code in sorted(type_counts)}, 'question_group_accuracy': {group: {'correct': group_correct[group], 'count': group_counts[group], 'accuracy': group_correct[group] / group_counts[group]} for group in sorted(group_counts)}, 'video_count': len(all_videos), 'video_coverage': len(videos_with_completed) / len(all_videos) if all_videos else 0.0, 'confusion_by_choice_index': {gold: dict(sorted(row.items())) for gold, row in sorted(confusion.items())}, 'graph_evidence_coverage': evidence_present_count / total if total else 0.0, 'provisional_error_attribution': dict(sorted(attribution.items())), 'retrieved_evidence_present_but_wrong_reasoning_count': retrieved_evidence_present_wrong_reasoning_count, 'graph_signal_present_but_retrieval_missed_count': None, 'hsgg_or_graph_coverage_gap_count': None, 'diagnostic_limitations': ['Provisional error categories use prediction status, evidence sufficiency, and official type only.', 'Determining graph-signal-present/retrieval-missed and HSGG/graph gaps requires a separate full-graph evidence audit.'], 'question_results': question_results}
