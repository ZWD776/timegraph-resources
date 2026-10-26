"""Offline error attribution for NExT-QA graph-only QA.

This module is evaluator-side diagnostics.  It reads completed predictions and
labels, then compares symmetric choice signals in the complete temporal graph,
the complete canonical graph, the normalized HSGG records, and the fixed
retrieval evidence.  It never reads RGB data, visual indexes, or calls an API.
Lexical evidence is deliberately reported as a diagnostic signal rather than
as a claim that the video contains the answer.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
ERROR_CATEGORIES = ('raw_hsgg_coverage_gap', 'temporal_graph_construction_gap', 'entity_or_event_binding_gap', 'candidate_recall_gap', 'temporal_order_ranking_gap', 'causal_reasoning_gap', 'answer_choice_disambiguation_gap', 'qa_generation_or_parse_gap')
STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'did', 'do', 'does', 'for', 'from', 'he', 'her', 'him', 'his', 'how', 'in', 'is', 'it', 'of', 'on', 'or', 'the', 'their', 'them', 'they', 'to', 'was', 'what', 'when', 'where', 'which', 'who', 'why', 'with', 'while', 'after', 'before', 'during', 'does', 'did', 'there', 'this', 'that', 'then', 'were'}
TOKEN_RE = re.compile('[a-z0-9]+')

def _tokens(text: Any) -> set[str]:
    result: set[str] = set()
    for token in TOKEN_RE.findall(str(text or '').lower()):
        if token in STOP_WORDS:
            continue
        result.add(token)
        result.add(_stem(token))
    return result

def _stem(token: str) -> str:
    if token in {'men', 'women', 'children', 'people', 'persons'}:
        return {'men': 'man', 'women': 'woman', 'children': 'child', 'people': 'person', 'persons': 'person'}[token]
    if len(token) > 5 and token.endswith('ing'):
        root = token[:-3]
        return root[:-1] if len(root) > 2 and root[-1] == root[-2] else root
    if len(token) > 4 and token.endswith('ied'):
        return token[:-3] + 'y'
    if len(token) > 4 and token.endswith('ed'):
        return token[:-2]
    if len(token) > 4 and token.endswith('es'):
        return token[:-2]
    if len(token) > 3 and token.endswith('s'):
        return token[:-1]
    return token

def _flatten(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [' '.join(value.split())]
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, child in value.items():
            result.append(str(key))
            result.extend(_flatten(child))
        return result
    if isinstance(value, Iterable) and (not isinstance(value, (bytes, bytearray))):
        result: list[str] = []
        for child in value:
            result.extend(_flatten(child))
        return result
    return [str(value)]

def _record_text(record: Mapping[str, Any]) -> str:
    explicit_text = record.get('text')
    if explicit_text is None:
        explicit_text = record.get('evidence_text')
    if explicit_text is not None:
        return ' '.join(str(explicit_text).split()).strip()
    values: list[str] = []
    for key, value in record.items():
        key_text = str(key).lower()
        if key_text.endswith('path') or key_text.endswith('_path'):
            continue
        if key_text in {'embedding', 'vector', 'image', 'box', 'boxes', 'scores'}:
            continue
        values.extend(_flatten(value))
    return ' '.join((value for value in values if value)).strip()

def _canonical_records(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ('entity_nodes', 'episode_nodes', 'transition_event_nodes', 'scene_nodes', 'state_value_nodes', 'stable_identity_nodes'):
        values = graph.get(key, [])
        if not isinstance(values, list):
            continue
        for row in values:
            if isinstance(row, Mapping):
                record_id = str(row.get('node_id') or row.get('event_id') or row.get('episode_id') or '')
                records.append({'record_id': record_id, 'source_kind': key[:-6] if key.endswith('_nodes') else key, 'text': _record_text(row), 'time_range': row.get('time_range') or {}})
    return records

def _temporal_records(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in ('entities', 'observations', 'relation_episodes', 'state_episodes', 'transition_events', 'scene_segments'):
        values = graph.get(key, [])
        if not isinstance(values, list):
            continue
        for index, row in enumerate(values):
            if isinstance(row, Mapping):
                record_id = str(row.get('entity_id') or row.get('relation_episode_id') or row.get('state_episode_id') or row.get('event_id') or f'{key}:{index}')
                records.append({'record_id': f'temporal:{record_id}', 'source_kind': key, 'text': _record_text(row), 'time_range': row})
    return records

def _hsgg_records(input_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename in ('objects.json', 'attributes.json', 'relations.json'):
        path = input_dir / filename
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, Mapping):
            continue
        for frame_name, row in payload.items():
            records.append({'record_id': f'hsgg:{filename}:{frame_name}', 'source_kind': f'hsgg_{filename[:-5]}', 'text': _record_text(row if isinstance(row, Mapping) else {'value': row}), 'time_range': {'frame_name': str(frame_name)}})
    return records

def _choice_signal(choice: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    query_tokens = _tokens(choice)
    hits: list[dict[str, Any]] = []
    for record in records:
        cached_tokens = record.get('_audit_tokens')
        record_tokens = cached_tokens if isinstance(cached_tokens, set) else _tokens(record.get('text'))
        overlap = sorted(query_tokens & record_tokens)
        if overlap:
            hits.append({'record_id': str(record.get('record_id') or ''), 'source_kind': str(record.get('source_kind') or ''), 'matched_tokens': overlap, 'time_range': record.get('time_range') or {}})
    return {'token_count': len(query_tokens), 'hit_count': len(hits), 'hits': hits[:30]}

def _question_signal(question: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = _tokens(question)
    count = 0
    for record in records:
        cached_tokens = record.get('_audit_tokens')
        record_tokens = cached_tokens if isinstance(cached_tokens, set) else _tokens(record.get('text'))
        count += bool(tokens & record_tokens)
    return {'token_count': len(tokens), 'record_hit_count': count}

def _cache_record_tokens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in records:
        record['_audit_tokens'] = _tokens(record.get('text'))
    return records

def _choice_category(*, official_type: str, prediction: Mapping[str, Any], gold_index: int, full_canonical: Sequence[Mapping[str, Any]], full_hsgg: Sequence[Mapping[str, Any]], retrieved: Sequence[Mapping[str, Any]], choices: Sequence[str]) -> tuple[str, str]:
    canonical_signals = [_choice_signal(choice, full_canonical) for choice in choices]
    hsgg_signals = [_choice_signal(choice, full_hsgg) for choice in choices]
    retrieved_signals = [_choice_signal(choice, retrieved) for choice in choices]
    gold_c = canonical_signals[gold_index]['hit_count']
    gold_h = hsgg_signals[gold_index]['hit_count']
    gold_r = retrieved_signals[gold_index]['hit_count']
    pred_index = prediction.get('predicted_choice_index')
    if prediction.get('status') != 'completed':
        return ('qa_generation_or_parse_gap', 'prediction did not complete')
    if gold_c == 0 and gold_h == 0:
        if official_type in {'CH', 'CW'}:
            return ('causal_reasoning_gap', 'causal choice has no lexical signal in structured graph; this is a lexical diagnostic, not proof of missing visual evidence')
        return ('raw_hsgg_coverage_gap', 'gold choice has no lexical signal in normalized HSGG or canonical graph; this is a lexical diagnostic, not proof of missing visual evidence')
    if gold_c == 0 and gold_h > 0:
        return ('temporal_graph_construction_gap', 'gold choice signal is present in normalized HSGG but absent from canonical graph')
    if gold_c > 0 and gold_r == 0:
        return ('candidate_recall_gap', 'gold choice lexical signal is present in full canonical graph but absent from fixed retrieved evidence')
    if official_type in {'TN', 'TC', 'TP'}:
        return ('temporal_order_ranking_gap', 'gold signal reached retrieved evidence but temporal ordering remains unresolved')
    if official_type in {'CH', 'CW'}:
        return ('causal_reasoning_gap', 'gold signal reached retrieved evidence but causal relation or intention was misreasoned')
    if pred_index is not None and int(pred_index) != gold_index:
        pred_r = retrieved_signals[int(pred_index)]['hit_count']
        if pred_r >= gold_r:
            return ('answer_choice_disambiguation_gap', 'retrieved evidence supports competing choices at least as strongly as gold')
    return ('entity_or_event_binding_gap', 'signal exists but entity/event binding is not decisive')

def audit_prediction_rows(predictions: Sequence[Mapping[str, Any]], runtime_questions: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], retrieval_rows: Sequence[Mapping[str, Any]], *, graph_paths: Mapping[str, Path], input_dirs: Mapping[str, Path]) -> dict[str, Any]:
    by_q = {str(row['question_id']): row for row in runtime_questions}
    by_label = {str(row['question_id']): row for row in labels}
    by_retrieval = {str(row['question_id']): row for row in retrieval_rows}
    records_by_video: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for prediction in predictions:
        qid = str(prediction.get('question_id') or '')
        question = by_q[qid]
        label = by_label[qid]
        retrieval = by_retrieval[qid]
        video_id = str(question['video_id'])
        if video_id not in records_by_video:
            graph = json.loads(graph_paths[video_id].read_text(encoding='utf-8'))
            records_by_video[video_id] = (_cache_record_tokens(_canonical_records(graph)), _cache_record_tokens(_temporal_records(graph)), _cache_record_tokens(_hsgg_records(input_dirs[video_id])))
        canonical, temporal, hsgg = records_by_video[video_id]
        evidence = retrieval.get('evidence') if isinstance(retrieval.get('evidence'), list) else []
        choices = [str(choice) for choice in question.get('answer_choices', [])]
        gold_index = int(label['gold_choice_index'])
        category, rationale = _choice_category(official_type=str(question.get('official_type') or ''), prediction=prediction, gold_index=gold_index, full_canonical=canonical, full_hsgg=hsgg, retrieved=evidence, choices=choices)
        if not bool(prediction.get('predicted_choice_index') == gold_index):
            category_counts[category] += 1
        canonical_signals = [_choice_signal(choice, canonical) for choice in choices]
        hsgg_signals = [_choice_signal(choice, hsgg) for choice in choices]
        retrieved_signals = [_choice_signal(choice, evidence) for choice in choices]
        wrong = prediction.get('predicted_choice_index') != gold_index
        secondary: list[str] = []
        if wrong and category == 'candidate_recall_gap':
            if str(question.get('official_type') or '') in {'TN', 'TC', 'TP'}:
                secondary.append('temporal_order_ranking_gap')
            elif str(question.get('official_type') or '') in {'CH', 'CW'}:
                secondary.append('causal_reasoning_gap')
        if wrong and category in {'causal_reasoning_gap', 'temporal_order_ranking_gap'}:
            secondary.append('answer_choice_disambiguation_gap')
        rows.append({'question_id': qid, 'video_id': video_id, 'official_type': question.get('official_type'), 'normalized_question': question.get('normalized_question'), 'answer_choices': choices, 'predicted_choice_index': prediction.get('predicted_choice_index'), 'predicted_choice_text': prediction.get('predicted_choice_text'), 'gold_choice_index': gold_index, 'gold_choice_text': choices[gold_index], 'correct': prediction.get('predicted_choice_index') == gold_index, 'prediction_reasoning': prediction.get('reasoning'), 'used_evidence_ranks': prediction.get('used_evidence_ranks', []), 'evidence_sufficiency': prediction.get('evidence_sufficiency'), 'retrieved_graph_record_ids': list(prediction.get('graph_record_ids', [])), 'retrieved_evidence': [{key: row.get(key) for key in ('rank', 'source_record_id', 'source_kind', 'evidence_text', 'time_range')} for row in evidence], 'signal_audit': {'question': _question_signal(str(question.get('normalized_question') or ''), canonical), 'choice_signals_full_canonical': canonical_signals, 'choice_signals_full_temporal': [_choice_signal(choice, temporal) for choice in choices], 'choice_signals_normalized_hsgg': hsgg_signals, 'choice_signals_retrieved': retrieved_signals, 'gold_choice_full_canonical_hit': bool(canonical_signals[gold_index]['hit_count']), 'gold_choice_hsgg_hit': bool(hsgg_signals[gold_index]['hit_count']), 'gold_choice_retrieved_hit': bool(retrieved_signals[gold_index]['hit_count'])}, 'primary_category': category, 'category_rationale': rationale, 'secondary_categories': sorted(set(secondary)), 'diagnostic_basis': 'lexical_overlap_over_structured_records', 'purity': {'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_index': False, 'calls_api': False, 'uses_gold_for_retrieval': False}})
    return {'schema_version': 'nextqa_graph_only_error_audit_v1', 'question_count': len(rows), 'wrong_question_count': sum((not bool(row['correct']) for row in rows)), 'category_counts': dict(sorted(category_counts.items())), 'secondary_category_counts': dict(sorted(Counter((category for row in rows for category in row['secondary_categories'])).items())), 'graph_signal_present_but_retrieval_missed_count': sum((not row['correct'] and row['signal_audit']['gold_choice_full_canonical_hit'] and (not row['signal_audit']['gold_choice_retrieved_hit']) for row in rows)), 'hsgg_or_graph_coverage_gap_count': sum((not row['correct'] and (not row['signal_audit']['gold_choice_hsgg_hit']) for row in rows)), 'diagnostic_limitations': ['Signal checks use lexical overlap over structured HSGG/canonical/evidence text and do not establish that a choice is visually true.', 'A no-hit causal choice may reflect latent intent or paraphrase rather than raw HSGG absence.', 'Temporal order and entity binding require human review of the listed record IDs and time ranges.'], 'rows': rows, 'purity': {'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_index': False, 'calls_api': False, 'uses_gold_for_retrieval': False, 'uses_gold_for_graph_construction': False}}
