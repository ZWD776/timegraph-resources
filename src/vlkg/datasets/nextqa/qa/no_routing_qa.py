"""Text-only QA transport for the NExT-QA no-routing ablation."""
from __future__ import annotations
import hashlib
import json
import time
from typing import Any, Mapping, Sequence
from vlkg.datasets.nextqa.qa.graph_only_qa import GraphOnlyMCQConfig, OpenAICompatibleMCQClient, REQUEST_EXCEPTIONS, STRICT_IR_FIELDS, SUFFICIENCY_VALUES, build_chat_payload, request_sha256, response_metadata

def validate_no_routing_input(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, graph_limit: int=16, visual_limit: int=12) -> None:
    question_id = str(question.get('question_id') or '')
    if not question_id or any((str(row.get('question_id') or '') != question_id for row in (graph, visual))):
        raise ValueError('question, graph, and visual IDs do not align')
    if str(question.get('video_id') or '') != str(graph.get('video_id') or '') or str(question.get('video_id') or '') != str(visual.get('video_id') or ''):
        raise ValueError(f'video IDs do not align for {question_id}')
    choices = question.get('answer_choices')
    if not isinstance(choices, list) or len(choices) != 5 or any((not str(choice).strip() for choice in choices)):
        raise ValueError(f'question must contain exactly five choices: {question_id}')
    if question.get('uses_gold_answer') is not False:
        raise ValueError(f'question gold purity failed: {question_id}')
    if graph.get('uses_gold_answer') is not False or graph.get('uses_official_type') is not False:
        raise ValueError(f'graph purity failed: {question_id}')
    required_false = ('uses_gold_answer', 'uses_official_type', 'uses_answer_choices', 'qa_receives_images', 'qa_receives_image_paths', 'qa_receives_visual_embeddings')
    if any((visual.get(field) is not False for field in required_false)):
        raise ValueError(f'no-routing visual purity failed: {question_id}')
    if visual.get('ablation_label') != 'no_routing':
        raise ValueError(f'visual ablation label is not fixed no_routing: {question_id}')
    if 'route_id' in visual or 'handler_id' in visual:
        raise ValueError(f'no-routing visual record contains route fields: {question_id}')
    graph_ids = list(graph.get('graph_record_ids', []))
    if graph_ids != list(visual.get('graph_record_ids', [])):
        raise ValueError(f'frozen graph record IDs changed: {question_id}')
    graph_evidence = graph.get('evidence')
    visual_evidence = visual.get('visual_evidence')
    if not isinstance(graph_evidence, list) or len(graph_evidence) > graph_limit:
        raise ValueError(f'graph evidence budget failed: {question_id}')
    if not isinstance(visual_evidence, list) or not visual_evidence or len(visual_evidence) > visual_limit:
        raise ValueError(f'visual evidence budget failed: {question_id}')
    serialized = json.dumps(visual_evidence, ensure_ascii=False).lower()
    forbidden = ('image_path', 'embedding', '.png', '.jpg', '.jpeg', 'official_type', 'gold_choice', 'route_id', 'handler_id')
    if any((token in serialized for token in forbidden)):
        raise ValueError(f'visual evidence contains forbidden transport data: {question_id}')

def build_no_routing_messages(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, graph_limit: int=16, visual_limit: int=12) -> list[dict[str, str]]:
    validate_no_routing_input(question, graph, visual, graph_limit=graph_limit, visual_limit=visual_limit)
    graph_rows = [{'rank': int(row.get('rank', index)), 'record_id': str(row.get('source_record_id') or ''), 'source_kind': str(row.get('source_kind') or ''), 'time_range': row.get('time_range') or {}, 'text': str(row.get('evidence_text') or '').strip()} for index, row in enumerate(graph['evidence'][:graph_limit], start=1)]
    visual_rows = [{'rank': int(row.get('rank', index)), 'evidence_role': '', 'text': str(row.get('text') or '').strip()} for index, row in enumerate(visual['visual_evidence'][:visual_limit], start=1)]
    prompt = {'task': 'nextqa_graph_plus_retrieved_visual_text_multiple_choice', 'question': str(question.get('normalized_question') or '').strip(), 'question_ir': {field: question.get(field) for field in STRICT_IR_FIELDS}, 'answer_choices': [{'choice_index': index, 'choice_text': str(choice).strip()} for index, choice in enumerate(question['answer_choices'])], 'frozen_graph_evidence': graph_rows, 'retrieved_graph_bound_visual_evidence': visual_rows, 'policy': {'use_only_supplied_text_evidence': True, 'do_not_request_or_infer_from_unseen_images': True, 'do_not_use_answer_keys_or_question_type_labels': True, 'assess_all_five_choices_symmetrically': True, 'same_combined_evidence_budget_for_every_choice': len(graph_rows) + len(visual_rows), 'choice_indices_are_zero_based': True, 'return_one_choice_only': True, 'state_when_evidence_is_incomplete': True}, 'output_schema': {'choice_index': 'integer 0..4', 'reasoning': 'brief evidence-grounded explanation', 'used_graph_evidence_ranks': 'array of frozen graph evidence ranks', 'used_visual_evidence_ranks': 'array of retrieved visual evidence ranks', 'evidence_sufficiency': 'sufficient|partial|insufficient'}}
    return [{'role': 'system', 'content': 'Answer NExT-QA multiple-choice questions using only the supplied temporal graph evidence and graph-bound text rendered from offline visual retrieval. You receive no images, paths, crops, or visual vectors. Evaluate all five choices with the same evidence and compute budget. Return exactly one JSON object.'}, {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False, sort_keys=True)}]

def _parse_response(response: Mapping[str, Any], choices: Sequence[str], graph_evidence_count: int, visual_evidence_count: int) -> dict[str, Any]:
    rows = response.get('choices')
    if not isinstance(rows, list) or not rows or (not isinstance(rows[0], Mapping)):
        raise ValueError('response has no choices')
    message = rows[0].get('message')
    if not isinstance(message, Mapping):
        raise ValueError('response choice has no message')
    content = message.get('content')
    if isinstance(content, list):
        content = '\n'.join((str(item.get('text') or '') for item in content if isinstance(item, Mapping)))
    if not isinstance(content, str) or not content.strip():
        raise ValueError('response message content is empty')
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
        if text.lower().startswith('json'):
            text = text[4:].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError('model response must be one JSON object')
    index = parsed.get('choice_index')
    if isinstance(index, bool) or not isinstance(index, int) or (not 0 <= index < len(choices)):
        raise ValueError('choice_index must be an integer in range')
    ranks = parsed.get('used_graph_evidence_ranks', [])
    if not isinstance(ranks, list) or any((isinstance(value, bool) or not isinstance(value, int) or value < 1 or (value > graph_evidence_count) for value in ranks)):
        raise ValueError('used_graph_evidence_ranks is invalid')
    visual_ranks = parsed.get('used_visual_evidence_ranks', [])
    if not isinstance(visual_ranks, list) or any((isinstance(value, bool) or not isinstance(value, int) or value < 1 or (value > visual_evidence_count) for value in visual_ranks)):
        raise ValueError('used_visual_evidence_ranks is invalid')
    sufficiency = str(parsed.get('evidence_sufficiency') or 'unknown').strip().lower()
    if sufficiency not in SUFFICIENCY_VALUES:
        raise ValueError('invalid evidence_sufficiency value')
    return {'choice_index': index, 'choice_text': str(choices[index]), 'reasoning': str(parsed.get('reasoning') or '').strip(), 'used_graph_evidence_ranks': ranks, 'used_visual_evidence_ranks': visual_ranks, 'evidence_sufficiency': sufficiency}

def answer_no_routing_mcq(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, config: GraphOnlyMCQConfig, answer_mode: str, client: OpenAICompatibleMCQClient | None=None, api_key_slot: int=0) -> dict[str, Any]:
    messages = build_no_routing_messages(question, graph, visual)
    choices = [str(item) for item in question['answer_choices']]
    graph_count = len(graph.get('evidence', []))
    visual_count = len(visual['visual_evidence'])
    attempts = 0
    parsed = None
    raw_response: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    last_stage = 'qa_generation'
    live_client = client
    if answer_mode == 'mock':
        index = int(hashlib.sha256(str(question['question_id']).encode('utf-8')).hexdigest(), 16) % 5
        raw_response = {'id': 'nextqa-no-routing-mock', 'choices': [{'message': {'content': json.dumps({'choice_index': index, 'reasoning': 'Deterministic offline mock response.', 'used_graph_evidence_ranks': [1] if graph_count else [], 'used_visual_evidence_ranks': [1], 'evidence_sufficiency': 'unknown'})}}]}
        attempts = 1
        parsed = _parse_response(raw_response, choices, graph_count, visual_count)
    elif answer_mode == 'live':
        live_client = client or OpenAICompatibleMCQClient(config, api_key_slot=api_key_slot)
        payload = build_chat_payload(messages, config=config)
        for attempt_index in range(config.max_attempts):
            attempts += 1
            try:
                raw_response = live_client.post(payload, attempt=attempt_index)
                last_stage = 'qa_response_parse'
                parsed = _parse_response(raw_response, choices, graph_count, visual_count)
                break
            except REQUEST_EXCEPTIONS + (RuntimeError, ValueError, TypeError) as exc:
                last_error = exc
                if attempts < config.max_attempts:
                    seed = hashlib.sha256(f"{question['question_id']}:{attempt_index}".encode('utf-8')).digest()
                    delay = config.retry_backoff_sec * 2 ** attempt_index + int.from_bytes(seed[:4], 'big') / 2 ** 32 * config.retry_jitter_sec
                    if delay > 0:
                        time.sleep(delay)
    else:
        raise ValueError(f'unsupported answer mode: {answer_mode!r}')
    common = {'schema_version': 'nextqa_visual_no_routing_mcq_prediction_v2', 'question_id': str(question['question_id']), 'video_id': str(question['video_id']), 'model': config.model if answer_mode == 'live' else 'nextqa-no-routing-mock-v2', 'answer_mode': answer_mode, 'qa_profile': config.profile_name, 'ablation_label': 'no_routing', 'graph_retrieval_profile': str(graph.get('graph_retrieval_profile') or ''), 'graph_record_ids': list(graph.get('graph_record_ids', [])), 'graph_evidence_count': graph_count, 'visual_evidence_count': visual_count, 'visual_candidate_ids': list(visual.get('used_visual_candidate_ids', [])), 'evidence_budget_per_choice': [graph_count + visual_count] * 5, 'request_sha256': request_sha256(messages), 'attempt_count': attempts, 'api_key_slot': getattr(live_client, 'api_key_slot', api_key_slot) if answer_mode == 'live' else None, 'api_key_pool_size': getattr(live_client, 'api_key_pool_size', None) if answer_mode == 'live' else None, 'uses_gold_answer': False, 'uses_official_type': False, 'uses_answer_choices': False, 'offline_reads_visual_index': True, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'uses_visual_index': True}
    if parsed is None:
        return {**common, 'status': 'error', 'graph_rag_status': 'graph_plus_uniform_visual_mcq_error', 'predicted_choice_index': None, 'predicted_choice_text': None, 'error_stage': last_stage, 'error_type': type(last_error).__name__ if last_error else 'UnknownError', 'error_message': str(last_error or 'model response did not validate'), 'response_metadata': response_metadata(raw_response) if raw_response is not None else None}
    return {**common, 'status': 'completed', 'graph_rag_status': 'graph_plus_uniform_visual_mcq_completed', 'predicted_choice_index': parsed['choice_index'], 'predicted_choice_text': parsed['choice_text'], 'reasoning': parsed['reasoning'], 'used_graph_evidence_ranks': parsed['used_graph_evidence_ranks'], 'used_visual_evidence_ranks': parsed['used_visual_evidence_ranks'], 'evidence_sufficiency': parsed['evidence_sufficiency'], 'response_metadata': response_metadata(raw_response or {})}
