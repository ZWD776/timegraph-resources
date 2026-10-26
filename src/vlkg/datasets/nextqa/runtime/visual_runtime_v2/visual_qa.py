"""Text-only graph plus retrieved-visual-evidence MCQ answering for NExT-QA."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
import yaml
from vlkg.datasets.nextqa.qa.graph_only_qa import GraphOnlyMCQConfig, OpenAICompatibleMCQClient, REQUEST_EXCEPTIONS, STRICT_IR_FIELDS, SUFFICIENCY_VALUES, build_chat_payload, request_sha256, response_metadata

@dataclass(frozen=True)
class VisualMCQConfig:
    transport: GraphOnlyMCQConfig
    graph_evidence_limit: int
    visual_evidence_limit: int

def load_visual_mcq_config(path: str | Path) -> VisualMCQConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if payload.get('schema_version') != 'nextqa_visual_enhanced_mcq_config_v2':
        raise ValueError('unsupported NExT-QA visual QA config schema')
    qa = payload.get('qa')
    if not isinstance(qa, Mapping):
        raise ValueError('NExT-QA visual QA config requires a qa mapping')
    graph_limit = int(qa.get('graph_evidence_limit', 16))
    visual_limit = int(qa.get('visual_evidence_limit', 12))
    transport = GraphOnlyMCQConfig(profile_name=str(qa.get('profile_name') or '').strip(), provider=str(qa.get('provider') or '').strip(), model=str(qa.get('model') or '').strip(), api_base_url=str(qa.get('api_base_url') or '').strip(), api_base_url_env_var=str(qa['api_base_url_env_var']).strip() if qa.get('api_base_url_env_var') else None, endpoint_path=str(qa.get('endpoint_path') or '').strip(), api_key_env_var=str(qa.get('api_key_env_var') or '').strip(), timeout_sec=float(qa.get('timeout_sec', 300.0)), temperature=float(qa.get('temperature', 0.1)), max_tokens=int(qa.get('max_tokens', 600)), max_attempts=int(qa.get('max_attempts', 2)), evidence_limit=graph_limit, thinking_enabled=bool(qa.get('thinking_enabled', False)), api_key_pool_env_var=str(qa.get('api_key_pool_env_var') or 'OPENAI_API_KEYS').strip(), retry_backoff_sec=float(qa.get('retry_backoff_sec', 1.0)), retry_jitter_sec=float(qa.get('retry_jitter_sec', 0.25)))
    if transport.model != 'gpt-5.6-luna':
        raise ValueError('NExT-QA visual QA requires gpt-5.6-luna')
    if transport.provider != 'openai_compatible':
        raise ValueError('NExT-QA visual QA requires the openai_compatible provider')
    if not transport.profile_name.startswith('nextqa_visual_enhanced_mcq_'):
        raise ValueError('visual QA profile must be NExT-QA isolated')
    if min(graph_limit, visual_limit, transport.max_tokens, transport.max_attempts) < 1:
        raise ValueError('visual QA budgets must be positive')
    return VisualMCQConfig(transport, graph_limit, visual_limit)

def validate_visual_qa_inputs(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, config: VisualMCQConfig) -> None:
    question_id = str(question.get('question_id') or '')
    ids = {question_id, str(graph.get('question_id') or ''), str(visual.get('question_id') or '')}
    if not question_id or len(ids) != 1:
        raise ValueError('question, graph, and visual evidence IDs do not align')
    video_ids = {str(question.get('video_id') or ''), str(graph.get('video_id') or ''), str(visual.get('video_id') or '')}
    if '' in video_ids or len(video_ids) != 1:
        raise ValueError(f'video IDs do not align for {question_id}')
    choices = question.get('answer_choices')
    if not isinstance(choices, list) or len(choices) != 5 or any((not str(choice).strip() for choice in choices)):
        raise ValueError(f'question must contain five non-empty choices: {question_id}')
    if question.get('uses_gold_answer') is not False:
        raise ValueError(f'question runtime gold purity failed: {question_id}')
    for row, name in ((graph, 'graph'), (visual, 'visual')):
        if row.get('uses_gold_answer') is not False or row.get('uses_official_type') is not False:
            raise ValueError(f'{name} evidence purity failed: {question_id}')
    if visual.get('uses_answer_choices') is not False or visual.get('qa_receives_images') is not False:
        raise ValueError(f'visual evidence isolation failed: {question_id}')
    if visual.get('qa_receives_image_paths') is not False or visual.get('qa_receives_visual_embeddings') is not False:
        raise ValueError(f'visual QA transport boundary failed: {question_id}')
    if list(graph.get('graph_record_ids', [])) != list(visual.get('graph_record_ids', [])):
        raise ValueError(f'frozen graph record IDs changed: {question_id}')
    graph_evidence = graph.get('evidence')
    visual_evidence = visual.get('visual_evidence')
    if not isinstance(graph_evidence, list) or len(graph_evidence) > config.graph_evidence_limit:
        raise ValueError(f'graph evidence budget failed: {question_id}')
    if not isinstance(visual_evidence, list) or not visual_evidence or len(visual_evidence) > config.visual_evidence_limit:
        raise ValueError(f'visual evidence budget failed: {question_id}')
    serialized = json.dumps(visual_evidence, ensure_ascii=False).lower()
    forbidden = ('image_path', 'embedding', '.png', '.jpg', '.jpeg', 'official_type', 'gold_choice')
    if any((value in serialized for value in forbidden)):
        raise ValueError(f'visual QA evidence contains a forbidden field or path: {question_id}')

def build_visual_qa_messages(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, config: VisualMCQConfig) -> list[dict[str, str]]:
    validate_visual_qa_inputs(question, graph, visual, config=config)
    graph_rows = [{'rank': int(row.get('rank', index)), 'record_id': str(row.get('source_record_id') or ''), 'source_kind': str(row.get('source_kind') or ''), 'time_range': row.get('time_range') or {}, 'text': str(row.get('evidence_text') or '').strip()} for index, row in enumerate(graph.get('evidence', []), start=1)]
    visual_rows = [{'rank': int(row.get('rank', index)), 'evidence_role': str(row.get('role_id') or ''), 'text': str(row.get('text') or '').strip()} for index, row in enumerate(visual['visual_evidence'], start=1)]
    choices = [str(choice).strip() for choice in question['answer_choices']]
    prompt = {'task': 'nextqa_graph_plus_retrieved_visual_text_multiple_choice', 'question': str(question.get('normalized_question') or '').strip(), 'question_ir': {field: question.get(field) for field in STRICT_IR_FIELDS}, 'answer_choices': [{'choice_index': index, 'choice_text': choice} for index, choice in enumerate(choices)], 'frozen_graph_evidence': graph_rows, 'retrieved_graph_bound_visual_evidence': visual_rows, 'policy': {'use_only_supplied_text_evidence': True, 'do_not_request_or_infer_from_unseen_images': True, 'do_not_use_answer_keys_or_question_type_labels': True, 'assess_all_five_choices_symmetrically': True, 'same_combined_evidence_budget_for_every_choice': len(graph_rows) + len(visual_rows), 'choice_indices_are_zero_based': True, 'return_one_choice_only': True, 'state_when_evidence_is_incomplete': True}, 'output_schema': {'choice_index': 'integer 0..4', 'reasoning': 'brief evidence-grounded explanation', 'used_graph_evidence_ranks': 'array of frozen graph evidence ranks', 'used_visual_evidence_ranks': 'array of retrieved visual evidence ranks', 'evidence_sufficiency': 'sufficient|partial|insufficient'}}
    return [{'role': 'system', 'content': 'Answer NExT-QA multiple-choice questions using only the supplied temporal graph evidence and graph-bound text rendered from offline visual retrieval. You receive no images, paths, crops, or visual vectors. Evaluate all five choices with the same evidence and compute budget. Return exactly one JSON object.'}, {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False, sort_keys=True)}]

def parse_visual_mcq_response(response: Mapping[str, Any], *, answer_choices: Sequence[str], graph_evidence_count: int, visual_evidence_count: int) -> dict[str, Any]:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices or (not isinstance(choices[0], Mapping)):
        raise ValueError('response has no choices')
    message = choices[0].get('message')
    if not isinstance(message, Mapping):
        raise ValueError('response choice has no message')
    content = message.get('content')
    if isinstance(content, list):
        content = '\n'.join((str(item.get('text') or '') for item in content if isinstance(item, Mapping)))
    if not isinstance(content, str):
        raise ValueError('response message content is empty')
    text = content.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
        if text.lower().startswith('json'):
            text = text[4:].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, Mapping):
        raise ValueError('model response must be one JSON object')
    choice_index = parsed.get('choice_index')
    if isinstance(choice_index, bool) or not isinstance(choice_index, int) or (not 0 <= choice_index < len(answer_choices)):
        raise ValueError('choice_index must be an integer in range')
    graph_ranks = _validate_ranks(parsed.get('used_graph_evidence_ranks', []), graph_evidence_count, 'graph')
    visual_ranks = _validate_ranks(parsed.get('used_visual_evidence_ranks', []), visual_evidence_count, 'visual')
    sufficiency = str(parsed.get('evidence_sufficiency') or 'unknown').strip().lower()
    if sufficiency not in SUFFICIENCY_VALUES:
        raise ValueError('invalid evidence_sufficiency value')
    return {'choice_index': choice_index, 'choice_text': str(answer_choices[choice_index]), 'reasoning': str(parsed.get('reasoning') or '').strip(), 'used_graph_evidence_ranks': graph_ranks, 'used_visual_evidence_ranks': visual_ranks, 'evidence_sufficiency': sufficiency}

def _validate_ranks(value: Any, count: int, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f'used_{name}_evidence_ranks must be a list')
    ranks: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or (not 1 <= item <= count):
            raise ValueError(f'used_{name}_evidence_ranks contains an invalid rank')
        if item not in ranks:
            ranks.append(item)
    return ranks

def answer_visual_mcq(question: Mapping[str, Any], graph: Mapping[str, Any], visual: Mapping[str, Any], *, config: VisualMCQConfig, answer_mode: str, client: OpenAICompatibleMCQClient | None=None, api_key_slot: int=0) -> dict[str, Any]:
    messages = build_visual_qa_messages(question, graph, visual, config=config)
    choices = [str(choice) for choice in question['answer_choices']]
    graph_count = len(graph.get('evidence', []))
    visual_count = len(visual['visual_evidence'])
    attempts = 0
    parsed: dict[str, Any] | None = None
    raw_response: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    last_stage = 'qa_generation'
    live_client = client
    if answer_mode == 'mock':
        index = int(hashlib.sha256(str(question['question_id']).encode('utf-8')).hexdigest(), 16) % 5
        raw_response = {'id': 'nextqa-visual-mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'choice_index': index, 'reasoning': 'Deterministic offline mock response.', 'used_graph_evidence_ranks': [1] if graph_count else [], 'used_visual_evidence_ranks': [1], 'evidence_sufficiency': 'unknown'})}}], 'usage': None}
        attempts = 1
        parsed = parse_visual_mcq_response(raw_response, answer_choices=choices, graph_evidence_count=graph_count, visual_evidence_count=visual_count)
    elif answer_mode == 'live':
        live_client = client or OpenAICompatibleMCQClient(config.transport, api_key_slot=api_key_slot)
        payload = build_chat_payload(messages, config=config.transport)
        for attempt_index in range(config.transport.max_attempts):
            attempts += 1
            try:
                raw_response = live_client.post(payload, attempt=attempt_index)
                last_stage = 'qa_response_parse'
                parsed = parse_visual_mcq_response(raw_response, answer_choices=choices, graph_evidence_count=graph_count, visual_evidence_count=visual_count)
                break
            except REQUEST_EXCEPTIONS + (RuntimeError, ValueError, TypeError) as exc:
                last_error = exc
                if attempts < config.transport.max_attempts:
                    seed = hashlib.sha256(f"{question['question_id']}:{attempt_index}".encode('utf-8')).digest()
                    jitter = int.from_bytes(seed[:4], 'big') / 2 ** 32 * config.transport.retry_jitter_sec
                    delay = config.transport.retry_backoff_sec * 2 ** attempt_index + jitter
                    if delay > 0:
                        time.sleep(delay)
    else:
        raise ValueError(f'unsupported answer mode: {answer_mode!r}')
    combined_count = graph_count + visual_count
    common = {'schema_version': 'nextqa_visual_enhanced_mcq_prediction_v2', 'question_id': str(question['question_id']), 'video_id': str(question['video_id']), 'model': config.transport.model if answer_mode == 'live' else 'nextqa-visual-mock-v2', 'answer_mode': answer_mode, 'qa_profile': config.transport.profile_name, 'graph_retrieval_profile': str(graph['graph_retrieval_profile']), 'graph_record_ids': list(graph.get('graph_record_ids', [])), 'graph_evidence_count': graph_count, 'visual_evidence_count': visual_count, 'visual_candidate_ids': list(visual.get('used_visual_candidate_ids', [])), 'route_id': str(visual['route_id']), 'handler_id': str(visual['handler_id']), 'handler_status': str(visual['handler_status']), 'evidence_budget_per_choice': [combined_count] * 5, 'request_sha256': request_sha256(messages), 'attempt_count': attempts, 'api_key_slot': getattr(live_client, 'api_key_slot', api_key_slot) if answer_mode == 'live' else None, 'api_key_pool_size': getattr(live_client, 'api_key_pool_size', None) if answer_mode == 'live' else None, 'uses_gold_answer': False, 'uses_official_type': False, 'offline_reads_rgb': False, 'offline_reads_visual_index': True, 'qa_receives_images': False, 'qa_receives_image_paths': False, 'qa_receives_visual_embeddings': False, 'uses_visual_index': True}
    if parsed is None:
        return {**common, 'status': 'error', 'graph_rag_status': 'graph_plus_visual_mcq_error', 'predicted_choice_index': None, 'predicted_choice_text': None, 'error_stage': last_stage, 'error_type': type(last_error).__name__ if last_error else 'UnknownError', 'error_message': str(last_error or 'model response did not validate'), 'response_metadata': response_metadata(raw_response) if raw_response is not None else None}
    return {**common, 'status': 'completed', 'graph_rag_status': 'graph_plus_visual_mcq_completed', 'predicted_choice_index': parsed['choice_index'], 'predicted_choice_text': parsed['choice_text'], 'reasoning': parsed['reasoning'], 'used_graph_evidence_ranks': parsed['used_graph_evidence_ranks'], 'used_visual_evidence_ranks': parsed['used_visual_evidence_ranks'], 'evidence_sufficiency': parsed['evidence_sufficiency'], 'response_metadata': response_metadata(raw_response or {})}
