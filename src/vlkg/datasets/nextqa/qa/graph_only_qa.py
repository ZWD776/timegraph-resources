"""Dataset-isolated graph-only multiple-choice QA for NExT-QA."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import yaml
REQUEST_EXCEPTIONS = (HTTPError, URLError, TimeoutError, ConnectionError, OSError, socket.timeout, json.JSONDecodeError)
STRICT_IR_FIELDS = ('target_entities', 'actors', 'actions', 'objects', 'temporal_direction', 'temporal_anchor', 'referenced_event', 'causal_query', 'count_query', 'location_query', 'attribute_query', 'answer_format')
SUFFICIENCY_VALUES = frozenset({'sufficient', 'partial', 'insufficient', 'unknown'})
REQUIRED_MODEL = 'gpt-5.6-luna'
COMPATIBLE_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'

@dataclass(frozen=True)
class GraphOnlyMCQConfig:
    profile_name: str
    provider: str
    model: str
    api_base_url: str
    api_base_url_env_var: str | None
    endpoint_path: str
    api_key_env_var: str
    timeout_sec: float
    temperature: float
    max_tokens: int
    max_attempts: int
    evidence_limit: int
    thinking_enabled: bool
    api_key_pool_env_var: str = 'OPENAI_API_KEYS'
    retry_backoff_sec: float = 1.0
    retry_jitter_sec: float = 0.25

    def resolved_api_base_url(self) -> str:
        if self.api_base_url_env_var:
            override = os.getenv(self.api_base_url_env_var, '').strip()
            if override:
                return override.rstrip('/')
        return self.api_base_url.rstrip('/')

    def endpoint_url(self) -> str:
        return urljoin(self.resolved_api_base_url() + '/', self.endpoint_path.lstrip('/'))

def load_graph_only_mcq_config(path: str | Path) -> GraphOnlyMCQConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if payload.get('schema_version') != 'nextqa_graph_only_mcq_config_v1':
        raise ValueError('unsupported NExT-QA graph-only QA config schema')
    qa = payload.get('qa')
    if not isinstance(qa, Mapping):
        raise ValueError('NExT-QA graph-only QA config requires a qa mapping')
    config = GraphOnlyMCQConfig(profile_name=str(qa.get('profile_name') or '').strip(), provider=str(qa.get('provider') or '').strip(), model=str(qa.get('model') or '').strip(), api_base_url=str(qa.get('api_base_url') or '').strip(), api_base_url_env_var=str(qa['api_base_url_env_var']).strip() if qa.get('api_base_url_env_var') else None, endpoint_path=str(qa.get('endpoint_path') or '').strip(), api_key_env_var=str(qa.get('api_key_env_var') or '').strip(), timeout_sec=float(qa.get('timeout_sec', 180.0)), temperature=float(qa.get('temperature', 0.1)), max_tokens=int(qa.get('max_tokens', 500)), max_attempts=int(qa.get('max_attempts', 2)), evidence_limit=int(qa.get('evidence_limit', 16)), thinking_enabled=bool(qa.get('thinking_enabled', False)), api_key_pool_env_var=str(qa.get('api_key_pool_env_var') or 'OPENAI_API_KEYS').strip(), retry_backoff_sec=float(qa.get('retry_backoff_sec', 1.0)), retry_jitter_sec=float(qa.get('retry_jitter_sec', 0.25)))
    if config.model != REQUIRED_MODEL:
        raise ValueError(f'NExT-QA graph-only QA requires {REQUIRED_MODEL}, got {config.model!r}')
    if config.provider != 'openai_compatible':
        raise ValueError('NExT-QA graph-only QA requires the openai_compatible provider')
    if not config.profile_name.startswith('nextqa_graph_only_mcq_'):
        raise ValueError('NExT-QA QA profile must be dataset-isolated')
    if not config.api_base_url or not config.endpoint_path or (not config.api_key_env_var):
        raise ValueError('NExT-QA graph-only QA endpoint configuration is incomplete')
    if config.max_tokens < 1 or config.max_attempts < 1 or config.evidence_limit < 1:
        raise ValueError('NExT-QA graph-only QA budgets must be positive')
    if config.retry_backoff_sec < 0 or config.retry_jitter_sec < 0:
        raise ValueError('NExT-QA retry delays must be non-negative')
    return config

def parse_api_key_pool(value: str | None=None) -> list[str]:
    """Parse a comma/semicolon/newline separated pool without exposing secrets."""
    raw = value
    if raw is None:
        raw = os.getenv('OPENAI_API_KEYS', '')
    keys: list[str] = []
    seen: set[str] = set()
    for item in re.split('[,;\\r\\n]+', str(raw or '')):
        key = item.strip()
        if key and key not in seen:
            keys.append(key)
            seen.add(key)
    return keys

def stable_api_key_slot(question_id: str, pool_size: int) -> int:
    if pool_size < 1:
        return 0
    digest = hashlib.sha256(str(question_id).encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % pool_size

def validate_qa_input_pair(question_row: Mapping[str, Any], retrieval_row: Mapping[str, Any], *, evidence_limit: int) -> None:
    question_id = str(question_row.get('question_id') or '')
    if not question_id or question_id != str(retrieval_row.get('question_id') or ''):
        raise ValueError('question and retrieval IDs do not match')
    if str(question_row.get('video_id') or '') != str(retrieval_row.get('video_id') or ''):
        raise ValueError(f'question and retrieval video IDs do not match for {question_id}')
    choices = question_row.get('answer_choices')
    if not isinstance(choices, list) or len(choices) != 5:
        raise ValueError(f'question must have exactly five answer choices: {question_id}')
    if any((not str(choice).strip() for choice in choices)):
        raise ValueError(f'question contains an empty answer choice: {question_id}')
    if question_row.get('uses_gold_answer') is not False:
        raise ValueError(f'question runtime purity failed: {question_id}')
    if retrieval_row.get('status') != 'completed':
        raise ValueError(f'graph retrieval is not completed: {question_id}')
    purity = retrieval_row.get('purity')
    required_false = ('uses_gold_answer', 'uses_official_type', 'uses_answer_choices', 'reads_rgb', 'reads_object_crop', 'reads_visual_embedding_cache', 'uses_visual_index', 'shared_hybrid_retrieve_called')
    if not isinstance(purity, Mapping) or any((purity.get(key) is not False for key in required_false)):
        raise ValueError(f'graph retrieval purity failed: {question_id}')
    evidence = retrieval_row.get('evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f'graph retrieval has no evidence: {question_id}')
    if len(evidence) > evidence_limit:
        raise ValueError(f'retrieval evidence exceeds fixed QA budget {evidence_limit}: {question_id}')

def build_qa_messages(question_row: Mapping[str, Any], retrieval_row: Mapping[str, Any], *, evidence_limit: int) -> list[dict[str, str]]:
    """Build one route-neutral prompt with one shared budget for all choices."""
    validate_qa_input_pair(question_row, retrieval_row, evidence_limit=evidence_limit)
    choices = [str(choice).strip() for choice in question_row['answer_choices']]
    evidence_rows = list(retrieval_row['evidence'])[:evidence_limit]
    evidence = [{'rank': int(row.get('rank', index)), 'record_id': str(row.get('source_record_id') or ''), 'source_kind': str(row.get('source_kind') or ''), 'time_range': row.get('time_range') or {}, 'text': str(row.get('evidence_text') or '').strip()} for index, row in enumerate(evidence_rows, start=1)]
    strict_ir = {field: question_row.get(field) for field in STRICT_IR_FIELDS}
    prompt = {'task': 'nextqa_graph_only_multiple_choice', 'question': str(question_row.get('normalized_question') or '').strip(), 'question_ir': strict_ir, 'answer_choices': [{'choice_index': index, 'choice_text': choice} for index, choice in enumerate(choices)], 'graph_evidence': evidence, 'policy': {'use_only_provided_graph_evidence': True, 'do_not_use_images_or_visual_indexes': True, 'do_not_use_answer_keys': True, 'assess_all_five_choices_symmetrically': True, 'same_graph_evidence_budget_for_every_choice': len(evidence), 'choice_indices_are_zero_based': True, 'return_one_choice_only': True, 'state_when_graph_evidence_is_incomplete': True}, 'output_schema': {'choice_index': 'integer 0..4', 'reasoning': 'brief graph-grounded explanation', 'used_evidence_ranks': 'array of graph evidence ranks', 'evidence_sufficiency': 'sufficient|partial|insufficient'}}
    return [{'role': 'system', 'content': 'You answer NExT-QA multiple-choice questions using only the supplied temporal graph evidence. Evaluate all five choices against the same evidence and compute budget. Do not infer from unseen frames, images, external video knowledge, question-type labels, or answer keys. Return exactly one JSON object matching the requested schema.'}, {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False, sort_keys=True)}]

def build_chat_payload(messages: Sequence[Mapping[str, str]], *, config: GraphOnlyMCQConfig) -> dict[str, Any]:
    payload: dict[str, Any] = {'model': config.model, 'messages': [dict(message) for message in messages], 'temperature': config.temperature, 'max_tokens': config.max_tokens, 'response_format': {'type': 'json_object'}}
    payload['thinking'] = {'type': 'enabled' if config.thinking_enabled else 'disabled'}
    return payload

class OpenAICompatibleMCQClient:

    def __init__(self, config: GraphOnlyMCQConfig, *, post_json: Callable[[dict[str, Any], str], dict[str, Any]] | None=None, api_key_slot: int=0) -> None:
        self.config = config
        self._post_json_override = post_json
        pool_raw = os.getenv(config.api_key_pool_env_var, '')
        self._api_keys = parse_api_key_pool(pool_raw)
        if not self._api_keys:
            single = os.getenv(config.api_key_env_var, '').strip()
            if single:
                self._api_keys = [single]
        self.api_key_slot = max(0, int(api_key_slot))

    @property
    def api_key_pool_size(self) -> int:
        return len(self._api_keys)

    def _api_key_for_attempt(self, attempt: int) -> str:
        if not self._api_keys:
            raise RuntimeError(f'missing environment variable: {self.config.api_key_env_var} or {self.config.api_key_pool_env_var}')
        return self._api_keys[(self.api_key_slot + max(0, int(attempt))) % len(self._api_keys)]

    def post(self, payload: dict[str, Any], *, attempt: int=0) -> dict[str, Any]:
        api_key = self._api_key_for_attempt(attempt)
        if self._post_json_override is not None:
            response = self._post_json_override(payload, api_key)
        else:
            request = Request(self.config.endpoint_url(), data=json.dumps(payload).encode('utf-8'), headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': COMPATIBLE_BROWSER_USER_AGENT}, method='POST')
            with urlopen(request, timeout=self.config.timeout_sec) as handle:
                response = json.loads(handle.read().decode('utf-8'))
        if not isinstance(response, dict):
            raise json.JSONDecodeError('response JSON is not an object', str(response), 0)
        return response

def _extract_content(response: Mapping[str, Any]) -> str:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices:
        raise ValueError('response has no choices')
    first = choices[0]
    if not isinstance(first, Mapping):
        raise ValueError('response choice is not an object')
    message = first.get('message')
    if not isinstance(message, Mapping):
        raise ValueError('response choice has no message')
    content = message.get('content')
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, Mapping):
                continue
            text = item.get('text')
            if isinstance(text, str):
                parts.append(text)
        return '\n'.join(parts).strip()
    raise ValueError('response message content is empty')

def _parse_json_object(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch('```(?:json)?\\s*(.*?)\\s*```', candidate, flags=re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, Mapping):
        raise ValueError('model response must be one JSON object')
    return parsed

def parse_mcq_response(response: Mapping[str, Any], *, answer_choices: Sequence[str], evidence_count: int) -> dict[str, Any]:
    parsed = _parse_json_object(_extract_content(response))
    choice_index = parsed.get('choice_index')
    if isinstance(choice_index, bool) or not isinstance(choice_index, int):
        raise ValueError('choice_index must be an integer')
    if choice_index < 0 or choice_index >= len(answer_choices):
        raise ValueError('choice_index is outside the valid range')
    ranks_value = parsed.get('used_evidence_ranks', [])
    if not isinstance(ranks_value, list):
        raise ValueError('used_evidence_ranks must be a list')
    ranks: list[int] = []
    for value in ranks_value:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError('used_evidence_ranks must contain integers')
        if value < 1 or value > evidence_count:
            raise ValueError('used_evidence_ranks contains an invalid rank')
        if value not in ranks:
            ranks.append(value)
    sufficiency = str(parsed.get('evidence_sufficiency') or 'unknown').strip().lower()
    if sufficiency not in SUFFICIENCY_VALUES:
        raise ValueError('invalid evidence_sufficiency value')
    reasoning = str(parsed.get('reasoning') or '').strip()
    return {'choice_index': choice_index, 'choice_text': str(answer_choices[choice_index]), 'reasoning': reasoning, 'used_evidence_ranks': ranks, 'evidence_sufficiency': sufficiency}

def request_sha256(messages: Sequence[Mapping[str, str]]) -> str:
    raw = json.dumps(list(messages), ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get('choices')
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get('finish_reason')
    usage = response.get('usage')
    return {'response_id': response.get('id'), 'finish_reason': finish_reason, 'usage': dict(usage) if isinstance(usage, Mapping) else None}

def answer_graph_only_mcq(question_row: Mapping[str, Any], retrieval_row: Mapping[str, Any], *, config: GraphOnlyMCQConfig, answer_mode: str, client: OpenAICompatibleMCQClient | None=None, api_key_slot: int=0) -> dict[str, Any]:
    messages = build_qa_messages(question_row, retrieval_row, evidence_limit=config.evidence_limit)
    choices = [str(choice) for choice in question_row['answer_choices']]
    evidence_count = len(retrieval_row['evidence'])
    prompt_hash = request_sha256(messages)
    attempts = 0
    last_error: Exception | None = None
    last_stage = 'qa_generation'
    parsed: dict[str, Any] | None = None
    raw_response: Mapping[str, Any] | None = None
    if answer_mode == 'mock':
        choice_index = int(hashlib.sha256(str(question_row['question_id']).encode('utf-8')).hexdigest(), 16) % 5
        raw_response = {'id': 'nextqa-mock', 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'choice_index': choice_index, 'reasoning': 'Deterministic offline mock response.', 'used_evidence_ranks': [1], 'evidence_sufficiency': 'unknown'})}}], 'usage': None}
        attempts = 1
        parsed = parse_mcq_response(raw_response, answer_choices=choices, evidence_count=evidence_count)
    elif answer_mode == 'live':
        live_client = client or OpenAICompatibleMCQClient(config, api_key_slot=api_key_slot)
        payload = build_chat_payload(messages, config=config)
        for attempt_index in range(config.max_attempts):
            attempts += 1
            try:
                raw_response = live_client.post(payload, attempt=attempt_index)
                last_stage = 'qa_response_parse'
                parsed = parse_mcq_response(raw_response, answer_choices=choices, evidence_count=evidence_count)
                break
            except REQUEST_EXCEPTIONS + (RuntimeError, ValueError, TypeError) as exc:
                last_error = exc
                if attempts < config.max_attempts:
                    jitter_seed = hashlib.sha256(f"{question_row['question_id']}:{attempt_index}".encode('utf-8')).digest()
                    jitter = int.from_bytes(jitter_seed[:4], 'big') / 2 ** 32 * config.retry_jitter_sec
                    delay = config.retry_backoff_sec * 2 ** attempt_index + jitter
                    if delay > 0:
                        time.sleep(delay)
    else:
        raise ValueError(f'unsupported answer mode: {answer_mode!r}')
    common = {'schema_version': 'nextqa_graph_only_mcq_prediction_v1', 'question_id': str(question_row['question_id']), 'video_id': str(question_row['video_id']), 'model': config.model if answer_mode == 'live' else 'nextqa-mock-v1', 'answer_mode': answer_mode, 'qa_profile': config.profile_name, 'graph_retrieval_profile': str(retrieval_row['graph_retrieval_profile']), 'graph_record_ids': list(retrieval_row.get('graph_record_ids', [])), 'graph_evidence_count': evidence_count, 'evidence_budget_per_choice': [evidence_count] * 5, 'request_sha256': prompt_hash, 'attempt_count': attempts, 'api_key_slot': getattr(live_client, 'api_key_slot', api_key_slot) if answer_mode == 'live' else None, 'api_key_pool_size': getattr(live_client, 'api_key_pool_size', None) if answer_mode == 'live' else None, 'uses_gold_answer': False, 'uses_official_type': False, 'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_embedding_cache': False, 'uses_visual_index': False}
    if parsed is None:
        return {**common, 'status': 'error', 'graph_rag_status': 'graph_only_mcq_error', 'predicted_choice_index': None, 'predicted_choice_text': None, 'error_stage': last_stage, 'error_type': type(last_error).__name__ if last_error else 'UnknownError', 'error_message': str(last_error or 'model response did not validate'), 'response_metadata': response_metadata(raw_response) if raw_response is not None else None}
    return {**common, 'status': 'completed', 'graph_rag_status': 'graph_only_mcq_completed', 'predicted_choice_index': parsed['choice_index'], 'predicted_choice_text': parsed['choice_text'], 'reasoning': parsed['reasoning'], 'used_evidence_ranks': parsed['used_evidence_ranks'], 'evidence_sufficiency': parsed['evidence_sufficiency'], 'response_metadata': response_metadata(raw_response or {})}
