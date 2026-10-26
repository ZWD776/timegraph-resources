"""Question-only LLM semantic compiler for NExT-QA."""
from __future__ import annotations
from copy import deepcopy
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
from vlkg.datasets.nextqa.retrieval.question_ir import normalize_question_text
REQUIRED_MODEL = 'gpt-5.6-luna'
IR_SCHEMA_VERSION = 'nextqa_question_only_llm_ir_v3'
PROFILE_NAME = 'nextqa_question_only_llm_ir_v3'
COMPATIBLE_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'
REQUEST_EXCEPTIONS = (HTTPError, URLError, TimeoutError, ConnectionError, OSError, socket.timeout, json.JSONDecodeError)
INTENT_CONTRACTS: dict[str, dict[str, frozenset[str] | str]] = {'causal_why': {'answer_targets': frozenset({'reason'}), 'temporal_relations': frozenset({'none', 'before'}), 'causal_relation': 'reason_for_effect', 'visual_evidence_demand': 'preceding_context'}, 'causal_how': {'answer_targets': frozenset({'action_chain'}), 'temporal_relations': frozenset({'none', 'before'}), 'causal_relation': 'mechanism_to_result', 'visual_evidence_demand': 'ordered_action_chain'}, 'temporal_previous': {'answer_targets': frozenset({'preceding_event'}), 'temporal_relations': frozenset({'before'}), 'causal_relation': 'none', 'visual_evidence_demand': 'preceding_event_window'}, 'temporal_next': {'answer_targets': frozenset({'following_event'}), 'temporal_relations': frozenset({'after'}), 'causal_relation': 'none', 'visual_evidence_demand': 'following_event_window'}, 'temporal_concurrent': {'answer_targets': frozenset({'concurrent_event'}), 'temporal_relations': frozenset({'concurrent'}), 'causal_relation': 'none', 'visual_evidence_demand': 'same_time_window'}, 'descriptive_count': {'answer_targets': frozenset({'count'}), 'temporal_relations': frozenset({'none'}), 'causal_relation': 'none', 'visual_evidence_demand': 'deduplicated_entities'}, 'descriptive_location': {'answer_targets': frozenset({'location'}), 'temporal_relations': frozenset({'none'}), 'causal_relation': 'none', 'visual_evidence_demand': 'spatial_context'}, 'descriptive_direct': {'answer_targets': frozenset({'object', 'action', 'state', 'attribute'}), 'temporal_relations': frozenset({'none'}), 'causal_relation': 'none', 'visual_evidence_demand': 'direct_visual'}}
REQUIRED_RESPONSE_KEYS = frozenset({'question_id', 'intent', 'answer_target', 'actors', 'actions', 'objects', 'states', 'anchor_event', 'temporal_relation', 'causal_relation', 'visual_evidence_demand', 'confidence'})
REQUIRED_ANCHOR_KEYS = frozenset({'text', 'actors', 'actions', 'objects', 'states'})
ANCHOR_REQUIRED_INTENTS = frozenset({'causal_how', 'temporal_previous', 'temporal_next', 'temporal_concurrent'})
OFFICIAL_TYPE_TO_INTENT = {'CH': 'causal_how', 'CW': 'causal_why', 'DC': 'descriptive_count', 'DL': 'descriptive_location', 'DO': 'descriptive_direct', 'TC': 'temporal_concurrent', 'TN': 'temporal_next', 'TP': 'temporal_previous'}
OFFICIAL_TYPE_LABELS = {'CH': 'causal_how', 'CW': 'causal_why', 'DC': 'descriptive_count', 'DL': 'descriptive_location', 'DO': 'descriptive_object_or_action', 'TC': 'temporal_concurrent', 'TN': 'temporal_next', 'TP': 'temporal_previous'}

@dataclass(frozen=True)
class LLMQuestionIRConfig:
    profile_name: str
    provider: str
    model: str
    api_base_url: str
    api_base_url_env_var: str | None
    endpoint_path: str
    api_key_env_var: str
    api_key_pool_env_var: str
    timeout_sec: float
    temperature: float
    max_tokens: int
    max_attempts: int
    thinking_enabled: bool
    retry_backoff_sec: float
    retry_jitter_sec: float

    def resolved_api_base_url(self) -> str:
        if self.api_base_url_env_var:
            override = os.getenv(self.api_base_url_env_var, '').strip()
            if override:
                return override.rstrip('/')
        return self.api_base_url.rstrip('/')

    def endpoint_url(self) -> str:
        return urljoin(self.resolved_api_base_url() + '/', self.endpoint_path.lstrip('/'))

def load_llm_question_ir_config(path: str | Path) -> LLMQuestionIRConfig:
    config_path = Path(path).resolve()
    if not config_path.name.startswith('nextqa_'):
        raise ValueError('NExT-QA LLM IR requires a dataset-prefixed config')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    if payload.get('schema_version') != 'nextqa_question_only_llm_ir_config_v1':
        raise ValueError('unsupported NExT-QA LLM IR config schema')
    values = payload.get('ir')
    if not isinstance(values, Mapping):
        raise ValueError('NExT-QA LLM IR config requires an ir mapping')
    config = LLMQuestionIRConfig(profile_name=str(values.get('profile_name') or '').strip(), provider=str(values.get('provider') or '').strip(), model=str(values.get('model') or '').strip(), api_base_url=str(values.get('api_base_url') or '').strip(), api_base_url_env_var=str(values['api_base_url_env_var']).strip() if values.get('api_base_url_env_var') else None, endpoint_path=str(values.get('endpoint_path') or '').strip(), api_key_env_var=str(values.get('api_key_env_var') or '').strip(), api_key_pool_env_var=str(values.get('api_key_pool_env_var') or 'OPENAI_API_KEYS').strip(), timeout_sec=float(values.get('timeout_sec', 300.0)), temperature=float(values.get('temperature', 0.1)), max_tokens=int(values.get('max_tokens', 900)), max_attempts=int(values.get('max_attempts', 2)), thinking_enabled=bool(values.get('thinking_enabled', False)), retry_backoff_sec=float(values.get('retry_backoff_sec', 1.0)), retry_jitter_sec=float(values.get('retry_jitter_sec', 0.25)))
    if config.profile_name != PROFILE_NAME:
        raise ValueError(f'NExT-QA LLM IR profile must be {PROFILE_NAME}')
    if config.provider != 'openai_compatible' or config.model != REQUIRED_MODEL:
        raise ValueError(f'NExT-QA LLM IR requires openai_compatible/{REQUIRED_MODEL}')
    if not config.api_base_url or not config.endpoint_path or (not config.api_key_env_var):
        raise ValueError('NExT-QA LLM IR endpoint configuration is incomplete')
    if config.max_tokens < 1 or config.max_attempts < 1 or config.timeout_sec <= 0:
        raise ValueError('NExT-QA LLM IR budgets must be positive')
    if config.retry_backoff_sec < 0 or config.retry_jitter_sec < 0:
        raise ValueError('NExT-QA LLM IR retry delays must be non-negative')
    return config

def parse_api_key_pool(value: str | None=None) -> list[str]:
    raw = value if value is not None else os.getenv('OPENAI_API_KEYS', '')
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
    digest = hashlib.sha256(question_id.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') % pool_size

def build_question_ir_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    question_id, question = _question_identity(row)
    prompt = {'task': 'compile_nextqa_question_semantics', 'input': {'question_id': question_id, 'question': question}, 'intent_definitions': {'causal_why': 'asks the reason for an observed action, event, or outcome', 'causal_how': 'asks the means, manner, method, action chain, or mechanism used to do something', 'temporal_previous': 'asks what happened before an anchor event', 'temporal_next': 'asks what happened after an anchor event', 'temporal_concurrent': 'asks what happened during or at the same time as an anchor event', 'descriptive_count': 'asks for a count', 'descriptive_location': 'asks where an entity or event is located', 'descriptive_direct': 'asks directly for an object, action, state, or attribute'}, 'rules': ['Use only the wording of the supplied question.', 'Compile semantics; do not answer the question and do not guess an unseen event.', 'Select causal_why for a why-question even if it also contains a temporal phrase.', 'Select descriptive_count for how-many and descriptive_location for where.', 'Otherwise, explicit before/previous, after/next, or while/as/when/during language selects the corresponding temporal intent.', 'A question asking for an action or state during a named ongoing activity is temporal_concurrent even when during is implicit.', 'After temporal markers are handled, a how-question asking means, manner, method, or mechanism is causal_how.', 'Use descriptive_direct only when none of the causal, temporal, count, or location intents applies.', 'Use concise open-vocabulary singular lemmas where practical.', 'The anchor event is the event explicitly described in the question that constrains the unknown answer.', 'Keep lists empty when the question does not name that semantic role.', 'Return exactly the required JSON object and no markdown.'], 'required_schema': {'question_id': 'copy exactly', 'intent': 'one intent_definitions key', 'answer_target': 'reason|action_chain|preceding_event|following_event|concurrent_event|count|location|object|action|state|attribute', 'actors': ['named or referenced actors'], 'actions': ['question-grounded actions'], 'objects': ['question-grounded objects'], 'states': ['question-grounded states or attributes'], 'anchor_event': {'text': 'concise question-grounded anchor event, or empty', 'actors': [], 'actions': [], 'objects': [], 'states': []}, 'temporal_relation': 'none|before|after|concurrent', 'causal_relation': 'none|reason_for_effect|mechanism_to_result', 'visual_evidence_demand': 'preceding_context|ordered_action_chain|preceding_event_window|following_event_window|same_time_window|deduplicated_entities|spatial_context|direct_visual', 'confidence': 'number from 0 to 1'}, 'consistency': {intent: {'answer_targets': sorted(contract['answer_targets']), 'temporal_relations': sorted(contract['temporal_relations']), 'causal_relation': contract['causal_relation'], 'visual_evidence_demand': contract['visual_evidence_demand']} for intent, contract in INTENT_CONTRACTS.items()}}
    return [{'role': 'system', 'content': 'You are a semantic compiler for short-video question answering. Do not answer questions. Use no hidden labels, candidate answers, graph evidence, visual evidence, or outside facts. Return strict JSON.'}, {'role': 'user', 'content': json.dumps(prompt, ensure_ascii=False)}]

def request_fingerprint(row: Mapping[str, Any]) -> str:
    encoded = json.dumps(build_question_ir_messages(row), ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def audit_prompt_isolation(row: Mapping[str, Any]) -> dict[str, bool]:
    baseline = request_fingerprint(row)
    checks: dict[str, bool] = {}
    perturbations = {'official_type': '__PERTURBED_TYPE__', 'answer_choices': [f'perturbed-{index}' for index in range(5)], 'gold_choice_index': 4, 'gold_answer': '__PERTURBED_GOLD__', 'video_id': '__PERTURBED_VIDEO__', 'vidor_id': '__PERTURBED_VIDOR__', 'image_path': '__PERTURBED_IMAGE_PATH__'}
    for key, value in perturbations.items():
        changed = deepcopy(dict(row))
        changed[key] = value
        checks[f'{key}_invariant'] = request_fingerprint(changed) == baseline
    return checks

def build_chat_payload(messages: Sequence[Mapping[str, str]], *, config: LLMQuestionIRConfig) -> dict[str, Any]:
    return {'model': config.model, 'messages': [dict(message) for message in messages], 'temperature': config.temperature, 'max_tokens': config.max_tokens, 'response_format': {'type': 'json_object'}, 'thinking': {'type': 'enabled' if config.thinking_enabled else 'disabled'}}

class OpenAICompatibleIRClient:

    def __init__(self, config: LLMQuestionIRConfig, *, api_key_slot: int=0, post_json: Callable[[dict[str, Any], str], dict[str, Any]] | None=None) -> None:
        self.config = config
        self.api_key_slot = max(0, int(api_key_slot))
        self._post_json_override = post_json
        self._api_keys = parse_api_key_pool(os.getenv(config.api_key_pool_env_var, ''))
        if not self._api_keys:
            single = os.getenv(config.api_key_env_var, '').strip()
            if single:
                self._api_keys = [single]

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

def parse_semantic_ir_response(response: Mapping[str, Any], *, expected_question_id: str) -> dict[str, Any]:
    content = _extract_content(response)
    if content.startswith('```'):
        content = re.sub('^```(?:json)?\\s*', '', content, flags=re.IGNORECASE)
        content = re.sub('\\s*```$', '', content)
    raw = json.loads(content)
    if not isinstance(raw, Mapping):
        raise ValueError('LLM IR response must be one JSON object')
    keys = frozenset((str(key) for key in raw))
    if keys != REQUIRED_RESPONSE_KEYS:
        raise ValueError(f'LLM IR response keys do not match the minimal v3 schema: missing={sorted(REQUIRED_RESPONSE_KEYS - keys)} extra={sorted(keys - REQUIRED_RESPONSE_KEYS)}')
    question_id = _required_string(raw.get('question_id'), 'question_id')
    if question_id != expected_question_id:
        raise ValueError('LLM IR response question_id does not match the request')
    intent = _required_string(raw.get('intent'), 'intent').lower()
    if intent not in INTENT_CONTRACTS:
        raise ValueError(f'unsupported NExT-QA intent: {intent!r}')
    answer_target = _required_string(raw.get('answer_target'), 'answer_target').lower()
    temporal_relation = _required_string(raw.get('temporal_relation'), 'temporal_relation').lower()
    causal_relation = _required_string(raw.get('causal_relation'), 'causal_relation').lower()
    visual_demand = _required_string(raw.get('visual_evidence_demand'), 'visual_evidence_demand').lower()
    contract = INTENT_CONTRACTS[intent]
    if answer_target not in contract['answer_targets']:
        raise ValueError(f'answer_target is inconsistent with {intent}')
    if temporal_relation not in contract['temporal_relations']:
        raise ValueError(f'temporal_relation is inconsistent with {intent}')
    if causal_relation != contract['causal_relation']:
        raise ValueError(f'causal_relation is inconsistent with {intent}')
    if visual_demand != contract['visual_evidence_demand']:
        raise ValueError(f'visual_evidence_demand is inconsistent with {intent}')
    anchor_raw = raw.get('anchor_event')
    if not isinstance(anchor_raw, Mapping):
        raise ValueError('anchor_event must be an object')
    anchor_keys = frozenset((str(key) for key in anchor_raw))
    if anchor_keys != REQUIRED_ANCHOR_KEYS:
        raise ValueError('anchor_event keys do not match the minimal v3 schema')
    anchor = {'text': _optional_string(anchor_raw.get('text'), 'anchor_event.text'), 'actors': _string_list(anchor_raw.get('actors'), 'anchor_event.actors'), 'actions': _string_list(anchor_raw.get('actions'), 'anchor_event.actions'), 'objects': _string_list(anchor_raw.get('objects'), 'anchor_event.objects'), 'states': _string_list(anchor_raw.get('states'), 'anchor_event.states')}
    if intent in ANCHOR_REQUIRED_INTENTS and (not anchor['text']):
        raise ValueError(f'anchor_event.text is required for {intent}')
    confidence = raw.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError('confidence must be numeric')
    confidence_value = float(confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError('confidence must be between 0 and 1')
    return {'question_id': question_id, 'intent': intent, 'answer_target': answer_target, 'actors': _string_list(raw.get('actors'), 'actors'), 'actions': _string_list(raw.get('actions'), 'actions'), 'objects': _string_list(raw.get('objects'), 'objects'), 'states': _string_list(raw.get('states'), 'states'), 'anchor_event': anchor, 'temporal_relation': temporal_relation, 'causal_relation': causal_relation, 'visual_evidence_demand': visual_demand, 'confidence': confidence_value}

def semantic_ir_to_runtime_record(question_row: Mapping[str, Any], semantic_ir: Mapping[str, Any], *, model: str, request_sha256: str, attempt_count: int, api_key_slot: int | None, api_key_pool_size: int | None, response_metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    question_id, question = _question_identity(question_row)
    if str(semantic_ir.get('question_id') or '') != question_id:
        raise ValueError('semantic IR and question row IDs do not match')
    video_id = str(question_row.get('video_id') or '').strip()
    choices = [str(value).strip() for value in question_row.get('answer_choices', [])]
    if len(choices) != 5 or any((not value for value in choices)):
        raise ValueError(f'question must have five non-empty choices: {question_id}')
    anchor = dict(semantic_ir['anchor_event'])
    actors = _dedupe_strings([*semantic_ir['actors'], *anchor['actors']])
    actions = _dedupe_strings([*semantic_ir['actions'], *anchor['actions']])
    objects = _dedupe_strings([*semantic_ir['objects'], *anchor['objects']])
    states = _dedupe_strings([*semantic_ir['states'], *anchor['states']])
    intent = str(semantic_ir['intent'])
    causal_query = {'causal_why': 'why', 'causal_how': 'how'}.get(intent, 'none')
    answer_target = str(semantic_ir['answer_target'])
    attribute_query = answer_target if answer_target in {'state', 'attribute'} else 'none'
    return {'schema_version': IR_SCHEMA_VERSION, 'status': 'completed', 'question_id': question_id, 'video_id': video_id, 'official_type': str(question_row.get('official_type') or '').strip(), 'normalized_question': question, 'answer_choices': choices, 'answer_format': str(question_row.get('answer_format') or 'multiple_choice_5'), 'intent': intent, 'answer_target': answer_target, 'actors': actors, 'actions': actions, 'objects': objects, 'states': states, 'target_entities': _dedupe_strings([*actors, *objects]), 'anchor_event': anchor, 'temporal_relation': str(semantic_ir['temporal_relation']), 'temporal_direction': str(semantic_ir['temporal_relation']), 'temporal_anchor': _temporal_anchor(question, str(anchor['text'])), 'referenced_event': str(anchor['text']), 'causal_relation': str(semantic_ir['causal_relation']), 'causal_query': causal_query, 'count_query': intent == 'descriptive_count', 'location_query': intent == 'descriptive_location', 'attribute_query': attribute_query, 'visual_evidence_demand': str(semantic_ir['visual_evidence_demand']), 'confidence': float(semantic_ir['confidence']), 'ir_builder': PROFILE_NAME, 'model': model, 'request_sha256': request_sha256, 'attempt_count': attempt_count, 'api_key_slot': api_key_slot, 'api_key_pool_size': api_key_pool_size, 'response_metadata': dict(response_metadata or {}), 'rule_fallback_used': False, 'retrieval_uses_official_type': False, 'retrieval_uses_answer_choices': False, 'retrieval_uses_visual_evidence_demand': False, 'uses_gold_answer': False}

def semantic_only_view(runtime_record: Mapping[str, Any]) -> dict[str, Any]:
    fields = ('schema_version', 'status', 'question_id', 'video_id', 'normalized_question', 'intent', 'answer_target', 'actors', 'actions', 'objects', 'states', 'anchor_event', 'temporal_relation', 'causal_relation', 'visual_evidence_demand', 'confidence', 'ir_builder', 'model', 'request_sha256', 'attempt_count', 'api_key_slot', 'api_key_pool_size', 'response_metadata', 'uses_gold_answer')
    return {field: deepcopy(runtime_record[field]) for field in fields}

def compile_question_ir(question_row: Mapping[str, Any], *, config: LLMQuestionIRConfig, mode: str, client: OpenAICompatibleIRClient | None=None, api_key_slot: int=0) -> dict[str, Any]:
    question_id, _ = _question_identity(question_row)
    messages = build_question_ir_messages(question_row)
    request_hash = request_fingerprint(question_row)
    attempts = 0
    last_error: Exception | None = None
    last_stage = 'ir_generation'
    response: Mapping[str, Any] | None = None
    semantic_ir: dict[str, Any] | None = None
    active_client: OpenAICompatibleIRClient | None = None
    if mode == 'mock':
        semantic_ir = _mock_semantic_ir(question_row)
        attempts = 1
    elif mode == 'live':
        active_client = client or OpenAICompatibleIRClient(config, api_key_slot=api_key_slot)
        payload = build_chat_payload(messages, config=config)
        for attempt_index in range(config.max_attempts):
            attempts += 1
            try:
                response = active_client.post(payload, attempt=attempt_index)
                last_stage = 'ir_response_parse'
                semantic_ir = parse_semantic_ir_response(response, expected_question_id=question_id)
                break
            except REQUEST_EXCEPTIONS + (RuntimeError, ValueError, TypeError) as exc:
                last_error = exc
                if attempts < config.max_attempts:
                    jitter_seed = hashlib.sha256(f'{question_id}:{attempt_index}'.encode('utf-8')).digest()
                    jitter = int.from_bytes(jitter_seed[:4], 'big') / 2 ** 32 * config.retry_jitter_sec
                    delay = config.retry_backoff_sec * 2 ** attempt_index + jitter
                    if delay > 0:
                        time.sleep(delay)
    else:
        raise ValueError(f'unsupported NExT-QA LLM IR mode: {mode!r}')
    if semantic_ir is None:
        return {'schema_version': IR_SCHEMA_VERSION, 'status': 'error', 'question_id': question_id, 'video_id': str(question_row.get('video_id') or '').strip(), 'ir_builder': PROFILE_NAME, 'model': config.model if mode == 'live' else 'nextqa-llm-ir-mock-v3', 'request_sha256': request_hash, 'attempt_count': attempts, 'api_key_slot': api_key_slot if mode == 'live' else None, 'api_key_pool_size': active_client.api_key_pool_size if active_client is not None else None, 'error_stage': last_stage, 'error_type': type(last_error).__name__ if last_error else 'UnknownError', 'error_message': str(last_error or 'LLM IR response did not validate'), 'uses_gold_answer': False, 'rule_fallback_used': False}
    return semantic_ir_to_runtime_record(question_row, semantic_ir, model=config.model if mode == 'live' else 'nextqa-llm-ir-mock-v3', request_sha256=request_hash, attempt_count=attempts, api_key_slot=api_key_slot if mode == 'live' else None, api_key_pool_size=active_client.api_key_pool_size if active_client is not None else None, response_metadata=_response_metadata(response or {}))

def _question_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    question_id = str(row.get('question_id') or '').strip()
    question = normalize_question_text(str(row.get('normalized_question') or ''))
    if not question_id.startswith('nextqa-val/') or not question:
        raise ValueError('NExT-QA LLM IR requires a stable question ID and question text')
    return (question_id, question)

def _extract_content(response: Mapping[str, Any]) -> str:
    choices = response.get('choices')
    if not isinstance(choices, list) or not choices or (not isinstance(choices[0], Mapping)):
        raise ValueError('response has no valid choices')
    message = choices[0].get('message')
    if not isinstance(message, Mapping):
        raise ValueError('response choice has no message')
    content = message.get('content')
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = [str(item.get('text') or '').strip() for item in content if isinstance(item, Mapping) and str(item.get('text') or '').strip()]
        if parts:
            return '\n'.join(parts)
    raise ValueError('response message has no text content')

def _response_metadata(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get('choices')
    finish_reason = None
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        finish_reason = choices[0].get('finish_reason')
    usage = response.get('usage')
    return {'response_id': response.get('id'), 'finish_reason': finish_reason, 'usage': dict(usage) if isinstance(usage, Mapping) else None}

def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty string')
    return ' '.join(value.strip().split())

def _optional_string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f'{field} must be a string')
    return ' '.join(value.strip().split())

def _string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any((not isinstance(item, str) for item in value)):
        raise ValueError(f'{field} must be a list of strings')
    return _dedupe_strings(value)

def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    return list(dict.fromkeys((' '.join(str(value).strip().lower().split()) for value in values if str(value).strip())))

def _temporal_anchor(question: str, anchor_text: str) -> str:
    text = f'{question} {anchor_text}'.lower()
    if 'beginning' in text or 'start of the video' in text:
        return 'video_beginning'
    if 'middle' in text:
        return 'video_middle'
    if 'end of the video' in text or 'at the end' in text:
        return 'video_end'
    return 'none'

def _mock_semantic_ir(row: Mapping[str, Any]) -> dict[str, Any]:
    question_id, question = _question_identity(row)
    if question.startswith('why '):
        intent = 'causal_why'
    elif question.startswith('how many '):
        intent = 'descriptive_count'
    elif question.startswith('how '):
        intent = 'causal_how'
    elif question.startswith('where '):
        intent = 'descriptive_location'
    elif re.search('\\b(?:before|previously|prior to)\\b', question):
        intent = 'temporal_previous'
    elif re.search('\\b(?:after|next|following)\\b', question):
        intent = 'temporal_next'
    elif re.search('\\b(?:while|during|at the same time)\\b', question):
        intent = 'temporal_concurrent'
    else:
        intent = 'descriptive_direct'
    contract = INTENT_CONTRACTS[intent]
    answer_target = sorted(contract['answer_targets'])[0]
    return {'question_id': question_id, 'intent': intent, 'answer_target': answer_target, 'actors': [], 'actions': [], 'objects': [], 'states': [], 'anchor_event': {'text': question if intent in ANCHOR_REQUIRED_INTENTS else '', 'actors': [], 'actions': [], 'objects': [], 'states': []}, 'temporal_relation': sorted(contract['temporal_relations'])[0], 'causal_relation': contract['causal_relation'], 'visual_evidence_demand': contract['visual_evidence_demand'], 'confidence': 0.0}
