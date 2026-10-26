"""OpenEQA-owned graph QA answer generation fork."""
from __future__ import annotations
import json
import os
import re
import socket
import string
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from vlkg.core.config.config import GraphQAConfig
from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options
from vlkg.core.temporal.time_adaptation import build_graph_time_adaptation
OPENAI_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'
LLM_REQUEST_EXCEPTIONS = (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ConnectionError, OSError, socket.timeout)
_RETRYABLE_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_HTTP_RETRY_BACKOFF_SECONDS = (5.0, 15.0, 30.0, 60.0)
_MAX_HTTP_RETRY_DELAY_SECONDS = 60.0
_TRANSPORT_METADATA_KEY = '_vlkg_openeqa_transport'
_API_KEY_POOL_LOCK = threading.Lock()
_API_KEY_POOL_INDEX = 0
_PROMPT_STRUCTURED_CONTEXT_KEYS = {'video_id', 'query_type', 'summary', 'filters', 'temporal_chains', 'temporal_span_requested', 'temporal_span_candidates', 'entity_aliases'}

class OpenEQAHTTPRetryError(ConnectionError):
    """Raised when the OpenEQA-only HTTP retry budget is exhausted."""

    def __init__(self, *, status_code: int | None, retry_count: int) -> None:
        self.status_code = status_code
        self.retry_count = retry_count
        status_label = 'network' if status_code is None else f'HTTP {status_code}'
        super().__init__(f'OpenEQA {status_label} remained unavailable after {retry_count} retries.')

def _configured_api_keys(env_var: str) -> list[str]:
    """Load an OpenEQA-only key pool without exposing key material."""
    pool_var = env_var if env_var.endswith('_KEYS') else f'{env_var}S'
    raw_pool = os.getenv(pool_var, '')
    values: list[str] = []
    if raw_pool:
        try:
            parsed = json.loads(raw_pool)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            values.extend((str(item).strip() for item in parsed))
        else:
            values.extend((part.strip() for part in re.split('[,;\\\\r\\\\n]+', raw_pool)))
    if not values:
        values.append(os.getenv(env_var, '').strip())
    return list(dict.fromkeys((value for value in values if value)))

def _next_api_key(env_var: str) -> str:
    global _API_KEY_POOL_INDEX
    keys = _configured_api_keys(env_var)
    if not keys:
        return ''
    with _API_KEY_POOL_LOCK:
        key = keys[_API_KEY_POOL_INDEX % len(keys)]
        _API_KEY_POOL_INDEX += 1
    return key

def _http_retry_delay_seconds(error: HTTPError, *, fallback_seconds: float) -> float:
    retry_after = error.headers.get('Retry-After') if error.headers is not None else None
    delay_seconds: float | None = None
    if retry_after is not None:
        try:
            delay_seconds = float(str(retry_after).strip())
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(str(retry_after).strip())
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                delay_seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay_seconds = None
    if delay_seconds is None:
        delay_seconds = fallback_seconds
    return max(0.0, min(delay_seconds, _MAX_HTTP_RETRY_DELAY_SECONDS))

class GraphQAResponder(Protocol):

    def answer(self, *, question: str, graph_result: dict[str, Any]) -> dict[str, Any]:
        ...

def graph_only_answer(*, question: str, graph_result: dict[str, Any]) -> dict[str, Any]:
    llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
    answer_brief = str(graph_result.get('answer_brief', '')).strip()
    uncertainty_notes = list(llm_ready_bundle.get('uncertainty_notes', []))
    answer_text = answer_brief or 'No graph-backed answer is available.'
    if uncertainty_notes:
        answer_text = f"{answer_text} Notes: {'; '.join(uncertainty_notes)}"
    return {'mode': 'graph_only', 'graph_rag_status': 'graph_only', 'provider': 'none', 'model': None, 'question': question, 'answer_text': answer_text, 'graph_rag_metadata': _build_graph_rag_metadata(graph_result), 'structured_answer': {'answer_text': answer_brief, 'used_focus_frames': [item.get('frame_index') for item in graph_result.get('retrieval_bundle', {}).get('focus_frames', [])], 'used_evidence_lines': llm_ready_bundle.get('evidence_lines', []), 'uncertainty': uncertainty_notes}}

class MockGraphQAResponder:

    def answer(self, *, question: str, graph_result: dict[str, Any]) -> dict[str, Any]:
        graph_only = graph_only_answer(question=question, graph_result=graph_result)
        structured = dict(graph_only['structured_answer'])
        answer_text = str(structured.get('answer_text', '')).strip() or 'No answer.'
        evidence_lines = list(structured.get('used_evidence_lines', []))
        if evidence_lines:
            answer_text = f'{answer_text} Evidence: {evidence_lines[0]}'
        uncertainty = list(structured.get('uncertainty', []))
        if uncertainty:
            answer_text = f'{answer_text} Caveat: {uncertainty[0]}'
        structured['answer_text'] = answer_text
        return {'mode': 'mock', 'graph_rag_status': 'mock_graph_qa', 'provider': 'mock_graph_qa', 'model': 'mock-v1', 'question': question, 'answer_text': answer_text, 'graph_rag_metadata': _build_graph_rag_metadata(graph_result), 'structured_answer': structured}

class OpenAICompatibleGraphQAResponder:

    def __init__(self, config: GraphQAConfig) -> None:
        self.config = config

    def answer(self, *, question: str, graph_result: dict[str, Any]) -> dict[str, Any]:
        api_key = _next_api_key(self.config.api_key_env_var)
        if not api_key:
            return {'mode': 'live', 'graph_rag_status': 'live_request_error', 'provider': self.config.provider, 'model': self.config.model, 'question': question, 'error': f'Missing environment variable: {self.config.api_key_env_var} or its key pool', 'answer_text': graph_result.get('answer_brief', ''), 'graph_rag_metadata': _build_graph_rag_metadata(graph_result)}
        graph_result = self._graph_result_with_unified_evidence_layer(graph_result=graph_result)
        graph_result = self._graph_result_with_llm_evidence_selection(question=question, graph_result=graph_result, api_key=api_key)
        graph_result = self._graph_result_with_atomic_answer_facts(graph_result=graph_result)
        graph_result = self._graph_result_with_answer_candidate_extraction(question=question, graph_result=graph_result, api_key=api_key)
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        payload = self._build_payload(question=question, llm_ready_bundle=llm_ready_bundle, graph_result=graph_result)
        try:
            raw_response = self._post_payload(payload, api_key=api_key)
        except LLM_REQUEST_EXCEPTIONS as exc:
            return {'mode': 'live', 'graph_rag_status': 'live_request_error', 'provider': self.config.provider, 'model': self.config.model, 'question': question, 'error': str(exc), 'answer_text': graph_result.get('answer_brief', ''), 'graph_rag_metadata': _build_graph_rag_metadata(graph_result)}
        retry_reason = _reasoning_exhausted_without_content(raw_response)
        parsed_result = self._parse_and_validate_response(raw_response, graph_result=graph_result, question=question)
        if retry_reason is None and parsed_result['normalized'] is None:
            retry_reason = str(parsed_result['error'] or 'LLM response did not validate.')
        if retry_reason is not None:
            retry_payload = self._build_payload(question=question, llm_ready_bundle=llm_ready_bundle, graph_result=graph_result, force_json_answer=True)
            try:
                retry_response = self._post_payload(retry_payload, api_key=api_key)
            except LLM_REQUEST_EXCEPTIONS as exc:
                return self._invalid_response_result(question=question, graph_result=graph_result, error=f'{retry_reason}; retry request failed: {exc}', raw_content=str(parsed_result.get('content') or ''), response_metadata=self._response_metadata(raw_response))
            retry_result = self._parse_and_validate_response(retry_response, graph_result=graph_result, question=question)
            if retry_result['normalized'] is None:
                return self._invalid_response_result(question=question, graph_result=graph_result, error=f"{retry_reason}; retry failed: {retry_result['error'] or 'LLM response did not validate.'}", raw_content=str(retry_result.get('content') or ''), response_metadata=self._response_metadata(retry_response))
            raw_response = retry_response
            parsed_result = retry_result
        normalized = parsed_result['normalized']
        if normalized is None:
            return self._invalid_response_result(question=question, graph_result=graph_result, error=str(parsed_result['error'] or 'LLM response did not validate.'), raw_content=str(parsed_result.get('content') or ''), response_metadata=self._response_metadata(raw_response))
        return {'mode': 'live', 'graph_rag_status': 'live_validated', 'provider': self.config.provider, 'model': self.config.model, 'question': question, 'answer_text': normalized.get('answer_text', graph_result.get('answer_brief', '')), 'graph_rag_metadata': _build_graph_rag_metadata(graph_result), 'structured_answer': normalized, 'response_metadata': self._response_metadata(raw_response)}

    def _parse_and_validate_response(self, raw_response: dict[str, Any], *, graph_result: dict[str, Any], question: str) -> dict[str, Any]:
        content = self._extract_content(raw_response)
        if content is None:
            return {'normalized': None, 'error': 'Empty LLM response content.', 'content': ''}
        parsed = self._parse_json_content(content)
        if parsed is None:
            return {'normalized': None, 'error': 'LLM response is not valid JSON.', 'content': content}
        normalized = self._normalize_structured_answer(parsed=parsed, graph_result=graph_result, question=question)
        if normalized is None:
            return {'normalized': None, 'error': 'LLM response failed graph QA validation.', 'content': content}
        return {'normalized': normalized, 'error': None, 'content': content}

    def _post_payload(self, payload: dict[str, Any], *, api_key: str) -> dict[str, Any]:
        endpoint = urljoin(self.config.api_base_url.rstrip('/') + '/', self.config.endpoint_path.lstrip('/'))
        request_data = json.dumps(payload).encode('utf-8')
        retry_count = 0
        current_api_key = api_key
        while True:
            request = Request(endpoint, data=request_data, headers={'Authorization': f'Bearer {current_api_key}', 'Content-Type': 'application/json', 'User-Agent': OPENAI_BROWSER_USER_AGENT}, method='POST')
            try:
                with urlopen(request, timeout=self.config.timeout_sec) as response:
                    raw_response = json.loads(response.read().decode('utf-8'))
                break
            except HTTPError as exc:
                if exc.code not in _RETRYABLE_HTTP_STATUS_CODES:
                    raise
                if retry_count >= len(_HTTP_RETRY_BACKOFF_SECONDS):
                    exc.close()
                    raise OpenEQAHTTPRetryError(status_code=exc.code, retry_count=retry_count) from exc
                delay_seconds = _http_retry_delay_seconds(exc, fallback_seconds=_HTTP_RETRY_BACKOFF_SECONDS[retry_count])
                retry_count += 1
                exc.close()
                current_api_key = _next_api_key(self.config.api_key_env_var)
                time.sleep(delay_seconds)
            except (TimeoutError, URLError, OSError) as exc:
                if retry_count >= len(_HTTP_RETRY_BACKOFF_SECONDS):
                    raise OpenEQAHTTPRetryError(status_code=None, retry_count=retry_count) from exc
                delay_seconds = _HTTP_RETRY_BACKOFF_SECONDS[retry_count]
                retry_count += 1
                current_api_key = _next_api_key(self.config.api_key_env_var)
                time.sleep(delay_seconds)
        if not isinstance(raw_response, dict):
            raise json.JSONDecodeError('Response JSON is not an object', str(raw_response), 0)
        raw_response[_TRANSPORT_METADATA_KEY] = {'http_retry_count': retry_count}
        return raw_response

    def _graph_result_with_llm_evidence_selection(self, *, question: str, graph_result: dict[str, Any], api_key: str) -> dict[str, Any]:
        if not self.config.evidence_rerank_enabled:
            return graph_result
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        if not isinstance(llm_ready_bundle, dict):
            return graph_result
        candidates = _build_llm_evidence_selection_candidates(llm_ready_bundle, graph_result=graph_result, question=question, max_candidates=self.config.evidence_rerank_max_candidates, balanced=self.config.experimental_balanced_evidence_candidates, answerable_candidate_packs=self.config.experimental_answerable_candidate_packs, answerable_candidate_pack_max_items=self.config.answerable_candidate_pack_max_items)
        if not candidates:
            return graph_result
        selection = self._select_evidence_with_llm(question=question, llm_ready_bundle=llm_ready_bundle, candidates=candidates, api_key=api_key)
        if not selection:
            return graph_result
        selected_ids = _ordered_unique_strings(selection.get('selected_ids', []))[:max(1, self.config.evidence_rerank_max_selected)]
        selected = [candidate for candidate in candidates if candidate['id'] in set(selected_ids)]
        if not selected:
            return graph_result
        answer_cue = _repair_response_text(str(selection.get('answer_cue') or '').strip())
        insufficiency = _repair_response_text(str(selection.get('insufficiency_reason') or '').strip())
        llm_selection = {'contract': 'generic_llm_evidence_selection_v1', 'available': True, 'selected_ids': [item['id'] for item in selected], 'selected_evidence': selected, 'answer_cue': answer_cue, 'insufficient_evidence': bool(selection.get('insufficient_evidence', False)), 'insufficiency_reason': insufficiency, 'rationale': _repair_response_text(str(selection.get('rationale') or '').strip()), 'candidate_count': len(candidates), 'model': self.config.model}
        updated_bundle = _bundle_with_llm_evidence_selection(llm_ready_bundle, llm_selection=llm_selection)
        graph_result['llm_ready_bundle'] = updated_bundle
        return graph_result

    def _graph_result_with_atomic_answer_facts(self, *, graph_result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.experimental_atomic_answer_facts:
            return graph_result
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        if not isinstance(llm_ready_bundle, dict):
            return graph_result
        updated_bundle = _bundle_with_atomic_answer_facts(llm_ready_bundle, max_items=self.config.atomic_answer_fact_max_items)
        graph_result['llm_ready_bundle'] = updated_bundle
        return graph_result

    def _graph_result_with_answer_candidate_extraction(self, *, question: str, graph_result: dict[str, Any], api_key: str) -> dict[str, Any]:
        if not self.config.answer_candidate_extraction_enabled:
            return graph_result
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        if not isinstance(llm_ready_bundle, dict):
            return graph_result
        evidence_candidates = _build_llm_evidence_selection_candidates(llm_ready_bundle, graph_result=graph_result, question=question, max_candidates=self.config.evidence_rerank_max_candidates, balanced=self.config.experimental_balanced_evidence_candidates, answerable_candidate_packs=self.config.experimental_answerable_candidate_packs, answerable_candidate_pack_max_items=self.config.answerable_candidate_pack_max_items)
        if not evidence_candidates:
            return graph_result
        extraction = self._extract_answer_candidates_with_llm(question=question, llm_ready_bundle=llm_ready_bundle, evidence_candidates=evidence_candidates, api_key=api_key)
        if not extraction:
            return graph_result
        updated_bundle = _bundle_with_answer_candidate_extraction(llm_ready_bundle, extraction=extraction, max_candidates=self.config.answer_candidate_extraction_max_candidates)
        graph_result['llm_ready_bundle'] = updated_bundle
        return graph_result

    def _graph_result_with_unified_evidence_layer(self, *, graph_result: dict[str, Any]) -> dict[str, Any]:
        if not self.config.experimental_unified_evidence_layer:
            return graph_result
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        if not isinstance(llm_ready_bundle, dict):
            return graph_result
        updated_bundle = _bundle_with_unified_evidence_layer(llm_ready_bundle, graph_result=graph_result, max_items=self.config.unified_evidence_max_items, graph_quota=self.config.unified_evidence_graph_quota, visual_quota=self.config.unified_evidence_visual_quota, frame_quota=self.config.unified_evidence_frame_quota)
        graph_result['llm_ready_bundle'] = updated_bundle
        return graph_result

    def _extract_answer_candidates_with_llm(self, *, question: str, llm_ready_bundle: dict[str, Any], evidence_candidates: list[dict[str, Any]], api_key: str) -> dict[str, Any] | None:
        payload = self._build_answer_candidate_extraction_payload(question=question, llm_ready_bundle=llm_ready_bundle, evidence_candidates=evidence_candidates)
        try:
            raw_response = self._post_payload(payload, api_key=api_key)
        except LLM_REQUEST_EXCEPTIONS:
            return None
        content = self._extract_content(raw_response)
        if content is None:
            return None
        parsed = self._parse_json_content(content)
        if parsed is None:
            return None
        return _normalize_answer_candidate_extraction(parsed, allowed_ids={item['id'] for item in evidence_candidates}, max_candidates=self.config.answer_candidate_extraction_max_candidates, model=self.config.model)

    def _build_answer_candidate_extraction_payload(self, *, question: str, llm_ready_bundle: dict[str, Any], evidence_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        question_ir = llm_ready_bundle.get('question_ir', {})
        llm_evidence_selection = llm_ready_bundle.get('llm_evidence_selection', {})
        payload: dict[str, Any] = {'model': self.config.model, 'temperature': self.config.answer_candidate_extraction_temperature, 'max_tokens': self.config.answer_candidate_extraction_max_tokens, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': 'You are a generic answer-candidate extraction layer for graph-backed video QA. Work for every answer type: binary yes/no, descriptive text, time, duration, count, attributes, objects, relations, actions, and events. Use only the supplied graph evidence candidates, the user question, and question_ir. Do not use answer keys, hidden labels, dataset-specific assumptions, or outside world knowledge. Extract concise candidate answers only when the evidence directly supports or reasonably implies them. Preserve all important question factors, constraints, target entities, actions, attributes, and time windows when judging support. If evidence is missing, conflicting, or only weak context, mark insufficient_evidence or low confidence instead of forcing an answer. Return strict JSON only.'}, {'role': 'user', 'content': json.dumps({'task': 'extract_generic_answer_candidates', 'question': question, 'question_ir': question_ir if isinstance(question_ir, dict) else {}, 'llm_evidence_selection': llm_evidence_selection if isinstance(llm_evidence_selection, dict) else {}, 'extraction_policy': {'generic_all_question_types': True, 'use_only_candidate_evidence': True, 'do_not_use_gold_answer': True, 'preserve_question_factors': True, 'allow_reasonable_graph_supported_inference': True, 'mark_indirect_or_weak_support_low_confidence': True, 'do_not_force_answer_when_binding_is_unclear': True, 'return_support_ids_from_candidates_only': True, 'max_candidates': self.config.answer_candidate_extraction_max_candidates}, 'allowed_answer_types': ['binary', 'text', 'time', 'duration', 'count', 'attribute', 'object', 'relation', 'event', 'unknown'], 'required_json_schema': {'available': 'boolean', 'candidates': [{'answer': 'short candidate answer', 'answer_type': 'binary|text|time|duration|count|attribute|object|relation|event|unknown', 'support_ids': ['evidence_line:1'], 'support_text': 'brief support summary', 'confidence': 'high|medium|low', 'reasoning': 'brief graph-grounded reason'}], 'insufficient_evidence': 'boolean', 'insufficiency_reason': 'string'}, 'evidence_candidates': evidence_candidates}, ensure_ascii=False, indent=2)}]}
        apply_reasoning_options(payload, reasoning_effort=self.config.reasoning_effort, thinking_enabled=False)
        return payload

    def _select_evidence_with_llm(self, *, question: str, llm_ready_bundle: dict[str, Any], candidates: list[dict[str, Any]], api_key: str) -> dict[str, Any] | None:
        payload = self._build_evidence_selection_payload(question=question, llm_ready_bundle=llm_ready_bundle, candidates=candidates)
        try:
            raw_response = self._post_payload(payload, api_key=api_key)
        except LLM_REQUEST_EXCEPTIONS:
            return None
        content = self._extract_content(raw_response)
        if content is None:
            return None
        parsed = self._parse_json_content(content)
        if parsed is None:
            return None
        return _normalize_llm_evidence_selection(parsed, allowed_ids={item['id'] for item in candidates})

    def _build_evidence_selection_payload(self, *, question: str, llm_ready_bundle: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        question_ir = llm_ready_bundle.get('question_ir', {})
        payload: dict[str, Any] = {'model': self.config.model, 'temperature': self.config.evidence_rerank_temperature, 'max_tokens': self.config.evidence_rerank_max_tokens, 'response_format': {'type': 'json_object'}, 'messages': [{'role': 'system', 'content': "You are a generic evidence selection layer for graph-backed video QA. Select only the evidence items that directly help answer the user's question. Use the question wording and any question_ir contract, but do not use answer keys, dataset-specific assumptions, outside world knowledge, or hidden labels. Do not prefer rank 1 automatically; prefer evidence that best satisfies the target, requested property/action/event, constraints, and time window. If no candidate directly supports an answer, mark insufficient_evidence true. Return strict JSON only with keys selected_ids, answer_cue, insufficient_evidence, insufficiency_reason, rationale."}, {'role': 'user', 'content': json.dumps({'task': 'select_answer_supporting_evidence', 'question': question, 'question_ir': question_ir if isinstance(question_ir, dict) else {}, 'selection_policy': {'generic': True, 'use_only_candidate_evidence': True, 'do_not_use_gold_answer': True, 'prefer_direct_answer_support': True, 'prefer_bound_evidence_over_context': True, 'return_ids_from_candidates_only': True, 'answer_cue_should_be_short': True}, 'candidates': candidates}, ensure_ascii=False, indent=2)}]}
        apply_reasoning_options(payload, reasoning_effort=self.config.reasoning_effort, thinking_enabled=False)
        return payload

    def _build_messages(self, *, question: str, llm_ready_bundle: dict[str, Any], graph_result: dict[str, Any]) -> list[dict[str, Any]]:
        evidence_lines = list(llm_ready_bundle.get('evidence_lines', []))[:self.config.max_evidence_lines]
        focus_frame_lines = list(llm_ready_bundle.get('focus_frame_lines', []))[:self.config.max_focus_frames]
        scene_lines = list(llm_ready_bundle.get('scene_lines', []))[:self.config.max_scene_lines]
        temporal_chain_lines = list(llm_ready_bundle.get('temporal_chain_lines', []))[:self.config.max_evidence_lines]
        temporal_span_candidate_lines = list(llm_ready_bundle.get('temporal_span_candidate_lines', []))[:self.config.max_evidence_lines]
        evidence_aggregation_lines = list(llm_ready_bundle.get('evidence_aggregation_lines', []))[:self.config.max_evidence_lines]
        evidence_aggregation = llm_ready_bundle.get('evidence_aggregation', {})
        candidate_facts = list(evidence_aggregation.get('candidate_facts', []))[:self.config.max_evidence_lines] if isinstance(evidence_aggregation, dict) else []
        question_ir = llm_ready_bundle.get('question_ir', {})
        question_ir_execution = llm_ready_bundle.get('question_ir_execution', {})
        question_ir_execution_lines = list(question_ir_execution.get('execution_lines', []))[:self.config.max_evidence_lines] if isinstance(question_ir_execution, dict) else []
        llm_evidence_selection = llm_ready_bundle.get('llm_evidence_selection', {})
        llm_selected_evidence_lines = list(llm_ready_bundle.get('llm_selected_evidence_lines', []))[:self.config.evidence_rerank_max_selected]
        answer_candidate_extraction = llm_ready_bundle.get('answer_candidate_extraction', {})
        answer_candidate_lines = list(llm_ready_bundle.get('answer_candidate_lines', []))[:self.config.answer_candidate_extraction_max_candidates]
        atomic_answer_facts = llm_ready_bundle.get('atomic_answer_facts', {})
        atomic_answer_fact_lines = list(llm_ready_bundle.get('atomic_answer_fact_lines', []))[:self.config.atomic_answer_fact_max_items]
        visual_faiss_evidence_lines = list(llm_ready_bundle.get('visual_faiss_evidence_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_ger_evidence_lines = list(llm_ready_bundle.get('visual_faiss_ger_evidence_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_ger_bound_fact_lines = list(llm_ready_bundle.get('visual_faiss_ger_bound_fact_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_unified_candidate_lines = list(llm_ready_bundle.get('visual_faiss_unified_candidate_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_object_local_fact_lines = list(llm_ready_bundle.get('visual_faiss_object_local_fact_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_local_subgraph_fact_lines = list(llm_ready_bundle.get('visual_faiss_local_subgraph_fact_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_role_aware_lines = list(llm_ready_bundle.get('visual_faiss_role_aware_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_role_aware_aux_lines = list(llm_ready_bundle.get('visual_faiss_role_aware_aux_lines', []))[:self.config.max_evidence_lines]
        visual_faiss_evidence_layer_lines = list(llm_ready_bundle.get('visual_faiss_evidence_layer_lines', []))[:self.config.max_evidence_lines]
        visual_entity_candidate_pack_lines = list(llm_ready_bundle.get('visual_entity_candidate_pack_lines', []))[:self.config.max_evidence_lines]
        anchor_bound_identity_candidate_lines = list(llm_ready_bundle.get('question_route_anchor_bound_identity_lines', []))[:self.config.max_evidence_lines]
        specialized_evidence_lines = list(llm_ready_bundle.get('question_route_specialized_evidence_lines', []))[:self.config.max_evidence_lines]
        frame_faiss_evidence_lines = list(llm_ready_bundle.get('frame_faiss_evidence_lines', []))[:self.config.max_evidence_lines]
        merged_frame_faiss_evidence_lines = list(llm_ready_bundle.get('merged_frame_faiss_evidence_lines', []))[:self.config.max_evidence_lines]
        evidence_demand = llm_ready_bundle.get('evidence_demand', {}) if isinstance(llm_ready_bundle.get('evidence_demand'), dict) else {}
        visual_override_contract = llm_ready_bundle.get('visual_override_contract', {}) if isinstance(llm_ready_bundle.get('visual_override_contract'), dict) else {}
        guarded_visual_contract = bool(visual_override_contract.get('contract') in {'openeqa_visual_override_guard.v4', 'openeqa_visual_override_guard.v4c', 'openeqa_visual_override_guard.v4d'})
        guarded_v4d_contract = bool(visual_override_contract.get('contract') == 'openeqa_visual_override_guard.v4d')
        guarded_v4d_component_preference = bool(guarded_v4d_contract and visual_override_contract.get('open_identity_component_suppression'))
        visual_evidence_available = bool(llm_ready_bundle.get('visual_evidence_contract') and (not guarded_visual_contract or visual_entity_candidate_pack_lines or anchor_bound_identity_candidate_lines or specialized_evidence_lines))
        guarded_visual_override = bool(guarded_visual_contract and visual_evidence_available)
        guarded_identity_support = bool(guarded_visual_override and visual_override_contract.get('can_support_object_identity'))
        allows_common_world_knowledge = bool(visual_evidence_available and evidence_demand.get('allows_common_world_knowledge'))
        context_text = _compact_llm_context_text_for_prompt(llm_ready_bundle, max_evidence_lines=self.config.max_evidence_lines, max_focus_frames=self.config.max_focus_frames, max_scene_lines=self.config.max_scene_lines)
        structured_graph_context = _compact_structured_graph_context_for_prompt(llm_ready_bundle.get('structured_graph_context', {}))
        citation_requirements = self._build_citation_requirements(graph_result=graph_result, llm_ready_bundle=llm_ready_bundle, focus_frame_lines=focus_frame_lines)
        time_adaptation = build_graph_time_adaptation(question=question, graph_result=graph_result)
        is_position_question = _graph_result_is_position_question(question=question, graph_result=graph_result)
        prompt_payload = {'task': 'graph_backed_video_qa', 'dataset_context': {'viewpoint': 'first-person robot/ego-camera navigation video', 'pronoun_policy': "In the question, 'I', 'you', and 'we' refer to the robot/ego camera whose observations were converted into the graph.", 'observation_scope': 'The graph describes what the robot camera saw and the scenes the robot moved through.', 'context_is_not_evidence': 'Use this background only to interpret the question wording. It is not visual evidence and cannot prove that an object, state, or action occurred.'}, 'question': question, 'instructions': {'use_only_graph_evidence': True, 'interpret_first_person_pronouns_as_robot_ego_camera': True, 'treat_see_saw_as_robot_camera_visual_observation': True, 'treat_right_now_as_question_time_robot_state': True, 'state_uncertainty_when_needed': True, 'avoid_inventing_frames_or_events': True, 'prefer_answer_brief_as_core_answer': True, 'used_focus_frames_must_be_subset_of_input': True, 'used_evidence_ranks_must_reference_graph_summary_indices': True, 'if_graph_evidence_is_insufficient_say_so': True, 'cite_multiple_evidence_lines_when_available': True, 'prefer_covering_more_than_one_time_anchor_for_temporal_queries': True, 'prefer_used_evidence_ranks_over_full_lines': True, 'do_not_copy_full_evidence_lines_into_output': True, 'prefer_entity_aliases_over_raw_entity_ids': True, 'avoid_irrelevant_evidence_that_conflicts_with_answer': True, 'use_temporal_chains_for_ordered_time_reasoning': True, 'use_temporal_span_candidates_when_available': True, 'for_when_how_long_before_after_between_enter_leave_questions_prioritize_temporal_chains': True, 'infer_intervals_from_ordered_chains_when_supported': True, 'separate_observed_boundaries_from_inferred_duration': True, 'question_metadata_and_format_hints_are_not_visual_evidence': True, 'do_not_use_answer_keys_or_external_world_knowledge': True, 'prioritize_direct_evidence_before_weak_context': True, 'bind_attributes_to_the_queried_entity': True, 'do_not_transfer_attributes_between_nearby_objects': True, 'if_target_exists_but_attribute_is_missing_report_the_gap': True, 'treat_generic_spatial_relations_as_weak_context': True, 'use_time_adaptation_candidates_for_relative_time_formatting': True, 'do_not_manually_recalculate_relative_time_values': True, 'use_question_ir_as_answer_contract_when_available': True, 'respect_ir_operation_aggregation_rendering_and_evidence_requirements': True, 'use_evidence_aggregation_before_final_yes_no_or_attribute_choice': True, 'binary_yes_no_answer_text_must_be_exact_label_only': True, 'use_llm_evidence_selection_when_available': True, 'use_answer_candidates_when_available': True, 'use_atomic_answer_facts_when_available': True, 'use_visual_faiss_evidence_when_available': True}, 'answering_policy': {'evidence_priority': ['direct state or attribute evidence bound to the queried entity', 'direct entity label, canonical label, semantic alias, or salient scene label', 'ordered scene chain, scene boundary, or temporal composition evidence', 'relation episodes whose subject or object matches the queried target', 'generic spatial or co-occurrence relations, used only as weak context'], 'temporal_axis_policy': ['For time, duration, before/after, between, enter, and leave questions, reason along the ordered scene chain first.', 'When graph_summary.temporal_span_candidate_lines is available, inspect those candidate start/end frame spans as generic entity/state interval hypotheses before choosing a duration.', 'A temporal_span_candidate is not a final answer by itself; verify that its target hits, entity/state binding, source kind, and warning fit the question before using it.', 'Prefer entity, stable-identity, entity-attribute, or episode spans over scene-level background spans and zero-length transition boundaries when both are available.', 'Use earlier-to-later scene order as the primary axis before summarizing isolated entity matches.', 'If evidence shows a state before a boundary and a different state after it, infer only the interval supported by the observed boundary frames.', 'Label approximate intervals, enter/exit boundaries, and persistence claims as inferred when no direct event node exists.', 'When time_adaptation.relative_time_candidates contains the chosen event timestamp, use its display_text instead of doing arithmetic in natural language.', 'If no time_adaptation candidate corresponds to the chosen graph event, preserve the graph timestamp/frame and state that the relative time is not normalized.'], 'attribute_binding_policy': ['Attributes, states, and descriptions belong only to the entity or label they are attached to in graph evidence.', 'Do not transfer an attribute from a nearby object, scene neighbor, or relation endpoint to the queried target.', 'If the target entity or scene is found but the requested attribute is absent, answer the located target and explicitly state that the attribute is not directly supported.'], 'metadata_policy': ['Dataset context can disambiguate viewpoint and pronouns only; it is not observation evidence.', 'Question time metadata can support timestamp arithmetic only; it is not observation evidence.', 'Answer-format hints can shape wording only; they are not evidence for what happened.', 'Use no answer keys, hidden labels, dataset-specific assumptions, or outside world knowledge.'], 'question_ir_policy': ['When question_ir is provided, treat operation, target entities, attribute dimension, temporal relation, hard constraints, aggregation, rendering, evidence requirements, and graph information needs as the answer contract.', 'Treat every question_ir.hard_constraints item as a question factor that must be considered and preserved when choosing evidence; do not drop it, but do allow reasonable graph-supported inference when the evidence is indirect.', 'Use question_ir_execution.semantic_event_candidates as graph-derived candidate events only when they are supported by scene chains or evidence lines.', 'For aggregation=multiple or rendering that asks for a set/list, cite and synthesize all directly bound matching evidence instead of returning only the top-ranked item.', "For answer_type=binary or rendering=yes_no, answer_text must be exactly 'yes' or exactly 'no' with no explanation, punctuation, or additional words. Answer 'yes' when the retrieved graph evidence directly or reasonably inferentially supports the question factors; answer 'no' when evidence contradicts them or no meaningful support is present. Put uncertainty or supporting evidence only in uncertainty, used_focus_frames, used_evidence_ranks, or used_time_anchor_ids.", 'Use graph_information_needs as diagnostic guidance, not as a veto list: mention missing ideal evidence when relevant, but do not reject a reasonable answer that is supported by other retrieved graph evidence.'], 'evidence_aggregation_policy': ['Use graph_summary.evidence_aggregation_lines to compare all retrieved evidence grouped by entity, scene, time range, question-factor hits, and IR requirement hits before selecting the final answer.', 'Use graph_summary.candidate_facts as compact graph-derived fact candidates; prefer facts with stronger binding_strength when they are relevant to the question.', 'Prefer candidate groups that cover more target terms and hard_constraint_hits, while still considering indirect evidence, aliases, interactions, and scene context that reasonably satisfy the question.', 'For attribute questions with multiple candidate entities, prefer the group that best matches the target terms and evidence requirements; do not blindly use rank 1 if another group better satisfies the target.', 'When evidence_aggregation_lines list a higher-ranked group with target and requirement hits, base attribute and descriptive answers on that group unless another group has strictly stronger direct binding.', 'If the top aggregation group contains a target-bound attribute fact, use that value even when raw evidence line rank 1 names a different candidate.', 'For binary questions, distinguish direct support, weak contextual support, and missing required action/state evidence using the grouped facts.', 'For temporal questions, use grouped temporal windows together with ordered temporal chains before choosing the event frame.'], 'llm_evidence_selection_policy': ['If graph_summary.llm_evidence_selection is available, treat it as a generic preselection of the retrieved candidates that most directly answer the question.', 'Use selected evidence as the first evidence to inspect, while still refusing unsupported conclusions when the selected evidence is weak or marked insufficient.', 'The selection is not a gold label and not external evidence; it only ranks the supplied graph evidence by directness.'], 'answer_candidate_policy': ['If graph_summary.answer_candidate_extraction is available, treat it as a generic extraction aid over the supplied graph evidence, not as a gold label or external evidence.', 'Inspect answer_candidate_lines before writing the final answer, but verify every candidate against graph_summary evidence, selected evidence, aggregation lines, temporal chains, and question_ir.', 'Prefer high- or medium-confidence candidates when their support covers the question factors; keep uncertainty or reject the candidate when support is weak, conflicting, or bound to the wrong entity/event.', 'Apply this policy uniformly to binary, descriptive, temporal, duration, count, attribute, relation, action, and event questions.'], 'atomic_answer_fact_policy': ['If graph_summary.atomic_answer_fact_lines is available, treat these lines as generic summaries mechanically derived from the supplied evidence, not as gold labels, external captions, or new facts.', 'Use atomic facts to inspect subject, predicate, object or attribute, time, support source, and binding hints before choosing between mixed evidence lines.', 'Prefer atomic facts whose subject/object/attribute binding is supported by the referenced evidence; reject or downweight facts whose support is weak, missing, or bound to the wrong target.', 'Apply this policy uniformly to all question types; do not assume the atomic fact is correct if stronger raw graph evidence contradicts it.'], 'visual_faiss_policy': ['If graph_summary visual/frame/merged-frame FAISS evidence lines are available, treat them as supplemental visual similarity evidence derived from the current sequence.', 'If graph_summary.visual_faiss_unified_candidate_lines is available, inspect these unified graph+FAISS candidates before raw visual FAISS lines; they combine object recall, frame-level HSGG object attributes/relations, and nearby canonical graph support.', "If graph_summary.visual_faiss_evidence_layer_lines is available, inspect these object-observation local packs before broader visual FAISS context; they package the recalled observation's HSGG attributes, HSGG relations, bbox neighbors, image anchor, and time anchor without using gold answers or question-type branches.", 'If graph_summary.visual_faiss_local_subgraph_fact_lines is available, inspect these entity-local subgraph facts before object-local or broader GER-like context; they map FAISS hits back to canonical entity ids and summarize entity-bound attributes, relations, time range, visual anchor, and bounded neighbors.', 'If graph_summary.visual_entity_candidate_pack_lines is available, inspect these candidate packs first; each pack is a FAISS-recalled entity/tracklet candidate with direct attributes, direct relations, constraints, time span, visual anchors, and neighbor context kept in separate fields.', 'For visual_entity_candidate_pack evidence, direct_attributes and direct_relations belong to the anchor entity; neighbor_context is only context and must not override or transfer attributes to the anchor entity.', 'If graph_summary.visual_faiss_role_aware_lines is available, inspect these target-bound visual FAISS candidate subjects before broader context; target_entities from question_ir define normal answer subjects, while event, spatial, and information-need hits are constraint_context attachments rather than independent answer subjects.', 'If graph_summary.visual_faiss_role_aware_aux_lines is available, treat these as auxiliary role checks over visual FAISS hits. They can help compare target and constraint coverage, but they do not replace GER-like, object-local, graph, or temporal evidence.', 'If graph_summary.visual_faiss_object_local_fact_lines is available, inspect these object-local facts before broader GER-like context; they summarize direct attributes, direct relations, same-frame anchor/bbox context, and direct neighbor attributes for each recalled object.', 'If graph_summary.visual_faiss_ger_bound_fact_lines is available, inspect these generic binding facts before raw visual FAISS lines; they summarize which visual object candidates satisfy question terms, constraints, attributes, relations, and nearby graph context.', 'When visual FAISS facts mark HSGG attributes as direct_object_attribute or neighbor_object_attribute, treat direct_object_attribute as a robot-view description of the named owner object at that frame/bbox; use neighbor_object_attribute only as neighbor/context evidence and do not transfer it to the target object.', 'Use visual similarity to notice repeated appearances or label instability, but verify the final answer against the graph evidence, focus frames, scene context, temporal chains, and question_ir.', 'Do not treat visual FAISS evidence as a gold answer, an external caption, or proof outside the selected video sequence.', 'When visual similarity conflicts with stronger graph evidence, prefer the stronger graph evidence and state uncertainty if needed.']}, 'time_formatting_policy': ['The time_adaptation block is computed from retrieved graph evidence and question current time only.', 'For minutes-ago or duration wording, select the graph event first from evidence, then quote the matching time_anchor display_text.', 'Do not invent or manually recompute relative-time numbers in prose when a matching time_anchor exists.', 'If the chosen evidence lacks a matching time_anchor, keep the frame/timestamp and add uncertainty instead of guessing minutes ago.'], 'time_adaptation': time_adaptation, 'citation_requirements': citation_requirements, 'graph_contract': {'allowed_output_keys': ['answer_text', 'used_focus_frames', 'used_evidence_ranks', 'used_time_anchor_ids', 'uncertainty'], 'forbidden_behaviors': ['inventing unseen entities', 'inventing unseen timestamps or frame indices', 'claiming certainty when graph evidence is ambiguous', 'using evidence ranks outside the provided evidence list', 'copying full evidence lines into output when ranks are sufficient', 'using outside knowledge, answer keys, or dataset-specific assumptions', 'treating question metadata or format hints as observation evidence', 'transferring attributes from nearby unrelated objects to the queried target', 'treating generic spatial relations as decisive evidence without target endpoint support', 'manually recalculating relative time when a matching time_adaptation candidate is available', 'using evaluation labels or hidden answer timestamps']}, 'graph_summary': {'answer_brief': llm_ready_bundle.get('answer_brief', ''), 'llm_evidence_selection': llm_evidence_selection if isinstance(llm_evidence_selection, dict) else {}, 'llm_selected_evidence_lines': llm_selected_evidence_lines, 'answer_candidate_extraction': answer_candidate_extraction if isinstance(answer_candidate_extraction, dict) else {}, 'answer_candidate_lines': answer_candidate_lines, 'atomic_answer_facts': atomic_answer_facts if isinstance(atomic_answer_facts, dict) else {}, 'atomic_answer_fact_lines': atomic_answer_fact_lines, 'visual_faiss_evidence_lines': visual_faiss_evidence_lines, 'visual_entity_candidate_pack_lines': visual_entity_candidate_pack_lines, 'visual_faiss_unified_candidate_lines': visual_faiss_unified_candidate_lines, 'visual_faiss_evidence_layer_lines': visual_faiss_evidence_layer_lines, 'visual_faiss_local_subgraph_fact_lines': visual_faiss_local_subgraph_fact_lines, 'visual_faiss_role_aware_lines': visual_faiss_role_aware_lines, 'visual_faiss_role_aware_aux_lines': visual_faiss_role_aware_aux_lines, 'visual_faiss_object_local_fact_lines': visual_faiss_object_local_fact_lines, 'visual_faiss_ger_bound_fact_lines': visual_faiss_ger_bound_fact_lines, 'visual_faiss_ger_evidence_lines': visual_faiss_ger_evidence_lines, 'frame_faiss_evidence_lines': frame_faiss_evidence_lines, 'merged_frame_faiss_evidence_lines': merged_frame_faiss_evidence_lines, 'evidence_aggregation_lines': evidence_aggregation_lines, 'candidate_facts': candidate_facts, 'evidence_lines': evidence_lines, 'focus_frame_lines': focus_frame_lines, 'scene_lines': scene_lines, 'temporal_chain_lines': temporal_chain_lines, 'temporal_span_candidate_lines': temporal_span_candidate_lines, 'question_ir_execution_lines': question_ir_execution_lines, 'match_highlights': llm_ready_bundle.get('match_highlights', []), 'uncertainty_notes': llm_ready_bundle.get('uncertainty_notes', [])}, 'question_ir': question_ir if isinstance(question_ir, dict) else {}, 'question_ir_execution': question_ir_execution if isinstance(question_ir_execution, dict) else {}, 'structured_graph_context': structured_graph_context, 'graph_context': context_text}
        if is_position_question:
            prompt_payload['instructions']['for_position_place_questions_return_event_time_not_coordinates'] = True
            prompt_payload['answering_policy']['position_rendering_policy'] = ['When a question asks where an observed target/event was seen but no coordinate evidence is supplied, answer by identifying the graph-backed scene, frame, timestamp, and target evidence.', 'Do not invent x/y/z, GPS, map, or object 3D coordinates from semantic graph evidence.', 'If the caller describes an ego-pose adapter, treat coordinates as an external rendering of the selected event time, not as graph memory content.']
            prompt_payload['graph_contract']['forbidden_behaviors'].append('inventing coordinates for position questions without coordinate evidence')
        if anchor_bound_identity_candidate_lines:
            prompt_payload['graph_summary']['question_route_anchor_bound_identity_lines'] = anchor_bound_identity_candidate_lines
            prompt_payload['instructions']['use_anchor_bound_identity_candidates_when_available'] = True
            prompt_payload['answering_policy']['anchor_bound_identity_policy'] = ['Treat each anchor-bound identity candidate as supplemental visual evidence for naming the answer entity.', 'Use it only when relation_verified=true and the target, relation, and anchor roles match the question.', 'Do not treat candidate order as unique-answer proof and do not discard stronger frozen graph or visual evidence.']
        if specialized_evidence_lines:
            prompt_payload['graph_summary']['question_route_specialized_evidence_lines'] = specialized_evidence_lines
            prompt_payload['instructions']['use_question_route_specialized_evidence_when_available'] = True
            prompt_payload['answering_policy']['specialized_evidence_policy'] = ['Treat question-route specialized lines as supplemental observations or aggregation guidance for the persisted semantic IR route.', 'Keep frozen graph and visual evidence available; do not discard stronger contradictory evidence.', 'Use specialized evidence only when its target, owner, state, relation, or attribute binding matches the question.']
        if visual_evidence_available:
            prompt_payload['evidence_demand'] = evidence_demand
            prompt_payload['instructions']['use_only_graph_evidence'] = False
            prompt_payload['instructions']['use_only_supplied_graph_and_visual_evidence'] = True
            prompt_payload['instructions']['visual_evidence_is_observation_evidence'] = True
            prompt_payload['instructions']['unbound_visual_observations_are_not_graph_facts'] = True
        if guarded_visual_override:
            prompt_payload['visual_override_contract'] = visual_override_contract
            prompt_payload['instructions']['graph_evidence_is_default_authority'] = True
            prompt_payload['instructions']['visual_candidates_without_route_proof_are_audit_only'] = True
            prompt_payload['instructions']['visual_evidence_may_override_graph_only_when_can_override_graph_is_true'] = True
            prompt_payload['instructions']['visual_crop_presence_does_not_prove_state_or_absence'] = True
            prompt_payload['instructions']['attribute_answers_require_owner_and_dimension_binding'] = True
            prompt_payload['graph_contract']['forbidden_behaviors'].extend(['using audit-only visual candidates as answer evidence', 'overriding canonical graph evidence with an unverified bbox relation', 'inferring object state or absence from crop similarity alone', 'answering one attribute dimension with a different dimension'])
        if guarded_identity_support:
            prompt_payload['instructions']['visual_identity_candidates_may_support_object_identity_only'] = True
            prompt_payload['instructions']['visual_identity_support_is_not_spatial_relation_proof'] = True
            prompt_payload['instructions']['unbound_identity_support_remains_non_graph_observation'] = True
            prompt_payload['answering_policy']['visual_identity_support_policy'] = ['A guarded_visual_identity_candidate with can_support_object_identity=true may be used to select or name an object candidate.', 'Unless relation_verified=true, that identity candidate does not prove the requested left/right/above/below/on/near relation.', 'It never proves state, absence, count, capacity, or an owner-bound attribute.']
            prompt_payload['graph_contract']['forbidden_behaviors'].extend(['treating identity-candidate support as proof of an unverified spatial direction', 'treating an unbound visual identity observation as a canonical graph fact'])
        if guarded_v4d_component_preference and guarded_identity_support:
            prompt_payload['instructions']['open_identity_prefers_complete_object_over_unrequested_component'] = True
            prompt_payload['answering_policy']['visual_identity_support_policy'].append('For an open object-identity question, prefer a candidate marked complete_object_preferred=true over armrests, backrests, cushions, handles, or other unrequested components when it satisfies the same descriptive constraints.')
        if allows_common_world_knowledge:
            prompt_payload['instructions']['do_not_use_answer_keys_or_external_world_knowledge'] = False
            prompt_payload['instructions']['allow_common_world_knowledge_after_grounded_identity'] = True
            prompt_payload['answering_policy']['metadata_policy'][-1] = 'Use no answer keys, hidden labels, or dataset-specific assumptions. For functional_world questions only, common stable world knowledge may be applied after supplied visual/graph evidence grounds the object identity.'
            prompt_payload['graph_contract']['forbidden_behaviors'] = [item for item in prompt_payload['graph_contract']['forbidden_behaviors'] if item != 'using outside knowledge, answer keys, or dataset-specific assumptions']
            prompt_payload['graph_contract']['forbidden_behaviors'].append('using world knowledge without a visually or graph-grounded object identity')
        world_knowledge_instruction = 'Do not use answer keys or dataset-specific assumptions. For a functional_world evidence demand only, you may apply common stable world knowledge after the supplied visual or graph evidence grounds the relevant object identity.' if allows_common_world_knowledge else 'Do not use answer keys, dataset-specific assumptions, or outside world knowledge.'
        if self.config.prompt_style == 'owner_minimal':
            return self._build_owner_minimal_messages(question=question, prompt_payload=prompt_payload, time_adaptation=time_adaptation)
        user_content: str | list[dict[str, Any]] = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        return [{'role': 'system', 'content': f"""You are the final verbalization layer for a canonical temporal memory graph with hybrid graph retrieval over first-person robot navigation video. In this dataset, 'I', 'you', and 'we' in a question refer to the robot/ego camera; 'see' means visually observed by that camera, and 'right now' means the robot state at the question time. This viewpoint background only disambiguates wording and is not observation evidence. The graph retrieval result is the source of truth; you may only rewrite, compress, and explain the supplied graph evidence. Follow this evidence priority: direct state or attribute evidence bound to the queried entity; direct entity labels, canonical labels, semantic aliases, or salient scene labels; ordered scene chains and scene boundaries; relation episodes with queried subject or object endpoints; then generic spatial or co-occurrence relations as weak context only. For temporal questions, first inspect graph_summary.temporal_chain_lines and structured_graph_context.temporal_chains. Treat each temporal chain as the main time axis: ordered evidence from earlier scene to later scene. Use these chains before isolated matches when reasoning about entering, leaving, before/after, between, and duration questions. When a chain shows a state before a boundary, a transition scene, and a state after the boundary, you may infer the interval between the observed boundary frames, but state clearly that the interval is inferred from graph evidence. Do not answer 'insufficient evidence' merely because there is no explicit ENTER or LEAVE event if the ordered temporal chains support a reasonable inference. When a question_ir block is provided, use it as the answer contract: respect its operation, target entities, attribute dimension, temporal relation, hard constraints, aggregation, rendering, evidence requirements, and graph information needs. Use question_ir_execution.semantic_event_candidates only as graph-derived candidates supported by the supplied scene chains or evidence lines; do not treat them as external labels. If aggregation asks for multiple values, synthesize all directly bound matching evidence rather than only the top-ranked item. For binary yes/no questions, answer_text must be exactly 'yes' or exactly 'no' with no explanation, punctuation, or additional words. Use 'yes' when the supplied graph evidence directly or reasonably inferentially supports the question factors; use 'no' when evidence contradicts them or no meaningful support is present. Put uncertainty or supporting evidence only in the JSON metadata fields, not in answer_text. Treat named graph information needs as diagnostic guidance, not as veto requirements: mention gaps when relevant, but still answer from other retrieved graph evidence when it reasonably supports the question. Before choosing a final yes/no, attribute, or event answer, inspect graph_summary.evidence_aggregation_lines and graph_summary.candidate_facts. These fields group the retrieved candidates by entity, scene, time range, and IR question-factor and requirement hits. Use them to compare competing candidates, bind attributes to the right target, and decide whether support for an action/state is direct, inferential, weak, or absent. When graph_summary.answer_candidate_extraction is available, treat it as a generic candidate extraction aid over the same graph evidence, not as a gold label or an extra source of facts. Use answer_candidate_lines to notice concise possible answers across binary, descriptive, temporal, duration, count, attribute, relation, action, and event questions, but verify each candidate against the supplied evidence and question_ir before using it. When graph_summary.atomic_answer_fact_lines is available, treat it as a generic mechanical summary over supplied evidence, not as a gold label or new source of facts. Use it to check subject, predicate, object/attribute, time, support source, and binding hints, especially when raw evidence mixes multiple nearby objects or attributes. Reject atomic facts when their support is weak, missing, contradicted, or bound to the wrong target. When graph_summary visual/frame/merged-frame FAISS evidence lines are available, treat them as supplemental visual similarity evidence from the selected video sequence. They may help with repeated appearances or unstable object labels, but they are not gold labels or external captions, and the final answer must still be supported by the supplied graph context and temporal focus. When graph_summary.visual_faiss_unified_candidate_lines is available, inspect it before raw visual FAISS lines: those candidates unify object FAISS recall, frame-level HSGG object attributes and relations, and nearby graph support. Prefer strong or medium unified candidates whose attribute or relation is bound to the recalled object over weak same-frame context. When graph_summary.visual_faiss_object_local_fact_lines is available, inspect it before broader GER-like context: these short facts keep attributes and relations local to the recalled object and its same-frame direct neighbors. Prefer them for object attributes, colors, relations, and local actions when they cover the question factors. When graph_summary.visual_faiss_role_aware_lines is available, inspect it before broader visual context: these lines promote question_ir target_entities as candidate subjects and keep event, spatial, or information-need visual hits as constraint_context attachments rather than answer subjects. When graph_summary.visual_faiss_role_aware_aux_lines is available, use it only as auxiliary role-check guidance over visual FAISS hits; do not let it override stronger GER-like, object-local, graph, or temporal evidence. When graph_summary.visual_faiss_ger_bound_fact_lines is available, inspect it before raw visual FAISS lines: those lines are generic, evidence-derived binding summaries over the same retrieved visual objects and nearby graph context. Prefer a lower raw visual rank with stronger bound terms and graph support over a higher raw rank whose object does not satisfy the question constraints. For attribute and descriptive questions, if the first aggregation group has target/question-factor/requirement hits and contains a target-bound attribute fact, answer from that group rather than the raw rank-1 evidence line, unless a later group has better overall support for the requested question factors. For attribute questions, bind attributes, states, and descriptions only to the entity or label they are attached to in the graph evidence. Do not transfer attributes from nearby objects, scene neighbors, or generic relation endpoints to the queried target. If the target is found but the requested attribute is missing, say that the target is present but the attribute is not directly supported. Question metadata and answer-format hints are not observation evidence. They may only guide timestamp arithmetic or output wording; never use them as proof of what appeared or happened. {world_knowledge_instruction} The time_adaptation block contains relative-time candidates computed only from retrieved graph timestamps and the question current time. First choose the event from graph evidence; if that event matches a time_anchor, quote the anchor's display_text and include its id in used_time_anchor_ids. Do not manually recalculate minutes in prose when a matching time_anchor is available. When structured_graph_context.entity_aliases is available, prefer its display_text instead of raw entity ids, while preserving the short id only when it helps disambiguate. Preserve frame indices as frame indices. If the user question provides current_time_unix and graph evidence provides Unix timestamps, you may compute minutes-ago or durations from those timestamps; otherwise do not rewrite frame indices as seconds. Prefer citing multiple evidence lines and focus frames when the query spans scenes, state changes, relations, or multiple matches. Return used_evidence_ranks as 1-based indices into graph_summary.evidence_lines. Do not copy full evidence lines into the JSON output unless absolutely necessary. Return strict JSON only with keys {{"answer_text":"...","used_focus_frames":[0],"used_evidence_ranks":[1],"used_time_anchor_ids":["time_anchor:01"],"uncertainty":["..."]}}. If the evidence is insufficient, say so in answer_text and uncertainty."""}, {'role': 'user', 'content': user_content}]

    def _build_owner_minimal_messages(self, *, question: str, prompt_payload: dict[str, Any], time_adaptation: dict[str, Any]) -> list[dict[str, str]]:
        graph_summary = dict(prompt_payload.get('graph_summary', {}))
        minimal_payload = {'task': 'answer_from_supplied_video_graph_evidence', 'question': question, 'question_ir': prompt_payload.get('question_ir', {}), 'question_ir_execution': prompt_payload.get('question_ir_execution', {}), 'time_adaptation': time_adaptation, 'graph_summary': graph_summary, 'structured_graph_context': prompt_payload.get('structured_graph_context', {}), 'graph_context': prompt_payload.get('graph_context', ''), 'output_json_schema': {'answer_text': 'short final answer', 'used_focus_frames': [], 'used_evidence_ranks': [], 'used_time_anchor_ids': [], 'uncertainty': []}}
        return [{'role': 'system', 'content': "Answer the question using only the supplied graph_summary, question_ir, structured_graph_context, and graph_context. The video is first-person robot/ego-camera navigation; 'I', 'you', and 'we' refer to the robot camera. First identify the target object/entity/event required by the question, then choose evidence whose owner, time, relation, and constraints match that target. HSGG attributes are robot-view visual descriptions attached to specific detected objects. direct_object_attribute belongs to the named owner object at that frame/bbox. neighbor_object_attribute is only neighbor or context evidence and must not be transferred to the target object. If the target is present but the requested direct attribute is missing, say the attribute is not directly supported instead of borrowing a neighbor attribute. For binary questions, answer_text must be exactly yes or no. For text questions, answer briefly. For time or duration questions, use graph-backed time anchors or interval evidence; do not invent numbers. Return strict JSON only with keys answer_text, used_focus_frames, used_evidence_ranks, used_time_anchor_ids, and uncertainty."}, {'role': 'user', 'content': json.dumps(minimal_payload, ensure_ascii=False, indent=2)}]

    def _build_payload(self, *, question: str, llm_ready_bundle: dict[str, Any], graph_result: dict[str, Any], force_json_answer: bool=False) -> dict[str, Any]:
        payload: dict[str, Any] = {'model': self.config.model, 'temperature': self.config.temperature, 'max_tokens': max(self.config.max_tokens, 900) if force_json_answer else self.config.max_tokens, 'response_format': {'type': 'json_object'}, 'messages': self._build_messages(question=question, llm_ready_bundle=llm_ready_bundle, graph_result=graph_result)}
        if force_json_answer:
            _append_json_retry_instruction(payload)
            payload['thinking'] = {'type': 'disabled'}
        else:
            apply_reasoning_options(payload, reasoning_effort=self.config.reasoning_effort, thinking_enabled=self.config.thinking_enabled)
        return payload

    def _extract_content(self, raw_response: dict[str, Any]) -> str | None:
        choices = raw_response.get('choices')
        if not isinstance(choices, list) or not choices:
            return None
        message = choices[0].get('message', {})
        content = message.get('content')
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_chunks = [item.get('text', '') for item in content if isinstance(item, dict)]
            merged = ''.join(text_chunks).strip()
            return merged or None
        return None

    def _parse_json_content(self, content: str) -> dict[str, Any] | None:
        stripped = _strip_code_fences(content)
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            candidate = _extract_first_json_object(stripped)
            if candidate is None:
                return None
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _normalize_structured_answer(self, *, parsed: dict[str, Any], graph_result: dict[str, Any], question: str='') -> dict[str, Any] | None:
        llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
        retrieval_bundle = graph_result.get('retrieval_bundle', {})
        ordered_allowed_evidence_lines = _ordered_unique_strings(llm_ready_bundle.get('evidence_lines', []))
        allowed_evidence_lines = set(ordered_allowed_evidence_lines)
        ordered_allowed_focus_frames = _ordered_unique_ints((item.get('frame_index') for item in retrieval_bundle.get('focus_frames', []) if isinstance(item, dict)))
        allowed_focus_frames = set(ordered_allowed_focus_frames)
        structured_graph_context = llm_ready_bundle.get('structured_graph_context', {})
        allowed_time_anchor_ids = _allowed_time_anchor_ids(build_graph_time_adaptation(question=question, graph_result=graph_result))
        citation_requirements = self._build_citation_requirements(graph_result=graph_result, llm_ready_bundle=llm_ready_bundle, focus_frame_lines=llm_ready_bundle.get('focus_frame_lines', []))
        answer_text = _repair_response_text(str(parsed.get('answer_text', '')).strip())
        if not answer_text:
            answer_text = _repair_response_text(str(graph_result.get('answer_brief', '')).strip())
        used_focus_frames = _ordered_unique_ints(parsed.get('used_focus_frames', []))
        used_evidence_lines = _ordered_unique_strings(parsed.get('used_evidence_lines', []))
        used_evidence_ranks = _ordered_unique_ints(parsed.get('used_evidence_ranks', []))
        if not used_evidence_ranks:
            used_evidence_ranks = _ordered_unique_ints(parsed.get('used_evidence_line_ranks', []))
        used_time_anchor_ids = _ordered_unique_strings(parsed.get('used_time_anchor_ids', []))
        uncertainty = [_repair_response_text(value) for value in _ordered_unique_strings(parsed.get('uncertainty', []))]
        ranked_evidence_lines = [ordered_allowed_evidence_lines[rank - 1] for rank in used_evidence_ranks if 1 <= rank <= len(ordered_allowed_evidence_lines)]
        used_evidence_lines = _ordered_unique_strings([*ranked_evidence_lines, *used_evidence_lines])
        if self.config.enforce_supported_evidence_only:
            used_focus_frames = [value for value in used_focus_frames if value in allowed_focus_frames]
            used_evidence_lines = [value for value in used_evidence_lines if value in allowed_evidence_lines]
            used_time_anchor_ids = [value for value in used_time_anchor_ids if value in allowed_time_anchor_ids]
        used_evidence_lines = _normalize_selected_evidence_lines(current=used_evidence_lines, available=ordered_allowed_evidence_lines, target_count=int(citation_requirements['target_evidence_lines']), min_count=int(citation_requirements['min_evidence_lines']), answer_text=answer_text)
        used_focus_frames = _backfill_ordered_values(current=used_focus_frames, available=ordered_allowed_focus_frames, target_count=int(citation_requirements['target_focus_frames']))
        answer_text = _enrich_answer_text_with_entity_aliases(answer_text, structured_graph_context)
        normalized = {'answer_text': answer_text, 'used_focus_frames': used_focus_frames, 'used_evidence_lines': used_evidence_lines, 'used_time_anchor_ids': used_time_anchor_ids, 'uncertainty': uncertainty}
        if self.config.enforce_supported_evidence_only and (not answer_text):
            return None
        if self.config.enforce_supported_evidence_only and len(used_evidence_lines) < int(citation_requirements['min_evidence_lines']):
            return None
        if self.config.enforce_supported_evidence_only and len(used_focus_frames) < int(citation_requirements['min_focus_frames']):
            return None
        return normalized

    def _build_citation_requirements(self, *, graph_result: dict[str, Any], llm_ready_bundle: dict[str, Any], focus_frame_lines: list[Any]) -> dict[str, Any]:
        matches = graph_result.get('matches', [])
        query_type = str(graph_result.get('query_type', '')).strip()
        evidence_count = len([item for item in llm_ready_bundle.get('evidence_lines', []) if str(item).strip()])
        focus_count = len([item for item in focus_frame_lines if str(item).strip()])
        source_kinds = {str(match.get('source_kind') or '').strip() for match in matches if isinstance(match, dict) and str(match.get('source_kind') or '').strip()}
        evidence_bonus = 0
        focus_bonus = 0
        complexity_flags: list[str] = []
        if query_type in {'frame_snapshot', 'scene_summary', 'canonical_temporal'}:
            evidence_bonus += self.config.complex_query_bonus_evidence_lines
            focus_bonus += self.config.complex_query_bonus_focus_frames
            complexity_flags.append('scene_frame_or_temporal_query')
        if len(matches) > 1:
            evidence_bonus += self.config.complex_query_bonus_evidence_lines
            focus_bonus += self.config.complex_query_bonus_focus_frames
            complexity_flags.append('multi_match')
        if query_type in {'entity_timeline', 'canonical_entity'} and any((len(match.get('evidence', {}).get('entity_segments', [])) > 1 for match in matches if isinstance(match, dict))):
            evidence_bonus += self.config.complex_query_bonus_evidence_lines
            complexity_flags.append('multi_segment_entity')
        if query_type == 'canonical_entity' and {'entity', 'episode'}.issubset(source_kinds):
            evidence_bonus += self.config.complex_query_bonus_evidence_lines
            complexity_flags.append('canonical_entity_mixed_lifecycle')
        min_evidence_lines = min(max(self.config.min_evidence_lines, 0), evidence_count)
        min_focus_frames = min(max(self.config.min_focus_frames, 0), focus_count)
        target_evidence_lines = min(evidence_count, max(min_evidence_lines, self.config.target_evidence_lines + evidence_bonus))
        target_focus_frames = min(focus_count, max(min_focus_frames, self.config.target_focus_frames + focus_bonus))
        return {'min_evidence_lines': min_evidence_lines, 'target_evidence_lines': target_evidence_lines, 'min_focus_frames': min_focus_frames, 'target_focus_frames': target_focus_frames, 'complexity_flags': complexity_flags}

    def _response_metadata(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        choice = {}
        choices = raw_response.get('choices')
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                choice = first
        metadata = {'id': raw_response.get('id'), 'model': raw_response.get('model'), 'finish_reason': choice.get('finish_reason'), 'usage': raw_response.get('usage')}
        transport_metadata = raw_response.get(_TRANSPORT_METADATA_KEY)
        if isinstance(transport_metadata, dict):
            metadata['transport'] = dict(transport_metadata)
        return metadata

    def _invalid_response_result(self, *, question: str, graph_result: dict[str, Any], error: str, raw_content: str | None=None, response_metadata: dict[str, Any] | None=None) -> dict[str, Any]:
        if self.config.fallback_to_graph_only_on_invalid_response:
            fallback = graph_only_answer(question=question, graph_result=graph_result)
            fallback.update({'mode': 'live', 'graph_rag_status': 'live_fallback_graph_only', 'provider': self.config.provider, 'model': self.config.model, 'question': question, 'error': error, 'fallback_strategy': 'graph_only'})
            if raw_content is not None:
                fallback['raw_content'] = raw_content
            if response_metadata is not None:
                fallback['response_metadata'] = response_metadata
            return fallback
        result = {'mode': 'live', 'graph_rag_status': 'live_invalid_response', 'provider': self.config.provider, 'model': self.config.model, 'question': question, 'error': error, 'answer_text': graph_result.get('answer_brief', ''), 'fallback_strategy': 'disabled', 'graph_rag_metadata': _build_graph_rag_metadata(graph_result)}
        if raw_content is not None:
            result['raw_content'] = raw_content
        if response_metadata is not None:
            result['response_metadata'] = response_metadata
        return result

def _build_graph_rag_metadata(graph_result: dict[str, Any]) -> dict[str, Any]:
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    llm_ready_bundle = graph_result.get('llm_ready_bundle', {})
    focus_frames = retrieval_bundle.get('focus_frames', [])
    scene_summaries = retrieval_bundle.get('scene_summaries', [])
    return {'contract_version': 'graph_rag_answer_v1', 'query_type': graph_result.get('query_type'), 'match_count': len(graph_result.get('matches', [])), 'focus_frame_count': len(focus_frames), 'scene_count': len(scene_summaries), 'evidence_line_count': len(llm_ready_bundle.get('evidence_lines', [])), 'evidence_selection': llm_ready_bundle.get('llm_evidence_selection', {}), 'answer_candidate_extraction': llm_ready_bundle.get('answer_candidate_extraction', {}), 'atomic_answer_facts': llm_ready_bundle.get('atomic_answer_facts', {})}

def _build_llm_evidence_selection_candidates(llm_ready_bundle: dict[str, Any], *, graph_result: dict[str, Any], max_candidates: int, question: str='', balanced: bool=False, answerable_candidate_packs: bool=False, answerable_candidate_pack_max_items: int=12) -> list[dict[str, Any]]:
    if _score_contract_enabled(llm_ready_bundle):
        return _build_score_contract_evidence_selection_candidates(llm_ready_bundle, max_candidates=max_candidates)
    candidates: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    def add_candidate(source: str, index: int, text: str, extra: dict[str, Any] | None=None) -> None:
        normalized_text = _repair_response_text(str(text or '').strip())
        if not normalized_text or normalized_text in seen_text:
            return
        seen_text.add(normalized_text)
        item = {'id': f'{source}:{index}', 'source': source, 'index': index, 'text': normalized_text}
        if extra:
            item.update(extra)
        candidates.append(item)
    for index, line in enumerate(llm_ready_bundle.get('visual_entity_candidate_pack_lines', []) or [], start=1):
        add_candidate('visual_entity_candidate_pack', index, str(line), {'fact_type': 'visual_entity_candidate_pack', 'score': _rank_score(index, base=112.0, step=2.0) + _similarity_bonus(line), 'contract': 'experimental_visual_faiss_entity_candidate_pack_v1', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    for index, line in enumerate(llm_ready_bundle.get('question_route_count_set_lines', []) or [], start=1):
        add_candidate('question_route_count_set', index, str(line), {'fact_type': 'count_candidate_set', 'score': _rank_score(index, base=108.0, step=2.0) + _similarity_bonus(line), 'contract': 'navqa_question_route_count_candidate_set_v1', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    for index, line in enumerate(llm_ready_bundle.get('evidence_lines', []) or [], start=1):
        add_candidate('evidence_line', index, str(line), {'evidence_rank': index})
    if answerable_candidate_packs:
        for index, item in enumerate(_build_answerable_candidate_packs(llm_ready_bundle, graph_result=graph_result, question=question, max_items=answerable_candidate_pack_max_items), start=1):
            add_candidate('answerable_candidate_pack', index, str(item.get('text') or ''), {'pack_key': item.get('pack_key'), 'support_ids': list(item.get('support_ids', []) or []), 'support_sources': list(item.get('support_sources', []) or []), 'support_ranks': list(item.get('support_ranks', []) or []), 'score': item.get('score'), 'contract': 'experimental_answerable_candidate_pack_v1', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_evidence_lines', []) or [], start=1):
        add_candidate('visual_faiss', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_ger_evidence_lines', []) or [], start=1):
        add_candidate('visual_faiss_ger', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_unified_candidate_lines', []) or [], start=1):
        add_candidate('visual_faiss_unified', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_evidence_layer_lines', []) or [], start=1):
        add_candidate('visual_faiss_evidence_layer', index, str(line), {'fact_type': 'evidence_layer_object_pack', 'score': _rank_score(index, base=106.0, step=2.0) + _similarity_bonus(line)})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_object_local_fact_lines', []) or [], start=1):
        add_candidate('visual_faiss_object_local', index, str(line), {'fact_type': 'object_local_visual_fact', 'score': _rank_score(index, base=99.0, step=2.0) + _similarity_bonus(line)})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_local_subgraph_fact_lines', []) or [], start=1):
        add_candidate('visual_faiss_local_subgraph', index, str(line), {'fact_type': 'entity_local_subgraph_fact', 'score': _rank_score(index, base=104.0, step=2.0) + _similarity_bonus(line)})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_role_aware_lines', []) or [], start=1):
        add_candidate('visual_faiss_role_aware', index, str(line), {'fact_type': 'visual_faiss_role_aware_candidate', 'score': _rank_score(index, base=105.0, step=2.0) + _similarity_bonus(line)})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_role_aware_aux_lines', []) or [], start=1):
        add_candidate('visual_faiss_role_aware_aux', index, str(line), {'fact_type': 'visual_faiss_role_aware_aux_candidate', 'score': _rank_score(index, base=45.0, step=1.5) + _similarity_bonus(line)})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_ger_bound_fact_lines', []) or [], start=1):
        add_candidate('visual_faiss_ger_bound', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('evidence_aggregation_lines', []) or [], start=1):
        add_candidate('aggregation_line', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('atomic_answer_fact_lines', []) or [], start=1):
        add_candidate('atomic_answer_fact', index, str(line))
    aggregation = llm_ready_bundle.get('evidence_aggregation', {})
    if isinstance(aggregation, dict):
        for index, fact in enumerate(aggregation.get('candidate_facts', []) or [], start=1):
            if not isinstance(fact, dict):
                continue
            text_parts = [str(fact.get('fact_text') or ''), ' '.join((str(value) for value in fact.get('support_examples', []) or []))]
            add_candidate('candidate_fact', index, ' :: '.join((part for part in text_parts if part.strip())), {'fact_type': fact.get('fact_type'), 'binding_strength': fact.get('binding_strength'), 'support_ranks': list(fact.get('support_ranks', []) or [])})
    for index, line in enumerate(llm_ready_bundle.get('temporal_chain_lines', []) or [], start=1):
        add_candidate('temporal_chain', index, str(line))
    for index, line in enumerate(llm_ready_bundle.get('temporal_span_candidate_lines', []) or [], start=1):
        add_candidate('temporal_span_candidate', index, str(line), {'fact_type': 'temporal_span_candidate', 'score': _rank_score(index, base=96.0, step=2.0)})
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        for index, item in enumerate(retrieval_bundle.get('evidence_digest', []) or [], start=1):
            if not isinstance(item, dict):
                continue
            add_candidate('evidence_digest', index, str(item.get('summary') or ''), {'source_kind': item.get('source_kind'), 'source_record_id': item.get('source_record_id'), 'retrieval_rank': item.get('rank')})
    limit = max(1, max_candidates)
    if balanced:
        return _balanced_evidence_candidates(candidates, limit=limit)
    return candidates[:limit]

def _score_contract_enabled(llm_ready_bundle: dict[str, Any]) -> bool:
    score_contract = llm_ready_bundle.get('score_contract', {})
    return isinstance(score_contract, dict) and bool(score_contract.get('enabled'))

def _build_score_contract_evidence_selection_candidates(llm_ready_bundle: dict[str, Any], *, max_candidates: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    def add_candidate(source: str, index: int, text: str, extra: dict[str, Any] | None=None) -> None:
        normalized_text = _repair_response_text(str(text or '').strip())
        if not normalized_text or normalized_text in seen_text:
            return
        seen_text.add(normalized_text)
        item = {'id': f'{source}:{index}', 'source': source, 'index': index, 'text': normalized_text, 'score_contract_candidate': True}
        if extra:
            item.update(extra)
        candidates.append(item)
    packs = []
    metadata = llm_ready_bundle.get('visual_entity_candidate_packs', {})
    if isinstance(metadata, dict):
        packs = [item for item in metadata.get('packs', []) or [] if isinstance(item, dict)]
    pack_by_rank = {index: pack for index, pack in enumerate(packs, start=1)}
    for index, line in enumerate(llm_ready_bundle.get('visual_entity_candidate_pack_lines', []) or [], start=1):
        pack = pack_by_rank.get(index, {})
        add_candidate('candidate_pack', index, str(line), {'fact_type': 'candidate_pack', 'contract': 'visual_candidate_pack_score_contract_v1', 'candidate_unit': 'candidate_pack', 'final_rank_key': pack.get('final_rank_key'), 'retrieval_score': pack.get('retrieval_score'), 'binding_score': pack.get('binding_score'), 'semantic_score': pack.get('semantic_score'), 'answerability_score': pack.get('answerability_score'), 'answerability_tier': pack.get('answerability_tier'), 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    fallback_lines = list(llm_ready_bundle.get('fallback_evidence_lines', []) or [])
    if not fallback_lines:
        pack_lines = {str(line).strip() for line in llm_ready_bundle.get('visual_entity_candidate_pack_lines', []) or []}
        fallback_lines = [str(line).strip() for line in llm_ready_bundle.get('evidence_lines', []) or [] if str(line).strip() and str(line).strip() not in pack_lines]
    for index, line in enumerate(fallback_lines, start=1):
        add_candidate('fallback_evidence', index, str(line), {'fact_type': 'fallback_evidence', 'contract': 'visual_candidate_pack_score_contract_v1', 'candidate_unit': 'fallback_evidence', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False})
    return candidates[:max(1, max_candidates)]
_ANSWERABLE_PACK_SECTION_KEYS = {'object', 'target', 'time', 'span', 'direct_attributes', 'attributes', 'direct_relations', 'relations', 'direct_neighbors', 'frame_context', 'nearby_graph', 'kg_context', 'anchor', 'matched_question_terms', 'matched_constraint_terms', 'q_terms', 'constraint_terms'}

def _build_answerable_candidate_packs(llm_ready_bundle: dict[str, Any], *, graph_result: dict[str, Any], question: str='', max_items: int) -> list[dict[str, Any]]:
    del question
    records = _answerable_pack_source_records(llm_ready_bundle, graph_result=graph_result)
    records = _answerable_pack_backfill_frame_label_records(records)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get('pack_key') or '').strip()
        if not key:
            continue
        grouped.setdefault(key, []).append(record)
    packs: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        pack = _answerable_pack_from_records(key, rows)
        if pack is not None:
            packs.append(pack)
    packs.sort(key=lambda item: (-float(item.get('score') or 0.0), str(item.get('pack_key') or '')))
    return packs[:max(1, int(max_items))]

def _answerable_pack_backfill_frame_label_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_keys: dict[tuple[str, str], list[str]] = {}
    for record in records:
        frame = str(record.get('frame') or '').strip()
        label = str(record.get('label') or '').strip()
        obj = str(record.get('obj') or '').strip()
        key = str(record.get('pack_key') or '').strip()
        if frame and label and obj and key:
            object_keys.setdefault((frame, label), []).append(key)
    expanded: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for record in records:
        record_id = str(record.get('id') or '')
        pack_key = str(record.get('pack_key') or '')
        marker = (record_id, pack_key)
        if marker not in seen:
            expanded.append(record)
            seen.add(marker)
        frame = str(record.get('frame') or '').strip()
        label = str(record.get('label') or '').strip()
        obj = str(record.get('obj') or '').strip()
        if obj or not frame or (not label):
            continue
        target_keys = _ordered_unique_strings(object_keys.get((frame, label), []))
        if len(target_keys) != 1:
            continue
        for target_key in target_keys:
            if target_key == pack_key:
                continue
            cloned = dict(record)
            cloned['pack_key'] = target_key
            marker = (record_id, target_key)
            if marker in seen:
                continue
            expanded.append(cloned)
            seen.add(marker)
    return expanded

def _answerable_pack_source_records(llm_ready_bundle: dict[str, Any], *, graph_result: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add_record(source: str, index: int, text: Any) -> None:
        normalized = _repair_response_text(str(text or '').strip())
        if not normalized:
            return
        values = _parse_atomic_key_values(normalized)
        key_parts = _answerable_pack_key_parts(normalized, values)
        if not key_parts:
            return
        sections = _answerable_pack_sections(normalized, values)
        records.append({'id': f'{source}:{index}', 'source': source, 'index': index, 'pack_key': '|'.join(key_parts), 'frame': _answerable_value(values, 'frame', 'frames'), 'obj': _answerable_value(values, 'obj', 'object_index'), 'label': _answerable_value(values, 'label'), 'text': normalized, 'sections': sections, 'score': _answerable_source_score(source, index, normalized, sections), 'support_rank': index if source in {'evidence_line', 'visual_faiss_evidence_layer', 'visual_faiss_local_subgraph', 'visual_faiss_role_aware', 'visual_faiss_role_aware_aux', 'visual_faiss_object_local', 'visual_faiss_ger_bound', 'visual_faiss_ger'} else None})
    for source, key in (('evidence_line', 'evidence_lines'), ('visual_faiss_evidence_layer', 'visual_faiss_evidence_layer_lines'), ('visual_faiss_local_subgraph', 'visual_faiss_local_subgraph_fact_lines'), ('visual_faiss_role_aware', 'visual_faiss_role_aware_lines'), ('visual_faiss_role_aware_aux', 'visual_faiss_role_aware_aux_lines'), ('visual_faiss_object_local', 'visual_faiss_object_local_fact_lines'), ('visual_faiss_ger_bound', 'visual_faiss_ger_bound_fact_lines'), ('visual_faiss_ger', 'visual_faiss_ger_evidence_lines'), ('visual_faiss_unified', 'visual_faiss_unified_candidate_lines'), ('visual_faiss', 'visual_faiss_evidence_lines'), ('aggregation_line', 'evidence_aggregation_lines'), ('atomic_answer_fact', 'atomic_answer_fact_lines'), ('temporal_chain', 'temporal_chain_lines'), ('temporal_span_candidate', 'temporal_span_candidate_lines')):
        for index, line in enumerate(llm_ready_bundle.get(key, []) or [], start=1):
            add_record(source, index, line)
    aggregation = llm_ready_bundle.get('evidence_aggregation', {})
    if isinstance(aggregation, dict):
        for index, fact in enumerate(aggregation.get('candidate_facts', []) or [], start=1):
            if not isinstance(fact, dict):
                continue
            text_parts = [str(fact.get('fact_text') or ''), ' '.join((str(value) for value in fact.get('support_examples', []) or []))]
            add_record('candidate_fact', index, ' :: '.join((part for part in text_parts if part.strip())))
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        for index, item in enumerate(retrieval_bundle.get('evidence_digest', []) or [], start=1):
            if isinstance(item, dict):
                add_record('evidence_digest', index, item.get('summary'))
    return records

def _answerable_pack_key_parts(text: str, values: dict[str, str]) -> list[str]:
    frame = _answerable_value(values, 'frame', 'frames')
    obj = _answerable_value(values, 'obj', 'object_index')
    label = _answerable_value(values, 'label')
    source_record = _answerable_value(values, 'source_record_id', 'id')
    if frame and obj:
        parts = [f'frame={frame}', f'obj={obj}']
        if label:
            parts.append(f'label={label}')
        return parts
    if frame and label:
        return [f'frame={frame}', f'label={label}']
    if source_record:
        return [f'record={source_record}']
    return []

def _answerable_value(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = values.get(key.strip().lower().replace('-', '_'))
        if value:
            return value
    return ''

def _answerable_pack_sections(text: str, values: dict[str, str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    if '::' in text:
        body = text.split('::', 1)[1]
    else:
        body = text
    for chunk in re.split('\\s*;\\s*', body):
        if '=' not in chunk:
            continue
        key, raw_value = chunk.split('=', 1)
        normalized_key = key.strip().lower().replace('-', '_')
        if normalized_key not in _ANSWERABLE_PACK_SECTION_KEYS:
            continue
        value = _answerable_pack_value(raw_value)
        if value:
            sections.setdefault(normalized_key, []).append(value)
    for key in _ANSWERABLE_PACK_SECTION_KEYS:
        value = values.get(key)
        if value:
            sections.setdefault(key, []).append(_answerable_pack_value(value))
    return {key: _ordered_unique_strings(values) for key, values in sections.items() if values}

def _answerable_pack_value(value: Any, *, limit: int=220) -> str:
    text = _repair_response_text(str(value or '').strip(' \'"'))
    text = re.sub('\\s+', ' ', text).strip(' |;')
    if len(text) <= limit:
        return text
    shortened = text[:limit].rsplit(' ', 1)[0].rstrip(' ,;:.')
    return shortened + '...'

def _answerable_source_score(source: str, index: int, text: str, sections: dict[str, list[str]]) -> float:
    source_weights = {'visual_faiss_evidence_layer': 54.0, 'visual_faiss_role_aware': 53.0, 'visual_faiss_local_subgraph': 52.0, 'visual_faiss_object_local': 48.0, 'visual_faiss_role_aware_aux': 46.0, 'visual_faiss_ger_bound': 44.0, 'visual_faiss_unified': 42.0, 'candidate_fact': 38.0, 'aggregation_line': 34.0, 'evidence_line': 32.0, 'visual_faiss_ger': 30.0, 'atomic_answer_fact': 28.0, 'visual_faiss': 24.0, 'temporal_span_candidate': 23.0, 'temporal_chain': 22.0, 'evidence_digest': 18.0}
    answerable_keys = {'direct_attributes', 'attributes', 'direct_relations', 'relations', 'direct_neighbors', 'frame_context', 'nearby_graph', 'kg_context', 'anchor', 'time', 'span'}
    slot_count = sum((1 for key in answerable_keys if sections.get(key)))
    matched_terms = sum((len(sections.get(key, [])) for key in ('matched_question_terms', 'matched_constraint_terms', 'q_terms', 'constraint_terms')))
    return source_weights.get(source, 10.0) + min(18.0, slot_count * 3.0) + min(8.0, matched_terms * 1.5) + _similarity_bonus(text) + _rank_score(index, base=10.0, step=0.8)

def _answerable_pack_from_records(key: str, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    ordered_rows = sorted(rows, key=lambda item: (-float(item.get('score') or 0.0), int(item.get('index') or 0), str(item.get('source') or '')))
    merged_sections: dict[str, list[str]] = {}
    for row in ordered_rows:
        sections = row.get('sections', {})
        if not isinstance(sections, dict):
            continue
        for section_key, values in sections.items():
            merged_sections.setdefault(str(section_key), []).extend((str(value) for value in values or [] if str(value).strip()))
    merged_sections = {section_key: _ordered_unique_strings(values)[:4] for section_key, values in merged_sections.items() if values}
    if not merged_sections:
        return None
    best = ordered_rows[0]
    support_sources = _ordered_unique_strings((f"{row.get('source')}:{row.get('index')}" for row in ordered_rows))
    support_ids = _ordered_unique_strings((row.get('id') for row in ordered_rows))
    support_ranks = [int(row['support_rank']) for row in ordered_rows if isinstance(row.get('support_rank'), int)][:8]
    score = sum((float(row.get('score') or 0.0) for row in ordered_rows[:4])) / min(len(ordered_rows), 4)
    text = _answerable_candidate_pack_line(pack_key=key, best=best, sections=merged_sections, support_sources=support_sources, support_texts=[str(row.get('text') or '') for row in ordered_rows[:3]])
    return {'pack_key': key, 'text': text, 'score': round(score, 6), 'support_sources': support_sources, 'support_ids': support_ids, 'support_ranks': support_ranks}

def _answerable_candidate_pack_line(*, pack_key: str, best: dict[str, Any], sections: dict[str, list[str]], support_sources: list[str], support_texts: list[str]) -> str:
    header = ['answerable_candidate_pack', f'key={pack_key}', 'sources=' + ','.join(support_sources[:8])]
    if best.get('frame'):
        header.append(f"frame={best.get('frame')}")
    if best.get('obj'):
        header.append(f"obj={best.get('obj')}")
    if best.get('label'):
        header.append(f"label={best.get('label')!r}")
    body_parts: list[str] = []
    for key in ('object', 'target', 'time', 'span', 'direct_attributes', 'attributes', 'direct_relations', 'relations', 'direct_neighbors', 'frame_context', 'nearby_graph', 'kg_context', 'anchor', 'matched_question_terms', 'matched_constraint_terms', 'q_terms', 'constraint_terms'):
        values = sections.get(key, [])
        if values:
            body_parts.append(f'{key}=' + ' | '.join(values[:3]))
    concise_support = [_answerable_pack_value(text.split('::', 1)[1] if '::' in text else text, limit=180) for text in support_texts if str(text).strip()]
    if concise_support:
        body_parts.append('support=' + ' | '.join(concise_support[:2]))
    return ' '.join(header) + ' :: ' + '; '.join(body_parts)

def _balanced_evidence_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        return candidates
    sources = ['evidence_line', 'candidate_fact', 'answerable_candidate_pack', 'atomic_answer_fact', 'aggregation_line', 'temporal_chain', 'temporal_span_candidate', 'visual_faiss_evidence_layer', 'visual_faiss_object_local', 'visual_faiss_local_subgraph', 'visual_faiss_role_aware', 'visual_faiss_role_aware_aux', 'visual_faiss_ger_bound', 'visual_faiss_ger', 'visual_faiss', 'semantic_event_candidate', 'evidence_digest']
    by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
    extras: list[dict[str, Any]] = []
    for item in candidates:
        source = str(item.get('source') or '')
        if source in by_source:
            by_source[source].append(item)
        else:
            extras.append(item)
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        item_id = str(item.get('id') or '')
        if not item_id or item_id in seen_ids or len(selected) >= limit:
            return
        selected.append(item)
        seen_ids.add(item_id)
    for source in sources:
        if by_source[source]:
            add(by_source[source][0])
    depth = 1
    while len(selected) < limit:
        added = False
        for source in sources:
            rows = by_source[source]
            if depth < len(rows):
                add(rows[depth])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        depth += 1
    for item in extras:
        add(item)
        if len(selected) >= limit:
            break
    for item in candidates:
        add(item)
        if len(selected) >= limit:
            break
    return selected

def _normalize_llm_evidence_selection(parsed: dict[str, Any], *, allowed_ids: set[str]) -> dict[str, Any] | None:
    selected_ids = [value for value in _ordered_unique_strings(parsed.get('selected_ids', [])) if value in allowed_ids]
    insufficient = bool(parsed.get('insufficient_evidence', False))
    answer_cue = _repair_response_text(str(parsed.get('answer_cue') or '').strip())
    insufficiency_reason = _repair_response_text(str(parsed.get('insufficiency_reason') or '').strip())
    rationale = _repair_response_text(str(parsed.get('rationale') or '').strip())
    if not selected_ids and (not insufficient):
        return None
    return {'selected_ids': selected_ids, 'answer_cue': answer_cue, 'insufficient_evidence': insufficient, 'insufficiency_reason': insufficiency_reason, 'rationale': rationale}

def _normalize_answer_candidate_extraction(parsed: dict[str, Any], *, allowed_ids: set[str], max_candidates: int, model: str) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    raw_candidates = parsed.get('candidates', [])
    if isinstance(raw_candidates, list):
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            answer = _repair_response_text(str(item.get('answer') or '').strip())
            if not answer:
                continue
            support_ids = [value for value in _ordered_unique_strings(item.get('support_ids', [])) if value in allowed_ids]
            if not support_ids:
                continue
            answer_type = str(item.get('answer_type') or 'unknown').strip().lower()
            if answer_type not in {'binary', 'text', 'time', 'duration', 'count', 'attribute', 'object', 'relation', 'event', 'unknown'}:
                answer_type = 'unknown'
            confidence = str(item.get('confidence') or 'low').strip().lower()
            if confidence not in {'high', 'medium', 'low'}:
                confidence = 'low'
            candidates.append({'answer': answer, 'answer_type': answer_type, 'support_ids': support_ids, 'support_text': _repair_response_text(str(item.get('support_text') or '').strip()), 'confidence': confidence, 'reasoning': _repair_response_text(str(item.get('reasoning') or '').strip())})
            if len(candidates) >= max(1, max_candidates):
                break
    insufficient = bool(parsed.get('insufficient_evidence', False))
    insufficiency_reason = _repair_response_text(str(parsed.get('insufficiency_reason') or '').strip())
    if not candidates and (not insufficient):
        return None
    return {'contract': 'generic_answer_candidate_extraction_v1', 'available': bool(candidates) or bool(parsed.get('available', False)), 'candidates': candidates, 'insufficient_evidence': insufficient, 'insufficiency_reason': insufficiency_reason, 'model': model}

def _bundle_with_answer_candidate_extraction(llm_ready_bundle: dict[str, Any], *, extraction: dict[str, Any], max_candidates: int) -> dict[str, Any]:
    candidates = [item for item in extraction.get('candidates', []) if isinstance(item, dict)][:max(1, max_candidates)]
    candidate_lines = [_answer_candidate_line(index, item) for index, item in enumerate(candidates, start=1)]
    updated = dict(llm_ready_bundle)
    updated['answer_candidate_extraction'] = dict(extraction)
    updated['answer_candidate_lines'] = candidate_lines
    structured = dict(updated.get('structured_graph_context', {}))
    structured['answer_candidate_extraction'] = dict(extraction)
    updated['structured_graph_context'] = structured
    context_lines = [str(updated.get('llm_context_text', '')).strip()]
    if candidate_lines:
        context_lines.append('Generic answer candidate extraction:')
        context_lines.extend((f'- {line}' for line in candidate_lines))
    if extraction.get('insufficient_evidence'):
        reason = str(extraction.get('insufficiency_reason') or '').strip()
        context_lines.append('Answer candidate extraction marked evidence insufficient' + (f': {reason}' if reason else '.'))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    return updated

def _bundle_with_atomic_answer_facts(llm_ready_bundle: dict[str, Any], *, max_items: int) -> dict[str, Any]:
    facts = _build_atomic_answer_facts(llm_ready_bundle, max_items=max_items)
    if not facts:
        return llm_ready_bundle
    fact_lines = [_atomic_answer_fact_line(index, item) for index, item in enumerate(facts, start=1)]
    updated = dict(llm_ready_bundle)
    updated['atomic_answer_facts'] = {'contract': 'experimental_atomic_answer_facts_v1', 'available': True, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'generation': 'mechanical_from_supplied_evidence', 'fact_count': len(facts), 'facts': facts}
    updated['atomic_answer_fact_lines'] = fact_lines
    structured = dict(updated.get('structured_graph_context', {}))
    structured['atomic_answer_facts'] = updated['atomic_answer_facts']
    structured['atomic_answer_fact_lines'] = fact_lines
    updated['structured_graph_context'] = structured
    context_lines = [str(updated.get('llm_context_text', '')).strip()]
    context_lines.append('Atomic answer facts:')
    context_lines.extend((f'- {line}' for line in fact_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    return updated

def _build_atomic_answer_facts(llm_ready_bundle: dict[str, Any], *, max_items: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    limit = max(1, int(max_items))

    def add_fact(*, source: str, source_index: int, text: Any, score: float, fact_type: str, metadata: dict[str, Any] | None=None) -> None:
        normalized = _repair_response_text(str(text or '').strip())
        if not normalized:
            return
        key_values = _parse_atomic_key_values(normalized)
        summary = _atomic_summary_from_text(normalized)
        subject = _atomic_value_from_keys(key_values, 'subject', 'entity', 'object', 'label', 'target', 'scene', 'source')
        predicate = _atomic_value_from_keys(key_values, 'predicate', 'action', 'relation', 'kind', 'support', 'type')
        obj = _atomic_value_from_keys(key_values, 'object', 'object_terms', 'q_terms', 'hard_constraint_hits', 'requirement_hits', 'target_terms')
        attribute = _atomic_value_from_keys(key_values, 'attribute', 'attributes', 'state', 'fact', 'answer', 'candidate')
        time_anchor = _atomic_value_from_keys(key_values, 'time', 'timestamp', 'ts', 'frame', 'frames', 'span', 'time_range')
        binding = _atomic_value_from_keys(key_values, 'binding', 'binding_strength', 'support', 'source', 'support_kind', 'confidence')
        support_ids = _atomic_support_ids_from_text(normalized, key_values)
        signature = '|'.join([source, subject, predicate, obj, attribute, time_anchor, summary])
        if signature in seen:
            return
        seen.add(signature)
        candidates.append({'id': f'atomic:{len(candidates) + 1}', 'source': source, 'source_index': source_index, 'fact_type': fact_type, 'score': round(float(score), 6), 'subject': subject, 'predicate': predicate, 'object': obj, 'attribute': attribute, 'time': time_anchor, 'binding': binding, 'support_ids': support_ids, 'summary': summary, 'source_text': normalized, 'metadata': metadata or {}})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_ger_bound_fact_lines', []) or [], start=1):
        add_fact(source='visual_faiss_ger_bound', source_index=index, text=line, score=_rank_score(index, base=98.0, step=2.0) + _similarity_bonus(line), fact_type='bound_visual_object_fact')
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_role_aware_lines', []) or [], start=1):
        add_fact(source='visual_faiss_role_aware', source_index=index, text=line, score=_rank_score(index, base=99.0, step=2.0) + _similarity_bonus(line), fact_type='visual_faiss_role_aware_candidate')
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_role_aware_aux_lines', []) or [], start=1):
        add_fact(source='visual_faiss_role_aware_aux', source_index=index, text=line, score=_rank_score(index, base=84.0, step=2.0) + _similarity_bonus(line), fact_type='visual_faiss_role_aware_aux_candidate')
    for index, line in enumerate(llm_ready_bundle.get('llm_selected_evidence_lines', []) or [], start=1):
        add_fact(source='llm_selected_evidence', source_index=index, text=line, score=_rank_score(index, base=94.0, step=2.0), fact_type='selected_evidence_fact')
    aggregation = llm_ready_bundle.get('evidence_aggregation', {})
    if isinstance(aggregation, dict):
        for index, item in enumerate(aggregation.get('candidate_facts', []) or [], start=1):
            if not isinstance(item, dict):
                continue
            text = str(item.get('fact_text') or '').strip()
            if not text:
                text = ' '.join((str(value).strip() for value in item.get('support_examples', []) or [] if str(value).strip()))
            add_fact(source='candidate_fact', source_index=index, text=text, score=_rank_score(index, base=92.0, step=2.0) + _binding_strength_bonus(item.get('binding_strength')), fact_type=str(item.get('fact_type') or 'candidate_fact'), metadata={'binding_strength': item.get('binding_strength'), 'support_ranks': list(item.get('support_ranks', []) or [])})
    for index, line in enumerate(llm_ready_bundle.get('evidence_aggregation_lines', []) or [], start=1):
        add_fact(source='evidence_aggregation', source_index=index, text=line, score=_rank_score(index, base=88.0, step=2.0), fact_type='aggregation_fact')
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_unified_candidate_lines', []) or [], start=1):
        add_fact(source='visual_faiss_unified', source_index=index, text=line, score=_rank_score(index, base=86.0, step=2.0) + _similarity_bonus(line), fact_type='unified_visual_candidate_fact')
    for index, line in enumerate(llm_ready_bundle.get('evidence_lines', []) or [], start=1):
        add_fact(source='evidence_line', source_index=index, text=line, score=_rank_score(index, base=82.0, step=2.0), fact_type=_support_kind_from_text(line, default='graph_evidence_fact'))
    for index, line in enumerate(llm_ready_bundle.get('temporal_chain_lines', []) or [], start=1):
        add_fact(source='temporal_chain', source_index=index, text=line, score=_rank_score(index, base=78.0, step=2.0), fact_type='temporal_chain_fact')
    for index, line in enumerate(llm_ready_bundle.get('temporal_span_candidate_lines', []) or [], start=1):
        add_fact(source='temporal_span_candidate', source_index=index, text=line, score=_rank_score(index, base=90.0, step=2.0), fact_type='temporal_span_candidate_fact')
    candidates.sort(key=lambda item: (-float(item.get('score') or 0.0), int(item.get('source_index') or 0), str(item.get('source') or '')))
    return candidates[:limit]

def _atomic_answer_fact_line(index: int, item: dict[str, Any]) -> str:
    parts = [f'atomic={index}', f"source={item.get('source')}", f"idx={item.get('source_index')}", f"type={item.get('fact_type')}", f"score={item.get('score')}"]
    if item.get('subject'):
        parts.append(f"subject={_quote_atomic_value(item.get('subject'))}")
    if item.get('predicate'):
        parts.append(f"predicate={_quote_atomic_value(item.get('predicate'))}")
    if item.get('object'):
        parts.append(f"object={_quote_atomic_value(item.get('object'))}")
    if item.get('attribute'):
        parts.append(f"attribute={_quote_atomic_value(item.get('attribute'))}")
    if item.get('time'):
        parts.append(f"time={_quote_atomic_value(item.get('time'))}")
    if item.get('binding'):
        parts.append(f"binding={_quote_atomic_value(item.get('binding'))}")
    support_ids = _ordered_unique_strings(item.get('support_ids', []))
    if support_ids:
        parts.append('support_ids=' + ','.join(support_ids[:6]))
    return ' '.join(parts) + f" :: {item.get('summary')}"

def _parse_atomic_key_values(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    pattern = re.compile('(?P<key>[A-Za-z][A-Za-z0-9_\\-]*)=(?P<value>\'[^\']*\'|\\"[^\\"]*\\"|[^|;:]+?)(?=\\s+[A-Za-z][A-Za-z0-9_\\-]*=|\\s*::|\\s*[|;:]|$)')
    for match in pattern.finditer(text):
        key = match.group('key').strip().lower().replace('-', '_')
        raw_value = match.group('value').strip().strip('\'"')
        value = _repair_response_text(raw_value)
        if key and value and (key not in values):
            values[key] = value[:240]
    return values

def _atomic_value_from_keys(values: dict[str, str], *keys: str) -> str:
    for key in keys:
        normalized_key = key.strip().lower().replace('-', '_')
        value = values.get(normalized_key)
        if value:
            return value
    return ''

def _atomic_support_ids_from_text(text: str, values: dict[str, str]) -> list[str]:
    candidates: list[str] = []
    for key in ('id', 'support_ids', 'support_ranks', 'evidence_rank', 'rank'):
        value = values.get(key)
        if value:
            candidates.extend(re.findall('[A-Za-z_]+:[A-Za-z0-9_\\-]+|\\b\\d+\\b', value))
    candidates.extend(re.findall('\\b(?:rank|selected|unified|candidate|group)=(\\d+)', text))
    return _ordered_unique_strings(candidates)[:8]

def _atomic_summary_from_text(text: str) -> str:
    body = str(text or '').split('::', 1)[1] if '::' in str(text or '') else str(text or '')
    body = re.sub('\\s+', ' ', body).strip(' -|;')
    if not body:
        body = str(text or '').strip()
    if len(body) <= 260:
        return body
    shortened = body[:260].rsplit(' ', 1)[0].rstrip(' ,;:.')
    return shortened + '...'

def _quote_atomic_value(value: Any) -> str:
    text = _repair_response_text(str(value or '').strip())
    if not text:
        return "''"
    escaped = text.replace("'", "\\'")
    return f"'{escaped}'"

def _bundle_with_unified_evidence_layer(llm_ready_bundle: dict[str, Any], *, graph_result: dict[str, Any], max_items: int, graph_quota: int, visual_quota: int, frame_quota: int) -> dict[str, Any]:
    original_evidence = _ordered_unique_strings(llm_ready_bundle.get('evidence_lines', []))
    items = _build_unified_evidence_items(llm_ready_bundle, graph_result=graph_result)
    selected_items = _select_unified_evidence_items(items, max_items=max_items, graph_quota=graph_quota, visual_quota=visual_quota, frame_quota=frame_quota)
    if not selected_items:
        return llm_ready_bundle
    unified_lines = [_unified_evidence_line(index, item) for index, item in enumerate(selected_items, start=1)]
    updated = dict(llm_ready_bundle)
    updated['unified_evidence_layer'] = {'contract': 'experimental_unified_evidence_layer_v1', 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False, 'candidate_count': len(items), 'selected_count': len(selected_items), 'source_counts': _unified_source_counts(items), 'selected_source_counts': _unified_source_counts(selected_items), 'items': selected_items}
    updated['unified_evidence_lines'] = unified_lines
    updated['evidence_lines'] = _ordered_unique_strings([*unified_lines, *original_evidence])[:max(1, max_items)]
    existing_aggregation_lines = _ordered_unique_strings(updated.get('evidence_aggregation_lines', []))
    unified_aggregation_lines = [_unified_evidence_aggregation_line(index, item) for index, item in enumerate(selected_items, start=1)]
    updated['evidence_aggregation_lines'] = _ordered_unique_strings([*unified_aggregation_lines, *existing_aggregation_lines])[:max(1, max_items)]
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}))
    existing_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = [*_unified_candidate_facts(selected_items), *existing_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    structured = dict(updated.get('structured_graph_context', {}))
    structured['unified_evidence_layer'] = updated['unified_evidence_layer']
    structured['unified_evidence_lines'] = unified_lines
    structured['evidence_lines'] = updated['evidence_lines']
    structured['evidence_aggregation'] = evidence_aggregation
    updated['structured_graph_context'] = structured
    context_lines = [str(updated.get('llm_context_text', '')).strip()]
    context_lines.append('Unified evidence layer:')
    context_lines.extend((f'- {line}' for line in unified_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        digest = list(retrieval_bundle.get('evidence_digest', []) or [])
        next_rank = len(digest) + 1
        for item in selected_items:
            digest.append({'rank': next_rank, 'source_record_id': item.get('id'), 'source_kind': 'unified_evidence', 'summary': item.get('text'), 'score': item.get('score'), 'original_source': item.get('source'), 'support_kind': item.get('support_kind')})
            next_rank += 1
        retrieval_bundle['evidence_digest'] = digest
        graph_result['retrieval_bundle'] = retrieval_bundle
    return updated

def _build_unified_evidence_items(llm_ready_bundle: dict[str, Any], *, graph_result: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_item(*, source: str, index: int, text: Any, score: float, support_kind: str, rank: int | None=None, metadata: dict[str, Any] | None=None) -> None:
        normalized = _repair_response_text(str(text or '').strip())
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        items.append({'id': f'{source}:{index}', 'source': source, 'source_index': index, 'rank': rank if rank is not None else index, 'score': round(float(score), 6), 'support_kind': support_kind, 'text': normalized, 'metadata': metadata or {}})
    for index, line in enumerate(llm_ready_bundle.get('evidence_lines', []) or [], start=1):
        add_item(source='graph', index=index, rank=index, text=line, score=_rank_score(index, base=100.0, step=4.0), support_kind=_support_kind_from_text(line, default='graph_evidence'))
    for index, line in enumerate(llm_ready_bundle.get('evidence_aggregation_lines', []) or [], start=1):
        add_item(source='graph_aggregation', index=index, text=line, score=_rank_score(index, base=88.0, step=3.0), support_kind='graph_aggregation')
    aggregation = llm_ready_bundle.get('evidence_aggregation', {})
    if isinstance(aggregation, dict):
        for index, fact in enumerate(aggregation.get('candidate_facts', []) or [], start=1):
            if not isinstance(fact, dict):
                continue
            fact_text = str(fact.get('fact_text') or '').strip()
            if not fact_text:
                fact_text = ' '.join((str(value).strip() for value in fact.get('support_examples', []) or [] if str(value).strip()))
            add_item(source='candidate_fact', index=index, text=fact_text, score=_rank_score(index, base=92.0, step=2.0) + _binding_strength_bonus(fact.get('binding_strength')), support_kind=str(fact.get('fact_type') or 'candidate_fact'), metadata={'binding_strength': fact.get('binding_strength'), 'support_ranks': list(fact.get('support_ranks', []) or [])})
    for index, line in enumerate(llm_ready_bundle.get('visual_faiss_evidence_lines', []) or [], start=1):
        add_item(source='visual_faiss', index=index, text=line, score=_rank_score(index, base=82.0, step=3.0) + _similarity_bonus(line), support_kind='visual_object_similarity')
    for field_name, source_name in [('frame_faiss_evidence_lines', 'frame_faiss'), ('merged_frame_faiss_evidence_lines', 'merged_frame_faiss')]:
        for index, line in enumerate(llm_ready_bundle.get(field_name, []) or [], start=1):
            add_item(source=source_name, index=index, text=line, score=_rank_score(index, base=72.0, step=3.0) + _similarity_bonus(line), support_kind='visual_frame_similarity')
    for index, line in enumerate(llm_ready_bundle.get('temporal_chain_lines', []) or [], start=1):
        add_item(source='temporal_chain', index=index, text=line, score=_rank_score(index, base=95.0, step=3.0), support_kind='temporal_chain')
    for index, line in enumerate(llm_ready_bundle.get('temporal_span_candidate_lines', []) or [], start=1):
        add_item(source='temporal_span_candidate', index=index, text=line, score=_rank_score(index, base=96.0, step=3.0), support_kind='temporal_span_candidate')
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    if isinstance(retrieval_bundle, dict):
        for index, item in enumerate(retrieval_bundle.get('evidence_digest', []) or [], start=1):
            if not isinstance(item, dict):
                continue
            add_item(source='evidence_digest', index=index, text=item.get('summary'), score=_rank_score(index, base=70.0, step=2.0), support_kind=str(item.get('source_kind') or 'evidence_digest'), metadata={'source_kind': item.get('source_kind'), 'source_record_id': item.get('source_record_id'), 'retrieval_rank': item.get('rank')})
    return sorted(items, key=lambda item: (-float(item.get('score') or 0.0), int(item.get('rank') or 0), str(item.get('id') or '')))

def _select_unified_evidence_items(items: list[dict[str, Any]], *, max_items: int, graph_quota: int, visual_quota: int, frame_quota: int) -> list[dict[str, Any]]:
    limit = max(1, int(max_items))
    graph_sources = {'graph', 'graph_aggregation', 'candidate_fact', 'temporal_chain', 'temporal_span_candidate'}
    visual_sources = {'visual_faiss'}
    frame_sources = {'frame_faiss', 'merged_frame_faiss'}
    family_quotas = {'graph': max(0, int(graph_quota)), 'visual': max(0, int(visual_quota)), 'frame': max(0, int(frame_quota))}
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    family_counts: dict[str, int] = {'graph': 0, 'visual': 0, 'frame': 0}

    def family_for(item: dict[str, Any]) -> str:
        source = str(item.get('source') or '')
        if source in visual_sources:
            return 'visual'
        if source in frame_sources:
            return 'frame'
        if source in graph_sources:
            return 'graph'
        return 'other'

    def add(item: dict[str, Any], *, respect_quota: bool) -> None:
        if len(selected) >= limit:
            return
        item_id = str(item.get('id') or '')
        if not item_id or item_id in seen_ids:
            return
        family = family_for(item)
        if respect_quota:
            quota = family_quotas.get(family)
            if quota is not None and family_counts.get(family, 0) >= quota:
                return
        selected.append(item)
        seen_ids.add(item_id)
        if family in family_counts:
            family_counts[family] = family_counts.get(family, 0) + 1
    for item in items:
        add(item, respect_quota=True)
    for item in items:
        add(item, respect_quota=False)
    return selected

def _unified_evidence_line(index: int, item: dict[str, Any]) -> str:
    return f"unified={index} source={item.get('source')} rank={item.get('rank')} score={item.get('score')} support={item.get('support_kind')} :: {item.get('text')}"

def _unified_evidence_aggregation_line(index: int, item: dict[str, Any]) -> str:
    return f"unified_group={index} source={item.get('source')} support={item.get('support_kind')} score={item.get('score')} fact={item.get('text')}"

def _unified_candidate_facts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        facts.append({'fact_type': 'unified_evidence', 'fact_text': str(item.get('text') or ''), 'binding_strength': item.get('support_kind'), 'support_examples': [str(item.get('text') or '')], 'support_ranks': [index], 'source': item.get('source'), 'score': item.get('score')})
    return facts

def _unified_source_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        source = str(item.get('source') or 'unknown')
        counts[source] = counts.get(source, 0) + 1
    return counts

def _rank_score(index: int, *, base: float, step: float) -> float:
    return max(0.0, float(base) - float(max(0, index - 1)) * float(step))

def _binding_strength_bonus(value: Any) -> float:
    normalized = str(value or '').strip().lower()
    if normalized in {'direct', 'strong', 'target_bound', 'target-bound'}:
        return 8.0
    if normalized in {'medium', 'indirect'}:
        return 3.0
    return 0.0

def _similarity_bonus(value: Any) -> float:
    text = str(value or '')
    matches = re.findall('(?:sim|max_sim|score)=([0-9]+(?:\\.[0-9]+)?)', text)
    if not matches:
        return 0.0
    try:
        best = max((float(item) for item in matches))
    except ValueError:
        return 0.0
    return min(8.0, best * 10.0)

def _support_kind_from_text(value: Any, *, default: str) -> str:
    text = str(value or '')
    match = re.search('\\bkind=([A-Za-z0-9_\\-]+)', text)
    if match:
        return match.group(1)
    return default

def _answer_candidate_line(index: int, item: dict[str, Any]) -> str:
    support_ids = _ordered_unique_strings(item.get('support_ids', []))
    parts = [f'candidate={index}', f"type={item.get('answer_type', 'unknown')}", f"confidence={item.get('confidence', 'low')}"]
    if support_ids:
        parts.append('support_ids=' + ','.join(support_ids[:6]))
    support_text = _repair_response_text(str(item.get('support_text') or '').strip())
    reasoning = _repair_response_text(str(item.get('reasoning') or '').strip())
    suffix_parts = [f"answer={item.get('answer')}", f'support={support_text}' if support_text else '', f'reason={reasoning}' if reasoning else '']
    return ' '.join(parts) + ' :: ' + ' | '.join((value for value in suffix_parts if value))

def _bundle_with_llm_evidence_selection(llm_ready_bundle: dict[str, Any], *, llm_selection: dict[str, Any]) -> dict[str, Any]:
    selected_evidence = [item for item in llm_selection.get('selected_evidence', []) if isinstance(item, dict)]
    selected_lines = [_llm_selected_evidence_line(index, item) for index, item in enumerate(selected_evidence, start=1)]
    updated = dict(llm_ready_bundle)
    updated['llm_evidence_selection'] = llm_selection
    updated['llm_selected_evidence_lines'] = selected_lines
    structured = dict(updated.get('structured_graph_context', {}))
    structured['llm_evidence_selection'] = llm_selection
    updated['structured_graph_context'] = structured
    context_lines = [str(updated.get('llm_context_text', '')).strip()]
    if selected_lines:
        context_lines.append('Generic LLM evidence selection:')
        context_lines.extend((f'- {line}' for line in selected_lines))
    if llm_selection.get('answer_cue'):
        context_lines.append(f"Selected answer cue: {llm_selection['answer_cue']}")
    if llm_selection.get('insufficient_evidence'):
        reason = str(llm_selection.get('insufficiency_reason') or '').strip()
        context_lines.append('Selected evidence marked insufficient' + (f': {reason}' if reason else '.'))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    return updated

def _llm_selected_evidence_line(index: int, item: dict[str, Any]) -> str:
    parts = [f'selected={index}', f"id={item.get('id')}", f"source={item.get('source')}"]
    if item.get('evidence_rank'):
        parts.append(f"evidence_rank={item.get('evidence_rank')}")
    if item.get('support_ranks'):
        parts.append('support_ranks=' + ','.join((str(value) for value in item.get('support_ranks', [])[:6])))
    if item.get('binding_strength'):
        parts.append(f"binding={item.get('binding_strength')}")
    return ' '.join(parts) + f" :: {item.get('text')}"

def _graph_result_is_position_question(*, question: str, graph_result: dict[str, Any]) -> bool:
    for plan in _graph_result_query_plans(graph_result):
        if _query_plan_is_position(plan):
            return True
    for container in (graph_result, graph_result.get('retrieval_bundle', {})):
        if isinstance(container, dict):
            query_family = str(container.get('query_family') or '').strip().lower()
            if query_family == 'position':
                return True
    normalized_question = ' '.join(str(question or '').lower().split())
    return '[position]' in normalized_question

def _compact_structured_graph_context_for_prompt(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items() if str(key) in _PROMPT_STRUCTURED_CONTEXT_KEYS}

def _compact_llm_context_text_for_prompt(llm_ready_bundle: dict[str, Any], *, max_evidence_lines: int, max_focus_frames: int, max_scene_lines: int) -> str:
    context_lines = _compact_context_header_lines(llm_ready_bundle.get('llm_context_text', ''))
    answer_brief = str(llm_ready_bundle.get('answer_brief') or '').strip()
    if answer_brief:
        context_lines.extend(['Answer brief:', answer_brief])
    _append_prompt_section(context_lines, 'Evidence lines:', llm_ready_bundle.get('evidence_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Focus frames:', llm_ready_bundle.get('focus_frame_lines', []), limit=max_focus_frames)
    _append_prompt_section(context_lines, 'Scene summaries:', llm_ready_bundle.get('scene_lines', []), limit=max_scene_lines)
    _append_prompt_section(context_lines, 'Temporal chains:', llm_ready_bundle.get('temporal_chain_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Temporal span candidates:', llm_ready_bundle.get('temporal_span_candidate_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Evidence aggregation:', llm_ready_bundle.get('evidence_aggregation_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Question IR execution:', llm_ready_bundle.get('question_ir_execution', {}).get('execution_lines', []) if isinstance(llm_ready_bundle.get('question_ir_execution'), dict) else [], limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Generic LLM evidence selection:', llm_ready_bundle.get('llm_selected_evidence_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Generic answer candidate extraction:', llm_ready_bundle.get('answer_candidate_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Atomic answer facts:', llm_ready_bundle.get('atomic_answer_fact_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS evidence:', llm_ready_bundle.get('visual_faiss_evidence_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS unified graph candidates:', llm_ready_bundle.get('visual_faiss_unified_candidate_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS evidence-layer local packs:', llm_ready_bundle.get('visual_faiss_evidence_layer_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS role-aware candidate subjects:', llm_ready_bundle.get('visual_faiss_role_aware_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS role-aware auxiliary role checks:', llm_ready_bundle.get('visual_faiss_role_aware_aux_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS object-local facts:', llm_ready_bundle.get('visual_faiss_object_local_fact_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS GER-like bound facts:', llm_ready_bundle.get('visual_faiss_ger_bound_fact_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Visual FAISS GER-like evidence:', llm_ready_bundle.get('visual_faiss_ger_evidence_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Frame FAISS evidence:', llm_ready_bundle.get('frame_faiss_evidence_lines', []), limit=max_evidence_lines)
    _append_prompt_section(context_lines, 'Merged-frame FAISS evidence:', llm_ready_bundle.get('merged_frame_faiss_evidence_lines', []), limit=max_evidence_lines)
    return '\n'.join(context_lines)

def _append_prompt_section(context_lines: list[str], title: str, values: Any, *, limit: int) -> None:
    lines = _ordered_unique_strings(values)[:max(0, limit)]
    if not lines:
        return
    context_lines.append(title)
    context_lines.extend((f'- {line}' for line in lines))

def _compact_context_header_lines(context_text: Any) -> list[str]:
    lines: list[str] = []
    for raw_line in str(context_text or '').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match('^[A-Za-z][A-Za-z ]{0,30}:', line) and line.split(':', 1)[0] in {'Video', 'Query type', 'Filters', 'Summary'}:
            lines.append(line)
        elif lines:
            break
    return lines

def _graph_result_query_plans(graph_result: dict[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    retrieval_bundle = graph_result.get('retrieval_bundle', {})
    raw_payload = retrieval_bundle.get('raw_payload', {}) if isinstance(retrieval_bundle, dict) else {}
    for candidate in (graph_result.get('query_plan'), retrieval_bundle.get('query_plan') if isinstance(retrieval_bundle, dict) else None, raw_payload.get('query_plan') if isinstance(raw_payload, dict) else None):
        if isinstance(candidate, dict) and candidate not in plans:
            plans.append(candidate)
    return plans

def _query_plan_is_position(plan: dict[str, Any]) -> bool:
    query_family = str(plan.get('query_family') or '').strip().lower()
    if query_family == 'position':
        return True
    compiler = plan.get('question_compiler', {})
    if isinstance(compiler, dict):
        answer_type = str(compiler.get('answer_type') or '').strip().lower()
        if answer_type == 'position':
            return True
        ranking_policy = compiler.get('ranking_policy', [])
        if isinstance(ranking_policy, list) and 'target_event_time_then_pose_render' in {str(item).strip() for item in ranking_policy}:
            return True
    return str(plan.get('compiler_answer_type') or '').strip().lower() == 'position'

def build_graph_qa_responder(config: GraphQAConfig, *, mode: str) -> GraphQAResponder | None:
    normalized_mode = mode.strip().lower()
    if normalized_mode == 'graph_only':
        return None
    if normalized_mode == 'mock':
        return MockGraphQAResponder()
    if normalized_mode == 'live':
        return OpenAICompatibleGraphQAResponder(config)
    raise ValueError(f'Unsupported graph QA mode: {mode}')

def _ordered_unique_strings(values: Any) -> list[str]:
    items = values if isinstance(values, list) else list(values or [])
    ordered: list[str] = []
    seen: set[str] = set()
    for value in items:
        normalized = _repair_response_text(str(value).strip())
        if not normalized or normalized in seen:
            continue
        ordered.append(normalized)
        seen.add(normalized)
    return ordered

def _ordered_unique_ints(values: Any) -> list[int]:
    items = values if isinstance(values, list) else list(values or [])
    ordered: list[int] = []
    seen: set[int] = set()
    for value in items:
        if not isinstance(value, int) or value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered

def _allowed_time_anchor_ids(time_adaptation: dict[str, Any]) -> set[str]:
    candidates = time_adaptation.get('relative_time_candidates', [])
    if not isinstance(candidates, list):
        return set()
    allowed: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        anchor_id = str(candidate.get('anchor_id') or '').strip()
        if anchor_id:
            allowed.add(anchor_id)
    return allowed

def _open_target_relation_image_parts(llm_ready_bundle: dict[str, Any], *, evidence_demand: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    del llm_ready_bundle, evidence_demand
    return ([], [])

def _reasoning_exhausted_without_content(raw_response: dict[str, Any]) -> str | None:
    choice = _first_choice(raw_response)
    message = choice.get('message') if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        return None
    content = message.get('content')
    reasoning_content = message.get('reasoning_content')
    finish_reason = str(choice.get('finish_reason') or '').strip().lower()
    if finish_reason == 'length' and isinstance(content, str) and (not content.strip()) and isinstance(reasoning_content, str) and reasoning_content.strip():
        return 'Reasoning output exhausted the response budget before JSON content'
    return None

def _first_choice(raw_response: dict[str, Any]) -> dict[str, Any]:
    choices = raw_response.get('choices')
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return choices[0]
    return {}

def _append_json_retry_instruction(payload: dict[str, Any]) -> None:
    messages = payload.get('messages')
    if not isinstance(messages, list):
        return
    messages.append({'role': 'user', 'content': 'Retry instruction: return the final answer now as strict JSON only. Do not include reasoning, analysis, markdown, or prose outside the JSON object. Use exactly these keys: answer_text, used_focus_frames, used_evidence_ranks, used_time_anchor_ids, uncertainty.'})

def _backfill_ordered_values(*, current: list[Any], available: list[Any], target_count: int) -> list[Any]:
    if target_count <= 0:
        return list(current)
    merged = list(current)
    seen = set(current)
    for value in available:
        if len(merged) >= target_count:
            break
        if value in seen:
            continue
        merged.append(value)
        seen.add(value)
    return merged

def _normalize_selected_evidence_lines(*, current: list[str], available: list[str], target_count: int, min_count: int, answer_text: str) -> list[str]:
    if target_count <= 0:
        return list(current)
    scored_available = {line: _evidence_line_relevance_score(answer_text, line) for line in available}
    positive_available_count = sum((1 for score in scored_available.values() if score > 0))
    merged = list(current)
    if positive_available_count >= max(1, min_count):
        relevant_current = [line for line in merged if scored_available.get(line, _evidence_line_relevance_score(answer_text, line)) > 0]
        if len(relevant_current) >= min_count:
            merged = relevant_current
    seen = set(merged)
    for value in available:
        if len(merged) >= target_count:
            break
        if value in seen:
            continue
        if scored_available.get(value, 0) <= 0 and len(merged) >= min_count:
            continue
        merged.append(value)
        seen.add(value)
    if len(merged) >= min_count:
        return merged
    return _backfill_ordered_values(current=merged, available=available, target_count=min_count)

def _evidence_line_relevance_score(answer_text: str, evidence_line: str) -> int:
    if _action_phrase_conflicts(answer_text, evidence_line):
        return 0
    answer_tokens = _content_tokens(answer_text)
    evidence_tokens = _content_tokens(evidence_line)
    if not answer_tokens or not evidence_tokens:
        return 0
    shared = answer_tokens & evidence_tokens
    if not shared:
        return 0
    numeric_bonus = sum((2 for token in shared if token.isdigit()))
    anchor_bonus = sum((2 for token in shared if token in {'appear', 'disappear', 'basketball', 'scoreboard', 'holding', 'dribbling', 'passing', 'near', 'looking', 'watching', 'scene'}))
    return len(shared) + numeric_bonus + anchor_bonus

def _action_phrase_conflicts(answer_text: str, evidence_line: str) -> bool:
    answer_phrases = _extract_action_phrase_tokens(answer_text)
    evidence_phrases = _extract_action_phrase_tokens(evidence_line)
    if not answer_phrases or not evidence_phrases:
        return False
    for verb, answer_objects in answer_phrases.items():
        evidence_objects = evidence_phrases.get(verb, [])
        if not evidence_objects:
            continue
        for answer_object in answer_objects:
            for evidence_object in evidence_objects:
                if answer_object and evidence_object and (not answer_object & evidence_object):
                    return True
    return False

def _extract_action_phrase_tokens(text: str) -> dict[str, list[set[str]]]:
    phrases: dict[str, list[set[str]]] = {}
    pattern = re.compile('\\b(holding|dribbling|shooting|wearing|running|standing|looking|guarding|watching|passing)\\b([^.;,:()]*)')
    for match in pattern.finditer(str(text or '').lower()):
        verb = match.group(1)
        tail = match.group(2)
        object_tokens = {token for token in re.findall('[a-z0-9]+', tail) if token not in {'the', 'a', 'an', 'and', 'or', 'with', 'at', 'from', 'to', 'again', 'also', 'is', 'are', 'on', 'in', 'of', 'man', 'woman', 'person', 'human', 'entity'}}
        if not object_tokens:
            continue
        phrases.setdefault(verb, []).append(object_tokens)
    return phrases

def _content_tokens(text: str) -> set[str]:
    tokens = {token for token in re.findall('[a-z0-9]+', str(text or '').lower()) if token and token not in {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'this', 'that', 'these', 'those', 'from', 'with', 'at', 'in', 'on', 'of', 'to', 'me', 'about', 'it', 'its', 'who', 'what', 'when', 'where', 'around', 'during', 'entity', 'scene', 'rank', 'kind', 'score', 'support', 'time', 'frame', 'frames'}}
    return tokens

def _enrich_answer_text_with_entity_aliases(answer_text: str, structured_graph_context: dict[str, Any]) -> str:
    text = str(answer_text or '').strip()
    if not text:
        return ''
    raw_aliases = structured_graph_context.get('entity_aliases', {})
    if not isinstance(raw_aliases, dict) or not raw_aliases:
        return text
    placeholders: dict[str, str] = {}
    updated = text
    for index, (entity_id, payload) in enumerate(sorted(raw_aliases.items(), key=lambda item: len(str(item[0])), reverse=True), start=1):
        if not isinstance(payload, dict):
            continue
        display_text = str(payload.get('display_text') or '').strip()
        short_id = str(payload.get('short_id') or '').strip()
        stable_name = str(payload.get('stable_name') or '').strip().lower()
        readable_name = str(payload.get('readable_name') or '').strip().lower()
        if not display_text:
            continue
        placeholder = f'__ENTITY_ALIAS_{index}__'
        patterns = [re.escape(str(entity_id))]
        alias_already_present = bool(stable_name and stable_name in updated.lower() or (readable_name and readable_name in updated.lower()))
        if short_id.startswith('entity:') and (not alias_already_present):
            suffix = short_id.split(':', 1)[1]
            patterns.append(f'(?i)\\bentity[: ]{re.escape(suffix)}\\b')
        elif short_id and (not alias_already_present):
            patterns.append(f'(?i)\\b{re.escape(short_id)}\\b')
        replaced = False
        for pattern in patterns:
            candidate, count = re.subn(pattern, placeholder, updated)
            if count > 0:
                updated = candidate
                replaced = True
        if replaced:
            placeholders[placeholder] = display_text
    for placeholder, display_text in placeholders.items():
        updated = updated.replace(placeholder, display_text)
    return updated

def _repair_response_text(text: str) -> str:
    original = str(text or '').strip()
    if not original:
        return ''
    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        stripped = value.strip()
        if stripped and stripped not in seen:
            candidates.append(stripped)
            seen.add(stripped)
    add_candidate(original)
    for source_encoding in ('latin-1', 'cp1252', 'gbk'):
        try:
            add_candidate(original.encode(source_encoding).decode('utf-8'))
        except UnicodeError:
            continue
    best_candidate = original
    best_score = _response_text_score(original)
    for candidate in candidates[1:]:
        candidate_score = _response_text_score(candidate)
        if candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score
    return best_candidate

def _response_text_score(text: str) -> int:
    cjk = sum((1 for char in text if '一' <= char <= '鿿'))
    replacement = text.count('�')
    mojibake_markers = sum((1 for char in text if char in 'ÃÂÐÑÄÖÜæçèéêëìíîïðñòóôõöøùúûüýþÿ'))
    readable = sum((1 for char in text if char.isalnum() or char in string.whitespace or char in '.,!?;:\'"()[]{}<>-_/\\'))
    return cjk * 6 + readable - replacement * 10 - mojibake_markers * 3

def _strip_code_fences(content: str) -> str:
    stripped = str(content or '').strip()
    if not stripped.startswith('```'):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].startswith('```'):
        lines = lines[1:]
    if lines and lines[-1].strip() == '```':
        lines = lines[:-1]
    return '\n'.join(lines).strip()

def _extract_first_json_object(text: str) -> str | None:
    source = str(text or '').strip()
    if not source:
        return None
    start_index = source.find('{')
    if start_index < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start_index, len(source)):
        char = source[index]
        if in_string:
            if escape:
                escape = False
            elif char == '\\':
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            depth += 1
            continue
        if char == '}':
            depth -= 1
            if depth == 0:
                return source[start_index:index + 1]
    return None
