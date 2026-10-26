"""Frozen OpenEQA-owned parity fork of the NAVQA route backbone.

Do not modify this v1 baseline.  Create a new versioned OpenEQA runtime for
strategy changes.
"""
from __future__ import annotations
from dataclasses import replace
import re
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.route_contracts import ROUTE_BINARY_EXISTENCE, ROUTE_COUNT_SET, ROUTE_EGO_MOTION_SIDE_TURN, ROUTE_ENTITY_ATTRIBUTE, ROUTE_ENTITY_RELATION_SPATIAL, ROUTE_OCR_LOGO_TEXT, ROUTE_OPEN_DESCRIPTIVE, ROUTE_TEMPORAL_INTERVAL, ROUTE_TEMPORAL_POINT
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_entity_candidate_packs import fuse_visual_entity_candidate_packs
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_faiss_grounding import VisualFaissConfig, fuse_visual_faiss_ger_evidence
QUESTION_ROUTER_HYBRID_IR_QA_ID = 'IR-QA-F-question-router-v2-hybrid-backbone'
QUESTION_ROUTER_SPECIALIZED_IR_QA_ID = 'IR-QA-F-question-router-v3-specialized-backbone'
QUESTION_ROUTER_MAINLINE_IR_QA_ID = 'IR-QA-F-question-router-v4-mainline'
QUESTION_ROUTER_HYBRID_CONTRACT = 'navqa_question_route_hybrid_backbone_v1'
QUESTION_ROUTER_SPECIALIZED_CONTRACT = 'navqa_question_route_specialized_backbone_v1'
BACKBONE_GER_V3_BINDING = 'ger_v3_binding'
BACKBONE_ENTITY_PACK_SCORE_CONTRACT = 'entity_pack_score_contract'
BACKBONE_COUNT_CANDIDATE_SET = 'count_candidate_set'
BACKBONE_GER_OBJECT_LOCAL_TEMPORAL = 'ger_object_local_temporal'
BACKBONE_GER_TEMPORAL_POINT = 'ger_temporal_point_candidate'
BACKBONE_GER_ATTRIBUTE_OWNER = 'ger_attribute_owner'
BACKBONE_GER_OBJECT_LOCAL_RELATION = 'ger_object_local_relation'
BACKBONE_GER_FALLBACK_EGO_MOTION = 'ger_fallback_ego_motion'
BUILDER_GER_BINDING = 'ger_binding'
BUILDER_ENTITY_PACK_SCORE_CONTRACT = 'entity_pack_score_contract'
BUILDER_COUNT_CANDIDATE_SET = 'count_candidate_set'
STABLE_GER_ROUTE_IDS = frozenset({ROUTE_BINARY_EXISTENCE, ROUTE_TEMPORAL_POINT, ROUTE_OPEN_DESCRIPTIVE, ''})
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_COUNT_META_TERMS = {'answer_entity', 'candidate', 'count', 'during', 'event_interval', 'occurrence', 'same_time_window', 'target', 'time_window'}
_COUNT_META_TOKENS = {'all', 'answer', 'candidate', 'count', 'during', 'entity', 'event', 'in', 'interval', 'occurrence', 'same', 'target', 'time', 'total', 'window'}
_TEMPORAL_INTERVAL_STOP_TOKENS = {'a', 'about', 'after', 'along', 'am', 'an', 'and', 'are', 'around', 'at', 'been', 'before', 'did', 'do', 'does', 'during', 'for', 'from', 'get', 'getting', 'have', 'how', 'i', 'in', 'interval', 'it', 'long', 'me', 'of', 'on', 'the', 'there', 'to', 'took', 'were', 'you'}
_TEMPORAL_STATE_CONFIRMATION_TOKENS = {'inside', 'enter', 'entered', 'entering'}
_TEMPORAL_TRAVEL_TO_TOKENS = {'arrive', 'arrived', 'arrival', 'get', 'getting', 'got', 'reach', 'reached'}
_TEMPORAL_INTERVAL_MIN_STATE_PAIR_FRAMES = 60
_TEMPORAL_INTERVAL_MAX_STATE_PAIR_FRAMES = 3600
_TEMPORAL_POINT_MIN_SELECTED_SCORE = 23.0
_TEMPORAL_POINT_WINDOW_TOLERANCE_SECONDS = 120.0
_TEMPORAL_POINT_SELECTED_OPERATIONS = frozenset({'occurrence'})
_TEMPORAL_POINT_NEAR_FRAME_GAP = 60
_COUNT_QUERY_STOP_TOKENS = _COUNT_META_TOKENS | {'a', 'an', 'are', 'as', 'at', 'be', 'been', 'being', 'by', 'can', 'could', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'him', 'how', 'i', 'it', 'its', 'let', 'letting', 'me', 'of', 'on', 'or', 'out', 'she', 'that', 'the', 'their', 'them', 'they', 'this', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'with', 'you', 'your'}
_COUNT_TARGET_TERMINATORS = {'am', 'are', 'can', 'could', 'did', 'do', 'does', 'had', 'has', 'have', 'is', 'was', 'were', 'will', 'would'}
_COUNT_QUANTITY_WORDS = {'one': 1, 'single': 1, 'two': 2, 'double': 2, 'pair': 2, 'couple': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
_EGO_TURN_TOKENS = {'turn', 'turned', 'turning', 'around', 'left', 'right'}
_EGO_TURN_PREDICATE_TOKENS = {'turn', 'turned', 'turning', 'u'}
_SPATIAL_RELATION_ONLY_TOKENS = {'left', 'right', 'of', 'to', 'the', 'near', 'has', 'around', 'adjacent'}

def hybrid_visual_faiss_config(config: VisualFaissConfig, *, question_ir: dict[str, Any] | None, specialized: bool=False) -> VisualFaissConfig:
    """Adjust FAISS query construction for the selected route backbone."""
    policy = question_route_backbone_policy(_question_route(question_ir), specialized=specialized)
    if policy['visual_faiss_entity_pack_query_enabled']:
        return replace(config, entity_pack_enabled=True, entity_pack_score_contract_enabled=policy['entity_pack_score_contract_enabled'], entity_pack_semantic_align_enabled=policy['entity_pack_semantic_align_enabled'], entity_pack_semantic_gated_role_enabled=policy['entity_pack_semantic_gated_role_enabled'])
    return replace(config, entity_pack_enabled=False, entity_pack_score_contract_enabled=False, entity_pack_semantic_align_enabled=False, entity_pack_semantic_gated_role_enabled=False)

def question_route_backbone_policy(question_route: dict[str, Any] | None, *, specialized: bool=False) -> dict[str, Any]:
    route_id = str((question_route or {}).get('route_id') or '').strip()
    if route_id == ROUTE_ENTITY_ATTRIBUTE:
        return _policy(BACKBONE_ENTITY_PACK_SCORE_CONTRACT, 'candidate_pack', 'entity_attribute uses target-entity candidate packs with direct attributes and window-aligned score-contract ranking.', evidence_builder_id=BUILDER_ENTITY_PACK_SCORE_CONTRACT, binding_enabled=False, visual_faiss_entity_pack_query_enabled=True, entity_pack_score_contract_enabled=True, entity_pack_semantic_align_enabled=False, entity_pack_semantic_gated_role_enabled=False, route_strategy_scope='route_specific')
    if route_id == ROUTE_COUNT_SET:
        return _policy(BACKBONE_COUNT_CANDIDATE_SET, 'count_candidate_set', 'count_set uses a policy-specific candidate set before QA.' if specialized else 'count_set needs an explicit deduplicated candidate set before QA.', evidence_builder_id=BUILDER_COUNT_CANDIDATE_SET, binding_enabled=False, visual_faiss_entity_pack_query_enabled=True, count_candidate_set_enabled=True, specialized_strategy_applied=bool(specialized), route_strategy_scope='route_specific')
    if route_id == ROUTE_TEMPORAL_INTERVAL:
        return _policy(BACKBONE_GER_OBJECT_LOCAL_TEMPORAL, 'ger_object_local_fact', 'temporal_interval keeps GER evidence and adds start/end boundary candidates.' if specialized else 'temporal_interval keeps GER evidence plus object-local facts for boundary clues.', object_local_facts_enabled=True, temporal_interval_boundary_enabled=bool(specialized), specialized_strategy_applied=bool(specialized), route_strategy_scope='route_specific')
    if route_id == ROUTE_TEMPORAL_POINT and specialized:
        return _policy(BACKBONE_GER_TEMPORAL_POINT, 'temporal_point_candidate', 'temporal_point keeps GER evidence and adds route-scoped point candidates.', temporal_point_candidate_enabled=True, specialized_strategy_applied=True, route_strategy_scope='route_specific')
    if route_id == ROUTE_OCR_LOGO_TEXT:
        return _policy(BACKBONE_GER_ATTRIBUTE_OWNER, 'ger_bound_fact', 'ocr_logo_text keeps GER binding and owner-aware HSGG attributes.', attribute_owner_enabled=True, route_strategy_scope='route_specific')
    if route_id == ROUTE_ENTITY_RELATION_SPATIAL:
        return _policy(BACKBONE_GER_OBJECT_LOCAL_RELATION, 'ger_object_local_fact', 'spatial relation questions need object-local relation and bbox context.', object_local_facts_enabled=True, route_strategy_scope='route_specific')
    if route_id == ROUTE_EGO_MOTION_SIDE_TURN:
        return _policy(BACKBONE_GER_FALLBACK_EGO_MOTION, 'ger_bound_fact', 'ego motion lacks a robust visual entity pack; preserve broad graph/GER evidence.', route_strategy_scope='route_specific')
    if route_id in STABLE_GER_ROUTE_IDS:
        return _policy(BACKBONE_GER_V3_BINDING, 'ger_bound_fact', 'existence/time/open routes use the stable GER-v3-binding wide evidence backbone.', route_strategy_scope='stable_shared')
    return _policy(BACKBONE_GER_V3_BINDING, 'ger_bound_fact', f'unknown route {route_id!r}; fallback to stable GER-v3-binding.', route_strategy_scope='fallback')

def fuse_question_route_hybrid_backbone(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, config: VisualFaissConfig, question: str, question_ir: dict[str, Any] | None, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None, specialized: bool=False, retrieval_window: dict[str, Any] | None=None) -> dict[str, Any]:
    question_ir = question_ir or {}
    question_route = _question_route(question_ir)
    policy = question_route_backbone_policy(question_route, specialized=specialized)
    builder_id = str(policy['evidence_builder_id'])
    if builder_id == BUILDER_ENTITY_PACK_SCORE_CONTRACT:
        graph_answer = fuse_visual_entity_candidate_packs(graph_answer, visual_payload, graph=graph, object_limit=config.ger_object_limit, neighbor_limit=config.local_subgraph_neighbor_limit, question=question, question_ir=question_ir, semantic_align_enabled=config.entity_pack_semantic_align_enabled, semantic_gated_role_enabled=config.entity_pack_semantic_gated_role_enabled, score_contract_enabled=True, retrieval_window=retrieval_window, semantic_text_encoder=semantic_text_encoder)
    elif builder_id == BUILDER_COUNT_CANDIDATE_SET:
        graph_answer = fuse_count_candidate_set_pack(graph_answer, visual_payload, graph=graph, object_limit=config.ger_object_limit, question=question, question_ir=question_ir, specialized=specialized, retrieval_window=retrieval_window)
    else:
        graph_answer = fuse_visual_faiss_ger_evidence(graph_answer, visual_payload, graph=graph, object_limit=config.ger_object_limit, context_radius=config.ger_context_radius, context_limit=config.ger_context_limit, binding_enabled=bool(policy['binding_enabled']), attribute_owner_enabled=bool(policy['attribute_owner_enabled']), object_local_facts_enabled=bool(policy['object_local_facts_enabled']), question=question, question_ir=question_ir)
        if policy['temporal_interval_boundary_enabled']:
            graph_answer = attach_temporal_interval_boundary_pack(graph_answer, graph=graph, question=question, question_ir=question_ir, retrieval_window=retrieval_window)
        if policy['temporal_point_candidate_enabled']:
            graph_answer = attach_temporal_point_candidate_pack(graph_answer, visual_payload, graph=graph, question=question, question_ir=question_ir, retrieval_window=retrieval_window)
    return attach_hybrid_backbone_metadata(graph_answer, policy=policy, question_route=question_route, specialized=specialized)

def fuse_count_candidate_set_pack(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, object_limit: int, question: str, question_ir: dict[str, Any] | None, specialized: bool=False, retrieval_window: dict[str, Any] | None=None) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    count_policy = _count_policy(question, question_ir or {}) if specialized else 'entity_set_count'
    filter_terms = _filter_terms(question_ir or {})
    effective_retrieval_window = _count_effective_retrieval_window(question, question_ir or {}, count_policy=count_policy, retrieval_window=retrieval_window)
    excluded_items: list[dict[str, Any]] = []
    candidate_limit = _count_candidate_limit(object_limit, count_policy)
    target_terms = _count_target_terms(question, question_ir or {})
    candidates = _count_candidates(visual_payload, graph=graph, question_ir=question_ir or {}, limit=candidate_limit, count_policy=count_policy, filter_terms=filter_terms, excluded_items=excluded_items, retrieval_window=effective_retrieval_window, target_terms=target_terms)
    if count_policy == 'event_episode_count':
        candidates = _event_episode_count_candidates(graph, visual_payload=visual_payload, question=question, question_ir=question_ir or {}, limit=candidate_limit, retrieval_window=effective_retrieval_window, excluded_items=excluded_items)
    elif count_policy == 'ego_action_count':
        candidates = _ego_action_count_candidates(graph, question=question, question_ir=question_ir or {}, limit=candidate_limit, retrieval_window=effective_retrieval_window)
    status = 'candidate_set_available' if candidates else 'insufficient_candidate_set'
    explicit_quantity_answer = _count_explicit_quantity_answer(candidates)
    line = _count_candidate_set_line(count=len(candidates), target_terms=target_terms, candidates=candidates, status=status, count_policy=count_policy, filter_terms=filter_terms, explicit_quantity_answer=explicit_quantity_answer)
    answer_candidate_line = _count_answer_candidate_line(count=len(candidates), status=status, count_policy=count_policy, support_text=line, explicit_quantity_answer=explicit_quantity_answer)
    dedupe_key = _count_dedupe_key(count_policy)
    window_bounds = _retrieval_window_bounds(effective_retrieval_window)
    time_window = _time_window_metadata(effective_retrieval_window, window_bounds)
    fact = {'fact_type': 'count_candidate_set', 'fact_text': line, 'count': len(candidates), 'count_policy': count_policy, 'filter_terms': filter_terms, 'candidate_limit': candidate_limit, 'candidate_limit_source': 'count_route_uncapped_policy', 'dedupe_key': dedupe_key, 'target_terms': target_terms, 'entity_keys': [item['entity_key'] for item in candidates], 'candidate_items': candidates, 'excluded_items_reason': excluded_items[:12], 'evidence_status': status, 'time_window': time_window, 'explicit_quantity_answer': explicit_quantity_answer, 'question': str(question), 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False}
    updated = dict(bundle)
    existing_evidence = _clean_lines(updated.get('evidence_lines', []))
    updated['question_route_count_set_lines'] = [line]
    existing_answer_candidate_lines = _clean_lines(updated.get('answer_candidate_lines', []))
    if candidates:
        updated['answer_candidate_lines'] = _ordered_unique([answer_candidate_line, *existing_answer_candidate_lines])
    else:
        updated['answer_candidate_lines'] = existing_answer_candidate_lines
    updated['answer_candidate_extraction'] = _count_answer_candidate_extraction(count=len(candidates), status=status, count_policy=count_policy, support_text=line, explicit_quantity_answer=explicit_quantity_answer)
    updated['evidence_lines'] = _ordered_unique([line, *existing_evidence])
    existing_aggregation_lines = _clean_lines(updated.get('evidence_aggregation_lines', []))
    updated['evidence_aggregation_lines'] = _ordered_unique([line, *existing_aggregation_lines])
    aggregation = dict(updated.get('evidence_aggregation', {}) or {})
    aggregation['candidate_facts'] = [fact, *[item for item in aggregation.get('candidate_facts', []) or []]]
    updated['evidence_aggregation'] = aggregation
    updated['question_route_count_set'] = {'contract': 'navqa_question_route_count_candidate_set_v1', 'candidate_unit': 'count_candidate_set', 'count': len(candidates), 'count_policy': count_policy, 'filter_terms': filter_terms, 'candidate_limit': candidate_limit, 'candidate_limit_source': 'count_route_uncapped_policy', 'dedupe_key': dedupe_key, 'target_terms': target_terms, 'candidates': candidates, 'candidate_items': candidates, 'excluded_items_reason': excluded_items[:12], 'evidence_status': status, 'time_window': time_window, 'explicit_quantity_answer': explicit_quantity_answer, 'uses_gold_answers': False, 'uses_question_route_branches': True, 'uses_question_type_branches': False}
    updated['visual_faiss_grounding'] = visual_payload
    structured = dict(updated.get('structured_graph_context', {}) or {})
    structured['question_route_count_set_lines'] = [line]
    structured['question_route_count_set'] = updated['question_route_count_set']
    structured['answer_candidate_extraction'] = updated['answer_candidate_extraction']
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    return graph_answer

def attach_hybrid_backbone_metadata(graph_answer: dict[str, Any], *, policy: dict[str, Any], question_route: dict[str, Any], specialized: bool=False) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    metadata = {'contract': QUESTION_ROUTER_SPECIALIZED_CONTRACT if specialized else QUESTION_ROUTER_HYBRID_CONTRACT, 'enabled': True, 'uses_question_route_hybrid_backbone': True, 'uses_question_route_specialized_backbone': bool(specialized), 'uses_gold_answers': False, 'uses_question_route_branches': True, 'uses_question_type_branches': False, 'question_route_id': str(question_route.get('route_id') or ''), **policy}
    updated = dict(bundle)
    updated['question_route_hybrid_backbone'] = metadata
    graph_answer['llm_ready_bundle'] = updated
    return graph_answer

def attach_temporal_interval_boundary_pack(graph_answer: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, retrieval_window: dict[str, Any] | None=None) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    boundary_pack = _temporal_interval_boundary_pack(graph_answer, graph=graph, question=question, question_ir=question_ir or {}, retrieval_window=retrieval_window)
    lines = _temporal_interval_boundary_lines(boundary_pack)
    updated = dict(bundle)
    existing_evidence = _clean_lines(updated.get('evidence_lines', []))
    updated['question_route_temporal_interval_lines'] = lines
    updated['evidence_lines'] = _ordered_unique([*lines, *existing_evidence])
    existing_aggregation_lines = _clean_lines(updated.get('evidence_aggregation_lines', []))
    updated['evidence_aggregation_lines'] = _ordered_unique([*lines, *existing_aggregation_lines])
    updated['question_route_temporal_interval'] = boundary_pack
    structured = dict(updated.get('structured_graph_context', {}) or {})
    structured['question_route_temporal_interval'] = boundary_pack
    structured['question_route_temporal_interval_lines'] = lines
    paired_candidates = list(boundary_pack.get('paired_duration_candidates', []))
    if paired_candidates:
        existing_spans = [item for item in structured.get('temporal_span_candidates', []) or [] if isinstance(item, dict)]
        structured['temporal_span_candidates'] = _dedupe_temporal_candidates([*paired_candidates, *existing_spans])
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    return graph_answer

def attach_temporal_point_candidate_pack(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, retrieval_window: dict[str, Any] | None=None) -> dict[str, Any]:
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    point_pack = _temporal_point_candidate_pack(visual_payload, graph=graph, question=question, question_ir=question_ir or {}, retrieval_window=retrieval_window)
    lines = _temporal_point_candidate_lines(point_pack)
    updated = dict(bundle)
    existing_evidence = _clean_lines(updated.get('evidence_lines', []))
    updated['question_route_temporal_point'] = point_pack
    updated['question_route_temporal_point_lines'] = lines
    if lines:
        updated['evidence_lines'] = _ordered_unique([*lines, *existing_evidence])
        existing_aggregation_lines = _clean_lines(updated.get('evidence_aggregation_lines', []))
        updated['evidence_aggregation_lines'] = _ordered_unique([*lines, *existing_aggregation_lines])
    selected = point_pack.get('selected_candidate')
    if isinstance(selected, dict) and selected:
        answer_line = _temporal_point_answer_candidate_line(selected)
        existing_answer_candidate_lines = _clean_lines(updated.get('answer_candidate_lines', []))
        updated['answer_candidate_lines'] = _ordered_unique([answer_line, *existing_answer_candidate_lines])
    structured = dict(updated.get('structured_graph_context', {}) or {})
    structured['question_route_temporal_point'] = point_pack
    structured['question_route_temporal_point_lines'] = lines
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    return graph_answer

def _policy(backbone_id: str, candidate_unit: str, reason: str, *, evidence_builder_id: str=BUILDER_GER_BINDING, binding_enabled: bool=True, attribute_owner_enabled: bool=False, object_local_facts_enabled: bool=False, temporal_interval_boundary_enabled: bool=False, temporal_point_candidate_enabled: bool=False, visual_faiss_entity_pack_query_enabled: bool=False, entity_pack_score_contract_enabled: bool=False, entity_pack_semantic_align_enabled: bool=False, entity_pack_semantic_gated_role_enabled: bool=False, count_candidate_set_enabled: bool=False, specialized_strategy_applied: bool=False, route_strategy_scope: str='shared') -> dict[str, Any]:
    return {'route_backbone_id': backbone_id, 'route_backbone_reason': reason, 'route_candidate_unit': candidate_unit, 'evidence_builder_id': evidence_builder_id, 'binding_enabled': bool(binding_enabled), 'attribute_owner_enabled': bool(attribute_owner_enabled), 'object_local_facts_enabled': bool(object_local_facts_enabled), 'temporal_interval_boundary_enabled': bool(temporal_interval_boundary_enabled), 'temporal_point_candidate_enabled': bool(temporal_point_candidate_enabled), 'visual_faiss_entity_pack_query_enabled': bool(visual_faiss_entity_pack_query_enabled), 'entity_pack_score_contract_enabled': bool(entity_pack_score_contract_enabled), 'entity_pack_semantic_align_enabled': bool(entity_pack_semantic_align_enabled), 'entity_pack_semantic_gated_role_enabled': bool(entity_pack_semantic_gated_role_enabled), 'count_candidate_set_enabled': bool(count_candidate_set_enabled), 'specialized_strategy_applied': bool(specialized_strategy_applied), 'route_strategy_scope': route_strategy_scope}

def _question_route(question_ir: dict[str, Any] | None) -> dict[str, Any]:
    route = question_ir.get('question_route') if isinstance(question_ir, dict) else None
    return route if isinstance(route, dict) else {}

def _count_candidate_limit(object_limit: int, count_policy: str) -> int:
    base = max(1, int(object_limit))
    if count_policy == 'entity_set_count':
        return max(base, 64)
    if count_policy in {'event_episode_count', 'ego_action_count'}:
        return max(base, 64)
    return base

def _count_filter_tokens(filter_terms: list[str] | None) -> set[str]:
    tokens: set[str] = set()
    for term in filter_terms or []:
        raw = ' '.join(str(term or '').split()).lower()
        if not raw or raw in _COUNT_META_TERMS:
            continue
        term_tokens = set(_tokens(raw))
        cleaned = term_tokens - _COUNT_META_TOKENS
        if not cleaned:
            continue
        tokens.update(cleaned)
    return tokens

def _is_ego_turn_count_event(tokens: set[str], predicate_tokens: set[str]) -> bool:
    if not tokens & _EGO_TURN_TOKENS:
        return False
    if predicate_tokens & _EGO_TURN_PREDICATE_TOKENS:
        return True
    relation_only = predicate_tokens & _SPATIAL_RELATION_ONLY_TOKENS
    if relation_only and (not predicate_tokens - _SPATIAL_RELATION_ONLY_TOKENS):
        return False
    return 'turn' in tokens and bool(predicate_tokens & {'raw', 'ego'})

def _count_candidates(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question_ir: dict[str, Any], limit: int, count_policy: str='entity_set_count', filter_terms: list[str] | None=None, excluded_items: list[dict[str, Any]] | None=None, retrieval_window: dict[str, Any] | None=None, target_terms: list[str] | None=None) -> list[dict[str, Any]]:
    if count_policy != 'entity_set_count':
        return []
    target_tokens = set(_tokens(' '.join(target_terms or _entity_count_target_terms(question_ir))))
    filter_tokens = _count_filter_tokens(filter_terms)
    entity_index = {str(entity.entity_id): entity for entity in graph.entity_nodes}
    anchor_index = _observation_anchor_index(graph)
    window_bounds = _retrieval_window_bounds(retrieval_window)
    prefer_mapped_rows = _visual_payload_has_mapped_target_rows(visual_payload, target_tokens=target_tokens, anchor_index=anchor_index)
    buckets: dict[str, dict[str, Any]] = {}
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        query_phrase = str(group.get('query_phrase') or '')
        query_role = str(group.get('query_phrase_role') or '')
        for row in group.get('results', []) or []:
            if not isinstance(row, dict):
                continue
            row_target_tokens = _row_target_tokens(row)
            row_all_tokens = _row_tokens(row) | set(_tokens(query_phrase))
            if target_tokens and (not row_target_tokens & target_tokens):
                _append_excluded(excluded_items, row, reason='target_label_or_direct_attribute_mismatch')
                continue
            if filter_tokens and (not row_all_tokens & filter_tokens):
                _append_excluded(excluded_items, row, reason='constraint_terms_not_observed_on_candidate')
                continue
            entity_id = _row_entity_id(row, anchor_index=anchor_index)
            if prefer_mapped_rows and (not entity_id):
                _append_excluded(excluded_items, row, reason='missing_graph_entity_anchor_for_count')
                continue
            entity = entity_index.get(entity_id)
            if not _row_overlaps_retrieval_window(row, entity, window_bounds):
                _append_excluded(excluded_items, row, reason='outside_retrieval_time_window')
                continue
            stable_id = str(getattr(entity, 'stable_identity_id', '') or '').strip()
            entity_key = stable_id or entity_id or _row_source_key(row)
            if not entity_key:
                continue
            quantity_hint = _quantity_hint_from_text(_candidate_quantity_text(row, entity), target_tokens=target_tokens)
            current = buckets.get(entity_key)
            score = float(row.get('score') or row.get('similarity') or 0.0)
            if current is not None and float(current.get('score') or 0.0) >= score:
                continue
            buckets[entity_key] = {'entity_key': entity_key, 'entity_id': entity_id, 'stable_identity_id': stable_id, 'label': str(getattr(entity, 'label', None) or row.get('label') or ''), 'frame': row.get('graph_frame_index'), 'object_index': row.get('object_index'), 'score': score, 'query_phrase': query_phrase, 'query_phrase_role': query_role, 'visual_attribute': row.get('visual_attribute'), 'count_policy': count_policy, 'dedupe_key': stable_id and 'stable_identity_id' or (entity_id and 'entity_id') or 'source_object_key', 'filter_hits': sorted(row_all_tokens & filter_tokens)[:8], 'quantity_hint': quantity_hint, **_time_range_fields(getattr(entity, 'time_range', None))}
    return sorted(buckets.values(), key=lambda item: (-float(item.get('score') or 0.0), str(item.get('entity_key') or '')))[:limit]

def _event_episode_count_candidates(graph: CanonicalTemporalMemoryGraph, *, visual_payload: dict[str, Any], question: str, question_ir: dict[str, Any], limit: int, retrieval_window: dict[str, Any] | None=None, excluded_items: list[dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    target_tokens = set(_tokens(' '.join(_count_target_terms(question, question_ir)))) - _COUNT_QUERY_STOP_TOKENS
    filter_tokens = _count_filter_tokens(_filter_terms(question_ir))
    action_tokens = _count_action_tokens(question, question_ir, target_tokens)
    query_tokens = (set(_tokens(question)) | target_tokens | filter_tokens | action_tokens) - _COUNT_QUERY_STOP_TOKENS
    window_bounds = _retrieval_window_bounds(retrieval_window)
    buckets: dict[str, dict[str, Any]] = {}
    for node in list(graph.episode_nodes) + list(graph.transition_event_nodes):
        if not _node_overlaps_retrieval_window(node, window_bounds):
            continue
        text = _node_semantic_text(node)
        tokens = set(_tokens(text))
        target_hits = sorted(tokens & target_tokens)
        filter_hits = sorted(tokens & filter_tokens)
        action_hits = sorted(tokens & action_tokens)
        if target_tokens and (not target_hits):
            continue
        if filter_tokens and (not (filter_hits or action_hits)):
            continue
        if action_tokens and (not (action_hits or filter_hits)):
            continue
        meaningful_hits = set(target_hits) | set(filter_hits) | set(action_hits)
        if query_tokens and (not meaningful_hits):
            continue
        key = str(getattr(node, 'episode_id', None) or getattr(node, 'event_id', None) or getattr(node, 'node_id', ''))
        if not key or key in buckets:
            continue
        time_range = getattr(node, 'time_range', None)
        buckets[key] = {'entity_key': key, 'event_id': key, 'label': str(getattr(node, 'label', '') or getattr(node, 'predicate', '') or 'event'), 'frame': getattr(time_range, 'start_frame', None), 'end_frame': getattr(time_range, 'end_frame', None), 'score': float(len(meaningful_hits)), 'count_policy': 'event_episode_count', 'dedupe_key': 'episode_or_event_id', 'target_hits': target_hits[:8], 'filter_hits': filter_hits[:8], 'action_hits': action_hits[:8], **_time_range_fields(time_range)}
    candidates = sorted(buckets.values(), key=lambda item: (-float(item.get('score') or 0.0), str(item.get('entity_key') or '')))[:limit]
    if candidates:
        return candidates
    return _event_target_state_entity_candidates(visual_payload, graph=graph, question=question, question_ir=question_ir, limit=limit, retrieval_window=retrieval_window, excluded_items=excluded_items)

def _ego_action_count_candidates(graph: CanonicalTemporalMemoryGraph, *, question: str, question_ir: dict[str, Any], limit: int, retrieval_window: dict[str, Any] | None=None) -> list[dict[str, Any]]:
    action_tokens = set(_tokens(' '.join([question, *_target_terms(question_ir), *_filter_terms(question_ir)]))) - _COUNT_META_TOKENS
    ego_tokens = action_tokens | _EGO_TURN_TOKENS
    window_bounds = _retrieval_window_bounds(retrieval_window)
    raw_items: list[dict[str, Any]] = []
    for node in list(graph.episode_nodes) + list(graph.transition_event_nodes):
        if not _node_overlaps_retrieval_window(node, window_bounds):
            continue
        text = _node_semantic_text(node)
        tokens = set(_tokens(text))
        predicate_tokens = set(_tokens(' '.join([str(getattr(node, 'predicate', '') or ''), str(getattr(node, 'raw_predicate', '') or ''), str(getattr(node, 'event_type', '') or ''), str(getattr(node, 'source_ref_id', '') or '')])))
        if not _is_ego_turn_count_event(tokens, predicate_tokens):
            continue
        time_range = getattr(node, 'time_range', None)
        frame = getattr(time_range, 'start_frame', None)
        if frame is None:
            continue
        raw_items.append({'entity_key': str(getattr(node, 'episode_id', None) or getattr(node, 'event_id', None) or getattr(node, 'node_id', '')), 'event_id': str(getattr(node, 'episode_id', None) or getattr(node, 'event_id', None) or getattr(node, 'node_id', '')), 'label': str(getattr(node, 'label', '') or getattr(node, 'predicate', '') or 'ego_action'), 'frame': frame, 'end_frame': getattr(time_range, 'end_frame', None), 'score': float(len(tokens & ego_tokens)), 'count_policy': 'ego_action_count', 'dedupe_key': 'ego_action_frame_cluster', 'filter_hits': sorted(tokens & ego_tokens)[:8], **_time_range_fields(time_range)})
    return _cluster_count_items_by_frame(raw_items, max_frame_gap=90)[:max(1, limit)]

def _event_target_state_entity_candidates(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], limit: int, retrieval_window: dict[str, Any] | None=None, excluded_items: list[dict[str, Any]] | None=None) -> list[dict[str, Any]]:
    target_tokens = set(_tokens(' '.join(_count_target_terms(question, question_ir)))) - _COUNT_QUERY_STOP_TOKENS
    if not target_tokens:
        return []
    action_tokens = _count_action_tokens(question, question_ir, target_tokens)
    entity_index = {str(entity.entity_id): entity for entity in graph.entity_nodes}
    anchor_index = _observation_anchor_index(graph)
    window_bounds = _retrieval_window_bounds(retrieval_window)
    prefer_mapped_rows = _visual_payload_has_mapped_target_rows(visual_payload, target_tokens=target_tokens, anchor_index=anchor_index, label_only=True)
    buckets: dict[str, dict[str, Any]] = {}
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        query_phrase = str(group.get('query_phrase') or '')
        query_role = str(group.get('query_phrase_role') or '')
        for row in group.get('results', []) or []:
            if not isinstance(row, dict):
                continue
            row_target_tokens = _row_label_tokens(row)
            row_all_tokens = _row_tokens(row)
            if query_role in {'event_constraint', 'hard_constraint', 'support'}:
                row_all_tokens |= set(_tokens(query_phrase))
            target_hits = sorted(row_target_tokens & target_tokens)
            action_hits = sorted(row_all_tokens & action_tokens)
            if not target_hits:
                _append_excluded(excluded_items, row, reason='event_target_entity_mismatch')
                continue
            if action_tokens and (not action_hits):
                _append_excluded(excluded_items, row, reason='event_state_terms_not_observed_on_candidate')
                continue
            entity_id = _row_entity_id(row, anchor_index=anchor_index)
            if prefer_mapped_rows and (not entity_id):
                _append_excluded(excluded_items, row, reason='missing_graph_entity_anchor_for_count')
                continue
            entity = entity_index.get(entity_id)
            if not _row_overlaps_retrieval_window(row, entity, window_bounds):
                _append_excluded(excluded_items, row, reason='outside_retrieval_time_window')
                continue
            stable_id = str(getattr(entity, 'stable_identity_id', '') or '').strip()
            entity_key = stable_id or entity_id or _row_source_key(row)
            if not entity_key:
                continue
            quantity_hint = _quantity_hint_from_text(_candidate_quantity_text(row, entity), target_tokens=target_tokens)
            score = float(row.get('score') or row.get('similarity') or 0.0)
            current = buckets.get(entity_key)
            if current is not None and float(current.get('score') or 0.0) >= score:
                continue
            buckets[entity_key] = {'entity_key': entity_key, 'entity_id': entity_id, 'stable_identity_id': stable_id, 'label': str(getattr(entity, 'label', None) or row.get('label') or ''), 'frame': row.get('graph_frame_index'), 'object_index': row.get('object_index'), 'score': score, 'query_phrase': query_phrase, 'query_phrase_role': query_role, 'visual_attribute': row.get('visual_attribute'), 'count_policy': 'event_episode_count', 'dedupe_key': stable_id and 'stable_identity_id' or (entity_id and 'entity_id') or 'source_object_key', 'candidate_source': 'event_target_state_visual_entity', 'target_hits': target_hits[:8], 'action_hits': action_hits[:8], 'quantity_hint': quantity_hint, **_time_range_fields(getattr(entity, 'time_range', None))}
    if buckets:
        return sorted(buckets.values(), key=lambda item: (-float(item.get('score') or 0.0), str(item.get('entity_key') or '')))[:limit]
    for entity in graph.entity_nodes:
        if not _time_range_overlaps_retrieval_window(getattr(entity, 'time_range', None), window_bounds):
            continue
        tokens = set(_tokens(_entity_semantic_text(entity)))
        target_hits = sorted(tokens & target_tokens)
        action_hits = sorted(tokens & action_tokens)
        if not target_hits:
            continue
        if action_tokens and (not action_hits):
            continue
        stable_id = str(getattr(entity, 'stable_identity_id', '') or '').strip()
        entity_id = str(getattr(entity, 'entity_id', '') or '').strip()
        entity_key = stable_id or entity_id or str(getattr(entity, 'node_id', '') or '')
        if not entity_key or entity_key in buckets:
            continue
        time_range = getattr(entity, 'time_range', None)
        buckets[entity_key] = {'entity_key': entity_key, 'entity_id': entity_id, 'stable_identity_id': stable_id, 'label': str(getattr(entity, 'label', '') or 'entity'), 'frame': getattr(time_range, 'start_frame', None), 'end_frame': getattr(time_range, 'end_frame', None), 'score': float(len(set(target_hits) | set(action_hits))), 'count_policy': 'event_episode_count', 'dedupe_key': 'stable_identity_id|entity_id', 'candidate_source': 'event_target_state_graph_entity', 'target_hits': target_hits[:8], 'action_hits': action_hits[:8], 'quantity_hint': _quantity_hint_from_text(_entity_semantic_text(entity), target_tokens=target_tokens), **_time_range_fields(time_range)}
    return sorted(buckets.values(), key=lambda item: (-float(item.get('score') or 0.0), str(item.get('entity_key') or '')))[:limit]

def _observation_anchor_index(graph: CanonicalTemporalMemoryGraph) -> dict[tuple[int, int], dict[str, Any]]:
    from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_faiss_grounding import _visual_faiss_observation_anchor_index
    return _visual_faiss_observation_anchor_index(graph)

def _row_entity_id(row: dict[str, Any], *, anchor_index: dict[tuple[int, int], dict[str, Any]]) -> str:
    direct = str(row.get('anchor_entity_id') or row.get('entity_id') or '').strip()
    if direct:
        return direct
    frame = _optional_int(row.get('graph_frame_index'))
    object_index = _optional_int(row.get('object_index'))
    if frame is None or object_index is None:
        return ''
    anchor = anchor_index.get((frame, object_index))
    if isinstance(anchor, dict):
        return str(anchor.get('entity_id') or '').strip()
    return ''

def _row_source_key(row: dict[str, Any]) -> str:
    if row.get('tracklet_hit') and row.get('tracklet_faiss_id') is not None:
        return f"tracklet:{row.get('tracklet_faiss_id')}"
    seq = str(row.get('seq_id') or '').strip()
    frame = row.get('graph_frame_index')
    object_index = row.get('object_index')
    if seq and frame is not None and (object_index is not None):
        return f'object:{seq}:{frame}:{object_index}'
    faiss_id = row.get('faiss_id') or row.get('object_faiss_id')
    return f'faiss:{faiss_id}' if faiss_id is not None else ''

def _target_terms(question_ir: dict[str, Any]) -> list[str]:
    route = _question_route(question_ir)
    values = [*_iter_text_values(route.get('primary_target_terms')), *_iter_text_values(question_ir.get('target_entities')), *_iter_text_values(question_ir.get('target_state'))]
    return _ordered_unique(values)

def _count_target_terms(question: str, question_ir: dict[str, Any]) -> list[str]:
    explicit_entity_terms = _iter_text_values(question_ir.get('target_entities'))
    if explicit_entity_terms:
        return _ordered_unique([*explicit_entity_terms, *_count_question_target_terms(question)])
    return _ordered_unique([*_entity_count_target_terms(question_ir), *_count_question_target_terms(question)])

def _count_question_target_terms(question: str) -> list[str]:
    raw_tokens = _TOKEN_PATTERN.findall(str(question or '').lower())
    start_index: int | None = None
    for index, token in enumerate(raw_tokens):
        if token == 'how' and index + 1 < len(raw_tokens) and (raw_tokens[index + 1] in {'many', 'much'}):
            start_index = index + 2
            break
        if token == 'number' and index + 1 < len(raw_tokens) and (raw_tokens[index + 1] == 'of'):
            start_index = index + 2
            break
    if start_index is None:
        return []
    target_tokens: list[str] = []
    for token in raw_tokens[start_index:]:
        if token in _COUNT_TARGET_TERMINATORS:
            break
        if token in _COUNT_QUERY_STOP_TOKENS:
            continue
        target_tokens.append(token)
        if len(target_tokens) >= 5:
            break
    return [' '.join(target_tokens)] if target_tokens else []

def _count_action_tokens(question: str, question_ir: dict[str, Any], target_tokens: set[str]) -> set[str]:
    target_family = _count_target_token_family(target_tokens)
    raw_tokens = set(_tokens(' '.join([question, *_filter_terms(question_ir)]))) - target_family - _COUNT_QUERY_STOP_TOKENS
    return _count_token_variants(raw_tokens)

def _count_target_token_family(tokens: set[str]) -> set[str]:
    family = _count_token_variants(tokens)
    for token in list(family):
        if len(token) > 2 and (not token.endswith('s')):
            family.add(f'{token}s')
        if len(token) > 3 and token.endswith('s'):
            family.add(token[:-1])
    return family

def _count_token_variants(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for token in list(tokens):
        if len(token) > 4 and token.endswith('ied'):
            expanded.add(token[:-3] + 'y')
        if len(token) > 4 and token.endswith('ed'):
            expanded.add(token[:-2])
            expanded.add(token[:-1])
        if len(token) > 5 and token.endswith('ing'):
            expanded.add(token[:-3])
    return {token for token in expanded if token and token not in _COUNT_QUERY_STOP_TOKENS}

def _row_tokens(row: dict[str, Any]) -> set[str]:
    values = [row.get('label'), row.get('visual_attribute'), row.get('representative_labels'), row.get('hsgg_visual_facts')]
    return set(_tokens(' '.join((_stringify(value) for value in values))))

def _row_target_tokens(row: dict[str, Any]) -> set[str]:
    values = [row.get('label'), row.get('representative_labels'), row.get('visual_attribute')]
    return set(_tokens(' '.join((_stringify(value) for value in values))))

def _row_label_tokens(row: dict[str, Any]) -> set[str]:
    values = [row.get('label'), row.get('representative_labels')]
    return set(_tokens(' '.join((_stringify(value) for value in values))))

def _visual_payload_has_mapped_target_rows(visual_payload: dict[str, Any], *, target_tokens: set[str], anchor_index: dict[tuple[int, int], dict[str, Any]], label_only: bool=False) -> bool:
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        for row in group.get('results', []) or []:
            if not isinstance(row, dict):
                continue
            row_tokens = _row_label_tokens(row) if label_only else _row_target_tokens(row)
            if target_tokens and (not row_tokens & target_tokens):
                continue
            if _row_entity_id(row, anchor_index=anchor_index):
                return True
    return False

def _candidate_quantity_text(row: dict[str, Any], entity: Any | None=None) -> str:
    values = [row.get('label'), row.get('visual_attribute'), row.get('representative_labels')]
    if entity is not None:
        values.extend([getattr(entity, 'label', ''), getattr(entity, 'labels_seen', []), getattr(entity, 'states_seen', []), getattr(entity, 'attributes', {})])
    return _stringify(values)

def _quantity_hint_from_text(text: str, *, target_tokens: set[str]) -> dict[str, Any] | None:
    target_family = _count_target_token_family(target_tokens)
    if not target_family:
        return None
    raw_tokens = _TOKEN_PATTERN.findall(str(text or '').lower())
    for index, token in enumerate(raw_tokens):
        quantity = _quantity_value(token)
        if quantity is None:
            continue
        window = raw_tokens[index + 1:index + 7]
        if not set(_tokens(' '.join(window))) & target_family:
            continue
        return {'quantity': quantity, 'quantity_token': token, 'source_text': ' '.join(raw_tokens[max(0, index - 2):index + 8]), 'uses_gold_answers': False}
    return None

def _quantity_value(token: str) -> int | None:
    if token.isdigit():
        value = int(token)
        return value if 0 < value <= 100 else None
    return _COUNT_QUANTITY_WORDS.get(token)

def _count_explicit_quantity_answer(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    hints: list[dict[str, Any]] = []
    for item in candidates:
        hint = item.get('quantity_hint')
        if not isinstance(hint, dict):
            continue
        quantity = _optional_int(hint.get('quantity'))
        if quantity is None or quantity <= 0:
            continue
        hints.append({**hint, 'quantity': quantity, 'entity_key': item.get('entity_key'), 'label': item.get('label'), 'frame': item.get('frame'), 'score': float(item.get('score') or 0.0)})
    if not hints:
        return None
    grouped: dict[int, list[dict[str, Any]]] = {}
    for hint in hints:
        grouped.setdefault(int(hint['quantity']), []).append(hint)
    quantity, members = sorted(grouped.items(), key=lambda pair: (-len(pair[1]), -max((float(item.get('score') or 0.0) for item in pair[1])), pair[0]))[0]
    best = sorted(members, key=lambda item: (-float(item.get('score') or 0.0), str(item.get('entity_key') or '')))[0]
    return {'quantity': quantity, 'source': 'candidate_explicit_quantity_phrase', 'source_text': best.get('source_text'), 'support_count': len(members), 'entity_key': best.get('entity_key'), 'label': best.get('label'), 'frame': best.get('frame'), 'uses_gold_answers': False}

def _count_candidate_set_line(*, count: int, target_terms: list[str], candidates: list[dict[str, Any]], status: str, count_policy: str='entity_set_count', filter_terms: list[str] | None=None, explicit_quantity_answer: dict[str, Any] | None=None) -> str:
    examples = []
    for item in candidates[:8]:
        label = str(item.get('label') or 'object')
        key = str(item.get('entity_key') or '')
        frame = item.get('frame')
        frame_text = f' frame={frame}' if frame is not None else ''
        examples.append(f'{label}({key}{frame_text})')
    target_text = ','.join(target_terms[:6]) if target_terms else 'unspecified'
    filter_text = ','.join((filter_terms or [])[:6]) if filter_terms else 'none'
    shown_count = min(len(candidates), 8)
    examples_text = '; '.join(examples) if examples else 'none'
    if explicit_quantity_answer:
        quantity = int(explicit_quantity_answer.get('quantity') or count)
        source_text = str(explicit_quantity_answer.get('source_text') or '').strip()
        source_suffix = f' explicit_quantity_source={source_text}' if source_text else ''
        return f'kind=count_candidate_set answer_count={quantity} answer_source=explicit_quantity_phrase candidate_count={count} count_policy={count_policy} target_terms={target_text} filter_terms={filter_text} evidence_status={status} shown_examples={shown_count}/{count} entities={examples_text}{source_suffix}'
    return f'kind=count_candidate_set exact_count={count} count={count} count_policy={count_policy} target_terms={target_text} filter_terms={filter_text} evidence_status={status} shown_examples={shown_count}/{count} entities={examples_text}'

def _count_answer_candidate_line(*, count: int, status: str, count_policy: str, support_text: str, explicit_quantity_answer: dict[str, Any] | None=None) -> str:
    confidence = 'medium' if status == 'candidate_set_available' else 'low'
    answer = count
    reason = f'deduplicated {count_policy} candidate set exact_count'
    if explicit_quantity_answer:
        answer = int(explicit_quantity_answer.get('quantity') or count)
        reason = f'explicit quantity phrase in count candidate; deduplicated_count={count}'
        explicit_source = str(explicit_quantity_answer.get('source_text') or '').strip()
        if explicit_source:
            support_text = f'{support_text} | explicit_quantity_source={explicit_source}'
    return f'candidate=1 type=count confidence={confidence} support_ids=question_route_count_set :: answer={answer} | support={support_text} | reason={reason}'

def _count_answer_candidate_extraction(*, count: int, status: str, count_policy: str, support_text: str, explicit_quantity_answer: dict[str, Any] | None=None) -> dict[str, Any]:
    if status != 'candidate_set_available':
        return {'contract': 'question_route_count_answer_candidate_v1', 'available': False, 'candidates': [], 'insufficient_evidence': True, 'insufficiency_reason': 'count_candidate_set did not find graph-backed candidates', 'uses_gold_answers': False}
    answer = count
    reasoning = f'exact count from deduplicated {count_policy} candidates'
    if explicit_quantity_answer:
        answer = int(explicit_quantity_answer.get('quantity') or count)
        reasoning = f'explicit quantity phrase in a graph-backed count candidate (deduplicated_count={count})'
        support_text = ' | '.join((value for value in [support_text, str(explicit_quantity_answer.get('source_text') or '').strip()] if value))
    return {'contract': 'question_route_count_answer_candidate_v1', 'available': True, 'candidates': [{'answer': str(answer), 'answer_type': 'count', 'support_ids': ['question_route_count_set'], 'support_text': support_text, 'confidence': 'medium', 'reasoning': reasoning}], 'insufficient_evidence': False, 'insufficiency_reason': '', 'uses_gold_answers': False}

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = ' '.join(value.split())
        return [text] if text else []
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target_state', 'relation', 'operation', 'target', 'value'):
            values.extend(_iter_text_values(value.get(key)))
        for key in ('target_entities', 'must_bind_to', 'aliases'):
            values.extend(_iter_text_values(value.get(key)))
        return values
    if isinstance(value, (list, tuple, set)):
        values: list[str] = []
        for item in value:
            values.extend(_iter_text_values(item))
        return values
    text = ' '.join(str(value).split())
    return [text] if text else []

def _filter_terms(question_ir: dict[str, Any]) -> list[str]:
    route = _question_route(question_ir)
    values = [*_iter_text_values(route.get('event_constraints')), *_iter_text_values(route.get('spatial_constraints')), *_iter_text_values(question_ir.get('event_constraints')), *_iter_text_values(question_ir.get('spatial_constraints')), *_iter_text_values(question_ir.get('hard_constraints'))]
    targets = {value.lower() for value in _entity_count_target_terms(question_ir)}
    return [value for value in _ordered_unique(values) if value.lower() not in targets]

def _entity_count_target_terms(question_ir: dict[str, Any]) -> list[str]:
    route = _question_route(question_ir)
    values = [*_iter_text_values(route.get('primary_target_terms')), *_iter_text_values(question_ir.get('target_entities'))]
    return _ordered_unique(values)

def _count_policy(question: str, question_ir: dict[str, Any]) -> str:
    text = ' '.join([question, ' '.join(_target_terms(question_ir)), ' '.join(_filter_terms(question_ir))]).lower()
    tokens = set(_tokens(text))
    if {'turn', 'turned', 'turning', 'around', 'left', 'right'} & tokens:
        return 'ego_action_count'
    if not _entity_count_target_terms(question_ir):
        return 'event_episode_count'
    if {'times', 'time', 'events', 'event'} & tokens:
        return 'event_episode_count'
    return 'entity_set_count'

def _count_effective_retrieval_window(question: str, question_ir: dict[str, Any], *, count_policy: str, retrieval_window: dict[str, Any] | None) -> dict[str, Any] | None:
    if not _retrieval_window_bounds(retrieval_window):
        return retrieval_window
    if count_policy == 'entity_set_count' and _count_question_has_global_scope(question, question_ir):
        return {'applied': False, 'reason': 'global_count_scope', 'source_window': retrieval_window}
    return retrieval_window

def _count_question_has_global_scope(question: str, question_ir: dict[str, Any]) -> bool:
    tokens = set(_tokens(' '.join([question, *_filter_terms(question_ir)])))
    return bool(tokens & {'all', 'entire', 'overall', 'total', 'whole'})

def _count_dedupe_key(count_policy: str) -> str:
    if count_policy == 'event_episode_count':
        return 'episode_or_event_id'
    if count_policy == 'ego_action_count':
        return 'ego_action_frame_cluster'
    return 'stable_identity_id|entity_id|source_object_key'

def _append_excluded(excluded_items: list[dict[str, Any]] | None, row: dict[str, Any], *, reason: str) -> None:
    if excluded_items is None or len(excluded_items) >= 24:
        return
    excluded_items.append({'reason': reason, 'label': row.get('label'), 'frame': row.get('graph_frame_index'), 'object_index': row.get('object_index'), 'visual_attribute': row.get('visual_attribute')})

def _retrieval_window_bounds(retrieval_window: dict[str, Any] | None) -> tuple[float, float] | None:
    if not isinstance(retrieval_window, dict):
        return None
    applied = retrieval_window.get('applied')
    if applied is False:
        return None
    start = _optional_float(retrieval_window.get('start_ts') if retrieval_window.get('start_ts') is not None else retrieval_window.get('start_time'))
    end = _optional_float(retrieval_window.get('end_ts') if retrieval_window.get('end_ts') is not None else retrieval_window.get('end_time'))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    if start > end:
        start, end = (end, start)
    return (start, end)

def _time_window_metadata(retrieval_window: dict[str, Any] | None, window_bounds: tuple[float, float] | None) -> dict[str, Any]:
    if window_bounds is None:
        return {'applied': False, 'reason': str((retrieval_window or {}).get('reason') or 'missing_or_disabled') if isinstance(retrieval_window, dict) else 'missing_or_disabled'}
    return {'applied': True, 'start_ts': window_bounds[0], 'end_ts': window_bounds[1], 'source': str((retrieval_window or {}).get('source') or 'question_retrieval_window')}

def _row_overlaps_retrieval_window(row: dict[str, Any], entity: Any, window_bounds: tuple[float, float] | None) -> bool:
    if window_bounds is None:
        return True
    entity_overlap = _time_range_overlap_status(getattr(entity, 'time_range', None), window_bounds)
    if entity_overlap is not None:
        return entity_overlap
    row_range = _row_time_range(row)
    if row_range is None:
        return False
    return _range_overlaps_window(row_range[0], row_range[1], window_bounds)

def _node_overlaps_retrieval_window(node: Any, window_bounds: tuple[float, float] | None) -> bool:
    if window_bounds is None:
        return True
    overlap = _time_range_overlap_status(getattr(node, 'time_range', None), window_bounds)
    return bool(overlap)

def _time_range_overlaps_retrieval_window(time_range: Any, window_bounds: tuple[float, float] | None) -> bool:
    if window_bounds is None:
        return True
    overlap = _time_range_overlap_status(time_range, window_bounds)
    return bool(overlap)

def _time_range_overlap_status(time_range: Any, window_bounds: tuple[float, float] | None) -> bool | None:
    if window_bounds is None:
        return True
    if time_range is None:
        return None
    start = _optional_float(getattr(time_range, 'start_ts', None))
    end = _optional_float(getattr(time_range, 'end_ts', None))
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    return _range_overlaps_window(start, end, window_bounds)

def _row_time_range(row: dict[str, Any]) -> tuple[float, float] | None:
    start = _optional_float(row.get('start_ts') if row.get('start_ts') is not None else row.get('start_time'))
    end = _optional_float(row.get('end_ts') if row.get('end_ts') is not None else row.get('end_time'))
    timestamp = _optional_float(row.get('timestamp') if row.get('timestamp') is not None else row.get('timestamp_sec'))
    if timestamp is None:
        timestamp = _optional_float(row.get('frame_timestamp') if row.get('frame_timestamp') is not None else row.get('graph_timestamp'))
    if start is None and end is None and (timestamp is not None):
        start = timestamp
        end = timestamp
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if start is None or end is None:
        return None
    if start > end:
        start, end = (end, start)
    return (start, end)

def _range_overlaps_window(start: float, end: float, window_bounds: tuple[float, float]) -> bool:
    window_start, window_end = window_bounds
    return end >= window_start and start <= window_end

def _time_range_fields(time_range: Any) -> dict[str, Any]:
    if time_range is None:
        return {}
    return {'start_frame': getattr(time_range, 'start_frame', None), 'end_frame': getattr(time_range, 'end_frame', None), 'start_ts': getattr(time_range, 'start_ts', None), 'end_ts': getattr(time_range, 'end_ts', None)}

def _node_semantic_text(node: Any) -> str:
    values = [getattr(node, 'label', ''), getattr(node, 'episode_kind', ''), getattr(node, 'event_type', ''), getattr(node, 'predicate', ''), getattr(node, 'raw_predicate', ''), getattr(node, 'attr_key', ''), getattr(node, 'attr_value', ''), getattr(node, 'previous_state_value', ''), getattr(node, 'current_state_value', ''), getattr(node, 'source_ref_id', ''), getattr(node, 'attributes', {})]
    return _stringify(values)

def _entity_semantic_text(entity: Any) -> str:
    values = [getattr(entity, 'label', ''), getattr(entity, 'stable_identity_display_name', ''), getattr(entity, 'stable_identity_canonical_label', ''), getattr(entity, 'labels_seen', []), getattr(entity, 'states_seen', []), getattr(entity, 'parent_labels_seen', []), getattr(entity, 'attributes', {})]
    return _stringify(values)

def _cluster_count_items_by_frame(items: list[dict[str, Any]], *, max_frame_gap: int) -> list[dict[str, Any]]:
    ordered = sorted(items, key=lambda item: (_optional_int(item.get('frame')) or 10 ** 12, str(item.get('entity_key') or '')))
    clusters: list[list[dict[str, Any]]] = []
    for item in ordered:
        frame = _optional_int(item.get('frame'))
        if frame is None:
            continue
        if clusters and _optional_int(clusters[-1][-1].get('frame')) is not None and (frame - int(clusters[-1][-1]['frame']) <= max_frame_gap):
            clusters[-1].append(item)
        else:
            clusters.append([item])
    results: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        first = cluster[0]
        last = cluster[-1]
        labels = _ordered_unique([str(item.get('label') or '') for item in cluster])
        results.append({'entity_key': f"ego_action_cluster:{index}:{first.get('frame')}", 'event_id': f"ego_action_cluster:{index}:{first.get('frame')}", 'label': labels[0] if labels else 'ego_action', 'frame': first.get('frame'), 'end_frame': last.get('end_frame') or last.get('frame'), 'score': max((float(item.get('score') or 0.0) for item in cluster)), 'count_policy': 'ego_action_count', 'dedupe_key': 'ego_action_frame_cluster', 'cluster_size': len(cluster), 'member_event_ids': [item.get('event_id') for item in cluster[:8]]})
    return results

def _temporal_interval_boundary_pack(graph_answer: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], retrieval_window: dict[str, Any] | None) -> dict[str, Any]:
    start_candidates, end_candidates, span_candidates, state_candidates = _interval_boundary_candidates(graph_answer, graph=graph, question=question, question_ir=question_ir, retrieval_window=retrieval_window)
    interval_mode = _temporal_interval_mode(question, question_ir)
    boundary_pairs = _pair_interval_boundaries(start_candidates, end_candidates)
    state_pairs = _state_confirmation_duration_pairs(start_candidates, state_candidates, graph=graph, interval_mode=interval_mode, retrieval_window=retrieval_window, allow_graph_state_confirmation=not (interval_mode == 'state_occupancy' and bool(boundary_pairs)))
    paired = [*state_pairs, *boundary_pairs]
    if paired:
        if state_pairs and paired[0] in state_pairs:
            interval_policy = 'state_confirmation_pair'
            reason = 'state confirmation candidates were paired with route-specific temporal starts'
        else:
            interval_policy = 'paired_boundary'
            reason = 'start and end boundary candidates were paired in frame order'
    elif start_candidates or end_candidates:
        interval_policy = 'incomplete_boundary'
        reason = 'only one side of the interval boundary was available'
    else:
        interval_policy = 'fallback_scene_context'
        reason = 'no usable non-scene boundary candidates were available'
    if not paired and span_candidates:
        paired = span_candidates[:3]
        interval_policy = 'paired_boundary'
        reason = 'non-scene temporal span candidate supplies both start and end'
    return {'contract': 'navqa_question_route_temporal_interval_boundary_v1', 'interval_policy': interval_policy, 'start_boundary_candidates': start_candidates[:8], 'end_boundary_candidates': end_candidates[:8], 'state_confirmation_candidates': state_candidates[:8], 'paired_duration_candidates': paired[:4], 'boundary_pair_reason': reason, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False}

def _interval_boundary_candidates(graph_answer: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], retrieval_window: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target_tokens = _temporal_interval_target_tokens(question, question_ir)
    interval_mode = _temporal_interval_mode(question, question_ir)
    window_bounds = _retrieval_window_bounds(retrieval_window)
    entity_text_by_id = _entity_text_index(graph)
    start_candidates: list[dict[str, Any]] = []
    end_candidates: list[dict[str, Any]] = []
    span_candidates: list[dict[str, Any]] = []
    state_candidates: list[dict[str, Any]] = []
    retrieval_bundle = graph_answer.get('retrieval_bundle', {})
    if not isinstance(retrieval_bundle, dict):
        retrieval_bundle = {}
    for rank, item in enumerate(retrieval_bundle.get('semantic_event_candidates', []) or [], start=1):
        if not isinstance(item, dict):
            continue
        frame = _optional_int(item.get('boundary_frame'))
        if frame is None:
            continue
        role = _boundary_role_from_text(str(item.get('event_type') or item.get('summary') or ''))
        candidate = {'rank': rank, 'frame': frame, 'source': 'semantic_event_candidate', 'event_type': item.get('event_type'), 'scene_id': item.get('boundary_scene_id'), 'score': _float_or_zero(item.get('score'), default=10.0), 'summary': str(item.get('summary') or '')}
        if role == 'end':
            end_candidates.append(candidate)
        else:
            start_candidates.append(candidate)
    for rank, item in enumerate(retrieval_bundle.get('temporal_span_candidates', []) or [], start=1):
        if not isinstance(item, dict):
            continue
        source_kind = str(item.get('source_kind') or '').strip().lower()
        start_frame = _optional_int(item.get('start_frame'))
        end_frame = _optional_int(item.get('end_frame'))
        if start_frame is None or end_frame is None:
            continue
        summary = str(item.get('summary') or '')
        score = _float_or_zero(item.get('score'), default=0.0)
        state_role = _temporal_state_confirmation_role(' '.join([summary, str(item.get('source') or ''), str(item.get('source_kind') or '')]), interval_mode=interval_mode)
        if state_role:
            state_candidates.append(_boundary_candidate(rank=rank, frame=start_frame, source='temporal_state_confirmation', source_kind=source_kind or 'episode', score=score + 8.0, summary=summary, state_role=state_role))
            continue
        if start_frame != end_frame and source_kind not in {'scene', 'transition_event'}:
            span_candidates.append(_paired_duration_candidate(start_frame=min(start_frame, end_frame), end_frame=max(start_frame, end_frame), source='question_route_existing_temporal_span', rank=rank, score=score, summary=summary, source_kind=source_kind or 'temporal_composition', reason='existing non-scene temporal span candidate'))
            continue
        if source_kind == 'scene':
            continue
        role = _boundary_role_from_text(' '.join([summary, str(item.get('source') or '')]))
        candidate = {'rank': rank, 'frame': start_frame, 'source': 'temporal_span_boundary', 'source_kind': source_kind, 'score': score, 'summary': summary}
        if role == 'end':
            end_candidates.append(candidate)
        else:
            start_candidates.append(candidate)
    for rank, node in enumerate(graph.episode_nodes, start=1):
        if not _node_overlaps_retrieval_window(node, window_bounds):
            continue
        node_text = _node_semantic_text(node)
        text = _node_semantic_text_with_entities(node, entity_text_by_id)
        tokens = set(_tokens(text))
        state_role = _temporal_state_confirmation_role(node_text, interval_mode=interval_mode)
        if target_tokens and (not tokens & target_tokens) and (not state_role):
            continue
        time_range = getattr(node, 'time_range', None)
        start_frame = getattr(time_range, 'start_frame', None)
        end_frame = getattr(time_range, 'end_frame', None)
        if start_frame is None or end_frame is None:
            continue
        if state_role:
            state_candidates.append(_boundary_candidate(rank=rank, frame=int(end_frame if end_frame is not None else start_frame), source='graph_state_confirmation', source_kind='episode', score=float(len(tokens & target_tokens)) + 14.0, summary=text[:240], state_role=state_role))
            continue
        if start_frame == end_frame:
            continue
        span_candidates.append(_paired_duration_candidate(start_frame=min(int(start_frame), int(end_frame)), end_frame=max(int(start_frame), int(end_frame)), source='question_route_graph_episode_span', rank=rank, score=float(len(tokens & target_tokens)), summary=text[:240], source_kind='episode', reason='graph episode matching interval target terms'))
    return (_sort_boundary_candidates(start_candidates), _sort_boundary_candidates(end_candidates), sorted(span_candidates, key=lambda item: (-float(item.get('score') or 0.0), _optional_int(item.get('start_frame')) or 10 ** 12)), _sort_boundary_candidates(state_candidates))

def _temporal_interval_target_tokens(question: str, question_ir: dict[str, Any]) -> set[str]:
    text = ' '.join([question, *_target_terms(question_ir), *_filter_terms(question_ir)])
    return {token for token in _tokens(text) if token not in _TEMPORAL_INTERVAL_STOP_TOKENS and len(token) > 1}

def _temporal_interval_mode(question: str, question_ir: dict[str, Any]) -> str:
    text = ' '.join([question, *_target_terms(question_ir), *_filter_terms(question_ir)]).lower()
    tokens = set(_tokens(text))
    if tokens & {'wait', 'waiting'}:
        return 'waiting'
    if tokens & _TEMPORAL_TRAVEL_TO_TOKENS and 'to' in tokens:
        return 'travel_to'
    target_state = ' '.join(_iter_text_values(question_ir.get('target_state'))).lower()
    if 'inside' in tokens or re.search('\\bin\\s+(?:the\\s+)?[a-z0-9]+', target_state):
        return 'state_occupancy'
    return 'generic_interval'

def _temporal_state_confirmation_role(text: str, *, interval_mode: str) -> str:
    if interval_mode not in {'state_occupancy', 'travel_to'}:
        return ''
    tokens = set(_tokens(text))
    if tokens & _TEMPORAL_STATE_CONFIRMATION_TOKENS:
        return 'arrival_state' if interval_mode == 'travel_to' else 'state_observed'
    return ''

def _state_confirmation_duration_pairs(start_candidates: list[dict[str, Any]], state_candidates: list[dict[str, Any]], *, graph: CanonicalTemporalMemoryGraph, interval_mode: str, retrieval_window: dict[str, Any] | None, allow_graph_state_confirmation: bool=True) -> list[dict[str, Any]]:
    if not allow_graph_state_confirmation:
        state_candidates = [item for item in state_candidates if item.get('source') != 'graph_state_confirmation']
    if not state_candidates:
        return []
    ordered_states = sorted(state_candidates, key=lambda item: (_optional_int(item.get('frame')) or -1, float(item.get('score') or 0.0)), reverse=True)
    state = ordered_states[0]
    state_frame = _optional_int(state.get('frame'))
    if state_frame is None:
        return []
    if interval_mode == 'travel_to':
        start_frame = _estimate_frame_for_window_start(graph, retrieval_window)
        if start_frame is None or start_frame >= state_frame:
            return []
        return [_paired_duration_candidate(start_frame=start_frame, end_frame=state_frame, source='question_route_window_to_state_confirmation', rank=state.get('rank'), score=float(state.get('score') or 0.0) + 24.0, summary=f"retrieval-window start paired with target-arrival state confirmation :: {state.get('summary') or ''}", source_kind='temporal_composition', reason='travel-to interval uses question window start and target arrival state')]
    if interval_mode != 'state_occupancy':
        return []
    preceding_starts = [item for item in start_candidates if _optional_int(item.get('frame')) is not None and _optional_int(item.get('frame')) < state_frame]
    if not preceding_starts:
        return []
    start = max(preceding_starts, key=lambda item: (_optional_int(item.get('frame')) or -1, float(item.get('score') or 0.0)))
    start_frame = _optional_int(start.get('frame'))
    if start_frame is None:
        return []
    frame_gap = state_frame - start_frame
    if frame_gap < _TEMPORAL_INTERVAL_MIN_STATE_PAIR_FRAMES or frame_gap > _TEMPORAL_INTERVAL_MAX_STATE_PAIR_FRAMES:
        return []
    gap_penalty = min(12.0, max(0.0, float(state_frame - start_frame) / 1000.0))
    return [_paired_duration_candidate(start_frame=start_frame, end_frame=state_frame, source='question_route_state_confirmation_pair', rank=state.get('rank'), score=float(start.get('score') or 0.0) + float(state.get('score') or 0.0) + 20.0 - gap_penalty, summary=f"start={start.get('summary') or start.get('source')} | state={state.get('summary') or state.get('source')}", source_kind='temporal_composition', reason='state interval uses nearest prior boundary and later state confirmation')]

def _boundary_candidate(*, rank: Any, frame: int, source: str, source_kind: str, score: float, summary: str, state_role: str='') -> dict[str, Any]:
    candidate = {'rank': rank, 'frame': frame, 'source': source, 'source_kind': source_kind, 'score': round(float(score or 0.0), 4), 'summary': ' '.join(str(summary or '').split())[:500]}
    if state_role:
        candidate['state_role'] = state_role
    return candidate

def _entity_text_index(graph: CanonicalTemporalMemoryGraph) -> dict[str, str]:
    result: dict[str, str] = {}
    for entity in graph.entity_nodes:
        text = _entity_semantic_text(entity)
        keys = [getattr(entity, 'entity_id', ''), getattr(entity, 'node_id', '')]
        for key in keys:
            key_text = str(key or '').strip()
            if key_text:
                result[key_text] = text
                if key_text.startswith('entity:'):
                    result[key_text[len('entity:'):]] = text
    return result

def _node_semantic_text_with_entities(node: Any, entity_text_by_id: dict[str, str]) -> str:
    values = [_node_semantic_text(node)]
    for entity_id in _node_entity_ids(node):
        entity_text = entity_text_by_id.get(entity_id)
        if entity_text:
            values.append(entity_text)
    return _stringify(values)

def _node_entity_ids(node: Any) -> list[str]:
    values: list[str] = []
    for attr in ('entity_id', 'primary_entity_id', 'subject_entity_id', 'object_entity_id'):
        value = str(getattr(node, attr, '') or '').strip()
        if value:
            values.append(value)
    participant_ids = getattr(node, 'participant_entity_ids', []) or []
    for value in participant_ids:
        text = str(value or '').strip()
        if text:
            values.append(text)
    return _ordered_unique(values)

def _estimate_frame_for_window_start(graph: CanonicalTemporalMemoryGraph, retrieval_window: dict[str, Any] | None) -> int | None:
    bounds = _retrieval_window_bounds(retrieval_window)
    if bounds is None:
        return None
    window_start = bounds[0]
    scene_ranges = [getattr(scene, 'time_range', None) for scene in graph.scene_nodes if getattr(scene, 'time_range', None) is not None]
    for time_range in scene_ranges:
        start_ts = _optional_float(getattr(time_range, 'start_ts', None))
        end_ts = _optional_float(getattr(time_range, 'end_ts', None))
        start_frame = _optional_int(getattr(time_range, 'start_frame', None))
        end_frame = _optional_int(getattr(time_range, 'end_frame', None))
        if start_ts is None or end_ts is None or start_frame is None or (end_frame is None) or (end_ts <= start_ts):
            continue
        if not start_ts <= window_start <= end_ts:
            continue
        ratio = (window_start - start_ts) / (end_ts - start_ts)
        return int(round(start_frame + ratio * (end_frame - start_frame)))
    return None

def _pair_interval_boundaries(start_candidates: list[dict[str, Any]], end_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for start in start_candidates:
        start_frame = _optional_int(start.get('frame'))
        if start_frame is None:
            continue
        for end in end_candidates:
            end_frame = _optional_int(end.get('frame'))
            if end_frame is None or end_frame <= start_frame:
                continue
            pairs.append(_paired_duration_candidate(start_frame=start_frame, end_frame=end_frame, source='question_route_paired_boundary', rank=None, score=float(start.get('score') or 0.0) + float(end.get('score') or 0.0), summary=f"start={start.get('summary') or start.get('source')} | end={end.get('summary') or end.get('source')}", source_kind='temporal_composition', reason='paired start/end boundary candidates'))
    return sorted(pairs, key=lambda item: (-float(item.get('score') or 0.0), _optional_int(item.get('start_frame')) or 10 ** 12, _optional_int(item.get('end_frame')) or 10 ** 12))

def _paired_duration_candidate(*, start_frame: int, end_frame: int, source: str, rank: Any, score: float, summary: str, source_kind: str, reason: str) -> dict[str, Any]:
    return {'candidate_type': 'temporal_span', 'source': source, 'rank': rank, 'source_kind': source_kind, 'score': round(float(score or 0.0), 4), 'target_hits': [], 'start_frame': start_frame, 'end_frame': end_frame, 'summary': ' '.join(str(summary or '').split())[:500], 'warning': '', 'boundary_pair_reason': reason}

def _boundary_role_from_text(text: str) -> str:
    tokens = set(_tokens(text))
    if tokens & {'leave', 'left', 'exit', 'exited', 'end', 'ended', 'finish', 'finished', 'disappear'}:
        return 'end'
    return 'start'

def _sort_boundary_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: (-float(item.get('score') or 0.0), _optional_int(item.get('frame')) or 10 ** 12))

def _temporal_interval_boundary_lines(pack: dict[str, Any]) -> list[str]:
    lines = [f"kind=temporal_interval_boundary interval_policy={pack.get('interval_policy')} start_candidates={len(pack.get('start_boundary_candidates') or [])} end_candidates={len(pack.get('end_boundary_candidates') or [])} paired_candidates={len(pack.get('paired_duration_candidates') or [])} reason={pack.get('boundary_pair_reason')}"]
    for item in (pack.get('paired_duration_candidates') or [])[:3]:
        lines.append(f"kind=paired_duration_candidate source={item.get('source')} time=frames {item.get('start_frame')}-{item.get('end_frame')} score={item.get('score')} :: {item.get('summary')}")
    return lines

def _temporal_point_candidate_pack(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], retrieval_window: dict[str, Any] | None) -> dict[str, Any]:
    operation = str(question_ir.get('operation') or '').strip().lower()
    target_tokens = _temporal_point_target_tokens(question, question_ir)
    state_tokens = _temporal_point_state_tokens(question, question_ir)
    candidates = _temporal_point_candidates(visual_payload, graph=graph, target_tokens=target_tokens, state_tokens=state_tokens, retrieval_window=retrieval_window)
    selection_status = 'unsupported_operation'
    selected: dict[str, Any] = {}
    if operation in _TEMPORAL_POINT_SELECTED_OPERATIONS:
        if candidates and float(candidates[0].get('score') or 0.0) >= _TEMPORAL_POINT_MIN_SELECTED_SCORE:
            selection_status = 'selected'
            selected = dict(candidates[0])
        elif candidates:
            selection_status = 'below_confidence_floor'
        else:
            selection_status = 'no_candidate'
    return {'contract': 'navqa_question_route_temporal_point_candidate_v1', 'candidate_unit': 'temporal_point_candidate', 'operation': operation, 'selection_status': selection_status, 'selected_candidate': selected, 'candidates': candidates[:8], 'target_tokens': sorted(target_tokens), 'state_tokens': sorted(state_tokens), 'min_selected_score': _TEMPORAL_POINT_MIN_SELECTED_SCORE, 'window_tolerance_seconds': _TEMPORAL_POINT_WINDOW_TOLERANCE_SECONDS, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False, 'uses_question_route_branches': True, 'uses_question_type_branches': False}

def _temporal_point_candidates(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, target_tokens: set[str], state_tokens: set[str], retrieval_window: dict[str, Any] | None) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    groups = visual_payload.get('groups', []) if isinstance(visual_payload, dict) else []
    for group_rank, group in enumerate(groups or [], start=1):
        if not isinstance(group, dict):
            continue
        query_phrase = str(group.get('query_phrase') or '')
        phrase_score = _temporal_point_phrase_score(query_phrase, target_tokens=target_tokens, state_tokens=state_tokens)
        for result_rank, item in enumerate(group.get('results', []) or [], start=1):
            if not isinstance(item, dict):
                continue
            frame = _optional_int(item.get('graph_frame_index'))
            if frame is None:
                continue
            timestamp = _graph_timestamp_for_frame(graph, frame)
            window_penalty = _temporal_point_window_penalty(timestamp, retrieval_window=retrieval_window)
            item_text = ' '.join((str(item.get(key) or '') for key in ('label', 'visual_attribute', 'object_summary')))
            item_tokens = _temporal_point_token_set(item_text)
            label_tokens = _temporal_point_token_set(item.get('label'))
            target_hits = sorted(item_tokens & target_tokens)
            label_hits = sorted(label_tokens & target_tokens)
            similarity = _float_or_zero(item.get('similarity'))
            object_score = _float_or_zero(item.get('score'))
            score = phrase_score + 80.0 * similarity + 3.0 * object_score + 2.0 * len(target_hits) + 3.0 * len(label_hits) - 0.4 * result_rank - window_penalty
            candidate = {'frame': frame, 'timestamp_sec': timestamp, 'score': round(score, 4), 'query_phrase': query_phrase, 'query_phrase_rank': group_rank, 'result_rank': result_rank, 'label': str(item.get('label') or ''), 'visual_attribute': str(item.get('visual_attribute') or ''), 'similarity': round(similarity, 6), 'object_score': round(object_score, 4), 'phrase_score': round(float(phrase_score), 4), 'window_penalty': round(float(window_penalty), 4), 'target_hits': target_hits, 'label_hits': label_hits, 'source': 'visual_faiss_query_group'}
            raw_candidates.append(candidate)
    ordered = sorted(raw_candidates, key=lambda item: (-float(item.get('score') or 0.0), _optional_int(item.get('frame')) or 10 ** 12, _optional_int(item.get('result_rank')) or 10 ** 12))
    return _dedupe_temporal_point_candidates(ordered)

def _temporal_point_phrase_score(query_phrase: str, *, target_tokens: set[str], state_tokens: set[str]) -> float:
    phrase_tokens = _temporal_point_token_set(query_phrase)
    if not phrase_tokens:
        return 0.0
    score = 4.0 * len(phrase_tokens & state_tokens)
    score += 1.0 * len(phrase_tokens & target_tokens)
    if state_tokens and state_tokens <= phrase_tokens:
        score += 8.0
    return score

def _temporal_point_window_penalty(timestamp: float | None, *, retrieval_window: dict[str, Any] | None) -> float:
    bounds = _retrieval_window_bounds(retrieval_window)
    if timestamp is None or bounds is None:
        return 0.0
    start_ts, end_ts = bounds
    tolerance = _TEMPORAL_POINT_WINDOW_TOLERANCE_SECONDS
    if start_ts - tolerance <= timestamp <= end_ts + tolerance:
        return 0.0
    before_minutes = max(0.0, (start_ts - timestamp) / 60.0)
    after_minutes = max(0.0, (timestamp - end_ts) / 60.0)
    return 10.0 + 2.0 * (before_minutes + after_minutes)

def _temporal_point_target_tokens(question: str, question_ir: dict[str, Any]) -> set[str]:
    question_route = _question_route(question_ir)
    values: list[Any] = [question, question_ir.get('target_state'), question_ir.get('target_entities'), question_ir.get('event_constraints'), question_ir.get('spatial_constraints'), question_ir.get('hard_constraints'), question_route.get('primary_target_terms'), question_route.get('event_constraints')]
    return _temporal_point_token_set(_stringify(values))

def _temporal_point_state_tokens(question: str, question_ir: dict[str, Any]) -> set[str]:
    question_route = _question_route(question_ir)
    values: list[Any] = [question_ir.get('target_state'), question_route.get('primary_target_terms')]
    tokens = _temporal_point_token_set(_stringify(values))
    return tokens or _temporal_point_token_set(question)

def _temporal_point_token_set(value: Any) -> set[str]:
    stop_tokens = _TEMPORAL_INTERVAL_STOP_TOKENS | {'current', 'event', 'photo', 'picture', 'time', 'timestamp', 'timestamps', 'visible', 'when'}
    tokens: set[str] = set()
    for token in _TOKEN_PATTERN.findall(str(value or '').lower()):
        variants = {token}
        if len(token) > 3 and token.endswith('s'):
            variants.add(token[:-1])
        if len(token) > 4 and token.endswith('ing'):
            variants.add(token[:-3])
        if len(token) > 3 and token.endswith('ed'):
            variants.add(token[:-2])
        for variant in variants:
            if variant and variant not in stop_tokens and (len(variant) > 1):
                tokens.add(variant)
    return tokens

def _graph_timestamp_for_frame(graph: CanonicalTemporalMemoryGraph, frame: int) -> float | None:
    for collection_name in ('scene_nodes', 'episode_nodes', 'transition_event_nodes'):
        for node in getattr(graph, collection_name, []) or []:
            time_range = getattr(node, 'time_range', None)
            if time_range is None:
                continue
            timestamp = _timestamp_from_time_range(time_range, frame)
            if timestamp is not None:
                return timestamp
    return None

def _timestamp_from_time_range(time_range: Any, frame: int) -> float | None:
    start_frame = _optional_int(getattr(time_range, 'start_frame', None))
    end_frame = _optional_int(getattr(time_range, 'end_frame', None))
    start_ts = _optional_float(getattr(time_range, 'start_ts', None))
    end_ts = _optional_float(getattr(time_range, 'end_ts', None))
    if start_frame is None or end_frame is None or start_ts is None:
        return None
    if end_ts is None:
        end_ts = start_ts
    lo = min(start_frame, end_frame)
    hi = max(start_frame, end_frame)
    if not lo <= frame <= hi:
        return None
    if end_frame == start_frame or end_ts == start_ts:
        return start_ts
    ratio = (frame - start_frame) / (end_frame - start_frame)
    return float(start_ts + ratio * (end_ts - start_ts))

def _dedupe_temporal_point_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in items:
        frame = _optional_int(item.get('frame'))
        if frame is None:
            continue
        if any((abs(frame - (_optional_int(existing.get('frame')) or frame)) <= _TEMPORAL_POINT_NEAR_FRAME_GAP for existing in result)):
            continue
        result.append(item)
        if len(result) >= 8:
            break
    return result

def _temporal_point_candidate_lines(pack: dict[str, Any]) -> list[str]:
    lines = [f"kind=temporal_point_candidate_pack operation={pack.get('operation')} status={pack.get('selection_status')} candidate_count={len(pack.get('candidates') or [])} min_score={pack.get('min_selected_score')}"]
    selected = pack.get('selected_candidate')
    if isinstance(selected, dict) and selected:
        lines.append(_temporal_point_candidate_line(selected, selected=True))
    selected_key = (selected.get('frame'), selected.get('query_phrase'), selected.get('score')) if isinstance(selected, dict) else None
    for item in (pack.get('candidates') or [])[:3]:
        item_key = (item.get('frame'), item.get('query_phrase'), item.get('score'))
        if selected_key is not None and item_key == selected_key:
            continue
        lines.append(_temporal_point_candidate_line(item, selected=False))
    return lines

def _temporal_point_candidate_line(item: dict[str, Any], *, selected: bool) -> str:
    status = 'selected' if selected else 'candidate'
    return f"kind=temporal_point_{status} frame={item.get('frame')} score={item.get('score')} query={item.get('query_phrase')!r} label={item.get('label')!r} target_hits={','.join(item.get('target_hits') or [])} window_penalty={item.get('window_penalty')}"

def _temporal_point_answer_candidate_line(item: dict[str, Any]) -> str:
    return f"kind=answer_candidate source=temporal_point_candidate answer_frame={item.get('frame')} score={item.get('score')} support={item.get('query_phrase')!r}"

def _dedupe_temporal_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, str]] = set()
    for item in items:
        key = (item.get('start_frame'), item.get('end_frame'), str(item.get('source') or ''))
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

def _ordered_unique(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = ' '.join(str(value or '').split())
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

def _clean_lines(values: Any) -> list[str]:
    return [str(value).strip() for value in values or [] if str(value).strip()]

def _tokens(value: str) -> list[str]:
    tokens = _TOKEN_PATTERN.findall(str(value or '').lower())
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if len(token) > 3 and token.endswith('s'):
            expanded.append(token[:-1])
    return expanded

def _stringify(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return ' '.join((_stringify(item) for item in value.values()))
    if isinstance(value, (list, tuple, set)):
        return ' '.join((_stringify(item) for item in value))
    return str(value)

def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == '':
            return None
        return int(value)
    except (TypeError, ValueError):
        return None

def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == '':
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def _float_or_zero(value: Any, *, default: float=0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
