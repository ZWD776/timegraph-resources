"""OpenEQA-owned canonical retrieval fork.

This file was forked from the NAVQA-compatible canonical retriever so OpenEQA
graph-only work can evolve without changing the frozen NAVQA runtime. Keep the
baseline behavior covered by differential tests before adding a new profile.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
from typing import Any
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.core.graph_retrieval.canonical_retrieval_types import CanonicalHybridRetrievalOutput, CanonicalProjection as _CanonicalProjection, CanonicalRetrievalMode, CanonicalSourceKind, GraphFragment as _GraphFragment, GraphPath as _GraphPath, GraphRecordView as _GraphRecordView, QueryPlan as _QueryPlan, QueryIntent as _QueryIntent, RelationSignal as _RelationSignal, ScoredRecord as _ScoredRecord
from vlkg.datasets.openeqa.runtime.canonical_record_vector_recall import DEFAULT_VECTOR_MODEL, VECTOR_GRAPH_HYBRID_PROFILE as CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE, VECTOR_PROFILE as CANONICAL_VECTOR_HYBRID_GRAPH_RETRIEVAL_PROFILE, VECTOR_STRUCTURED_PROFILE as CANONICAL_VECTOR_STRUCTURED_GRAPH_RETRIEVAL_PROFILE, VECTOR_UNION_PROFILE as CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE, VECTOR_EVIDENCE_PROFILE as CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE, CanonicalRecordVectorIndex, VectorRecallHit, canonical_vector_query_text
from vlkg.datasets.openeqa.retrieval.semantic_question_ir import semantic_question_ir_projection
from vlkg.core.config.domain_lexicon import DEFAULT_DOMAIN_PROFILE, DomainLexicon, get_domain_lexicon
from vlkg.core.memory.types import MemoryQuery
from vlkg.core.models.temporal_graph import TemporalGraphDocument
from vlkg.core.graph_retrieval.question_compiler import compile_question
TOKEN_PATTERN = re.compile('[a-z0-9_:-]+')
COMPOSITE_TOKEN_SPLIT_PATTERN = re.compile('[_:-]+')
SEED_KIND_BOOST: dict[CanonicalRetrievalMode, dict[CanonicalSourceKind, float]] = {'temporal': {'scene': 1.1, 'entity': 0.5, 'entity_attribute': 1.4, 'stable_identity': 0.8, 'episode': 1.8, 'transition_event': 1.7, 'state_value': 0.6, 'interaction_type': 0.6, 'scene_base': 0.9, 'scene_delta': 1.2, 'temporal_composition': 2.0}, 'entity': {'scene': 0.6, 'entity': 1.7, 'entity_attribute': 0.9, 'stable_identity': 1.9, 'episode': 1.3, 'transition_event': 1.0, 'state_value': 0.5, 'interaction_type': 0.5, 'scene_base': 0.7, 'scene_delta': 1.0, 'temporal_composition': 0.4}}
RESULT_KIND_BOOST: dict[CanonicalRetrievalMode, dict[CanonicalSourceKind, float]] = {'temporal': {'scene': 0.8, 'entity': 0.3, 'entity_attribute': 0.9, 'stable_identity': 0.5, 'episode': 1.1, 'transition_event': 1.0, 'state_value': 0.2, 'interaction_type': 0.2, 'scene_base': 0.4, 'scene_delta': 0.7, 'temporal_composition': 1.4}, 'entity': {'scene': 0.3, 'entity': 1.0, 'entity_attribute': 0.6, 'stable_identity': 1.1, 'episode': 0.9, 'transition_event': 0.5, 'state_value': 0.2, 'interaction_type': 0.2, 'scene_base': 0.4, 'scene_delta': 0.6, 'temporal_composition': 0.2}}
EDGE_TYPE_WEIGHT: dict[CanonicalRetrievalMode, dict[str, float]] = {'temporal': {'NEXT_EPISODE': 2.3, 'PREVIOUS_EPISODE': 2.2, 'TRIGGERED_BY': 2.1, 'INTERACTS_WITH': 1.9, 'HAS_MEMBER_ENTITY': 1.6, 'BELONGS_TO_SEGMENT': 1.7, 'HAS_EPISODE': 1.6, 'IN_SCENE': 1.5, 'HAS_STATE': 1.3, 'HAS_INTERACTION_TYPE': 1.2, 'STARTS_AT': 1.2, 'ENDS_AT': 1.2, 'OVERLAPS_WITH': 1.5, 'TEMPORALLY_NEAR': 1.7, 'SCENE_CONTAINS': 1.3, 'SCENE_UPDATES': 1.6, 'DESCRIBES_ENTITY': 1.5}, 'entity': {'INTERACTS_WITH': 2.2, 'HAS_MEMBER_ENTITY': 2.1, 'HAS_EPISODE': 1.9, 'HAS_STATE': 1.5, 'HAS_INTERACTION_TYPE': 1.3, 'BELONGS_TO_SEGMENT': 1.2, 'IN_SCENE': 1.1, 'NEXT_EPISODE': 1.0, 'PREVIOUS_EPISODE': 1.0, 'TRIGGERED_BY': 1.4, 'STARTS_AT': 0.9, 'ENDS_AT': 0.9, 'OVERLAPS_WITH': 1.0, 'TEMPORALLY_NEAR': 1.0, 'SCENE_CONTAINS': 1.1, 'SCENE_UPDATES': 1.3, 'DESCRIBES_ENTITY': 1.8}}
PRIORITY_BY_KIND: dict[CanonicalSourceKind, int] = {'episode': 0, 'transition_event': 1, 'temporal_composition': 2, 'scene_delta': 3, 'stable_identity': 4, 'entity_attribute': 5, 'entity': 6, 'scene': 7, 'scene_base': 8, 'state_value': 9, 'interaction_type': 10}
RELATION_OBSERVED_BURST_KIND = 'RELATION_OBSERVED_BURST'
DERIVED_TEMPORAL_EDGE_TYPES = {'OVERLAPS_WITH', 'TEMPORALLY_NEAR'}
DERIVED_PREVIOUS_EPISODE_EDGE_TYPE = 'PREVIOUS_EPISODE'
DERIVED_STRUCTURAL_EDGE_TYPES = {'HAS_EPISODE', 'BELONGS_TO_SEGMENT', 'IN_SCENE'}
TEMPORAL_SEMANTIC_EPISODE_KINDS = {'STATE', 'RELATION'}
TEMPORAL_NEAR_MAX_FRAME_GAP = 5
TEMPORAL_NEAR_MAX_TS_GAP = 1.0
BASE_DELTA_SOURCE_KINDS = {'scene_base', 'scene_delta'}
BASE_DELTA_QUERY_HINTS = {'anchor', 'background', 'base', 'change', 'changed', 'changes', 'changing', 'delta', 'dynamic', 'static', 'update', 'updates', 'updating'}
BASE_DELTA_QUERY_PHRASE_HINTS = ('base delta', 'base/delta', 'scene base', 'scene delta', 'stable context', 'static context', 'dynamic context', 'dynamic change', 'delta observation')
TEMPORAL_COMPOSITION_QUERY_HINTS = {'after', 'before', 'between', 'change', 'changed', 'changes', 'during', 'duration', 'enter', 'entered', 'entering', 'exit', 'exited', 'exiting', 'first', 'from', 'inside', 'into', 'last', 'leave', 'leaving', 'left', 'long', 'outside', 'sequence', 'since', 'start', 'started', 'stop', 'stopped', 'through', 'timeline', 'until', 'when', 'while'}
TEMPORAL_ORDER_QUERY_HINTS = {'after', 'before', 'between', 'during', 'first', 'from', 'last', 'sequence', 'since', 'then', 'through', 'timeline', 'until', 'when', 'while'}
DURATION_QUERY_HINTS = {'duration', 'long', 'spent', 'took'}
POSITION_QUERY_HINTS = {'closest', 'near', 'nearest', 'position', 'where'}
EXISTENCE_QUERY_HINTS = {'any', 'ever', 'saw', 'see', 'seen'}
ATTRIBUTE_QUERY_HINTS = {'appearance', 'colored', 'color', 'colour', 'colors', 'look', 'looks', 'wearing', 'wore'}
ATTRIBUTE_KEY_ALIASES: dict[str, tuple[str, ...]] = {'appearance': ('appearance', 'look', 'looks', 'wearing', 'wore'), 'color': ('color', 'colour', 'colored', 'colors')}
COLOR_TOKENS = {'beige', 'black', 'blue', 'brown', 'gray', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow'}
STRUCTURED_ATTRIBUTE_DIMENSION_TOKENS: dict[str, set[str]] = {'color': COLOR_TOKENS | {'cream', 'dark', 'gold', 'golden', 'light', 'multicolor', 'multicolored', 'offwhite'}, 'material': {'brick', 'ceramic', 'cloth', 'concrete', 'fabric', 'glass', 'granite', 'iron', 'leather', 'marble', 'metal', 'metallic', 'paper', 'plastic', 'stone', 'steel', 'textile', 'tile', 'tiled', 'velvet', 'wood', 'wooden', 'wool', 'woven'}, 'shape': {'circle', 'circular', 'cylinder', 'cylindrical', 'oval', 'rectangle', 'rectangular', 'round', 'rounded', 'sphere', 'spherical', 'square', 'triangle', 'triangular'}, 'side': {'left', 'right'}, 'state': {'broken', 'clean', 'closed', 'damaged', 'dirty', 'empty', 'folded', 'full', 'intact', 'lit', 'off', 'on', 'open', 'opened', 'shut', 'unfolded', 'unlit'}}
STRUCTURED_ATTRIBUTE_DIMENSION_ALIASES = {'colour': 'color', 'look': 'appearance', 'style': 'appearance'}
CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES = {CANONICAL_VECTOR_HYBRID_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE, CANONICAL_VECTOR_STRUCTURED_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE}
CANONICAL_VECTOR_STRUCTURED_RETRIEVAL_PROFILES = {CANONICAL_VECTOR_STRUCTURED_GRAPH_RETRIEVAL_PROFILE}
CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES = {CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE}
UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILE = 'navqa_temporal'
UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES = {UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_HYBRID_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE, CANONICAL_VECTOR_STRUCTURED_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE, CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE}
SPATIAL_RELATION_QUERY_TOKENS = {'above', 'against', 'behind', 'below', 'beside', 'between', 'front', 'in', 'inside', 'left', 'near', 'next', 'on', 'opposite', 'over', 'right', 'under'}
SPATIAL_BACKGROUND_ANCHOR_TOKENS = {'area', 'bathroom', 'bedroom', 'ceiling', 'floor', 'garage', 'hallway', 'home', 'house', 'kitchen', 'living', 'room', 'side', 'wall'}
EGO_DIRECTION_QUERY_HINTS = {'direction', 'driving', 'left', 'right', 'side', 'turn', 'turned', 'walking'}
TEMPORAL_COMPOSITION_QUERY_PHRASE_HINTS = ('how long', 'how much time', 'what happened between', 'what changed', 'state transition', 'scene transition', 'enter and leave', 'leave and enter', 'start and end')
TEMPORAL_AXIS_QUERY_FAMILIES = {'temporal', 'duration'}
TEMPORAL_AXIS_SOURCE_KIND_PRIORITY: dict[CanonicalSourceKind, int] = {'temporal_composition': 0, 'scene': 1, 'transition_event': 2, 'scene_delta': 3, 'episode': 4, 'scene_base': 5, 'entity': 6, 'entity_attribute': 7, 'stable_identity': 8, 'state_value': 9, 'interaction_type': 10}
TEMPORAL_AXIS_DURATION_SOURCE_KIND_PRIORITY: dict[CanonicalSourceKind, int] = {'episode': 0, 'transition_event': 1, 'scene_delta': 2, 'entity': 3, 'stable_identity': 4, 'entity_attribute': 5, 'temporal_composition': 6, 'scene': 7, 'scene_base': 8, 'state_value': 9, 'interaction_type': 10}
SCENE_CONTEXT_CUES: dict[str, set[str]] = {'indoor': set(), 'outdoor': set(), 'threshold': set()}
SCENE_CONTEXT_ALIASES: dict[str, tuple[str, ...]] = {'indoor': (), 'outdoor': (), 'threshold': ()}
APPEARANCE_QUERY_HINTS = ('appear', 'appears', 'appeared', 'appearance', 'show up', 'shows up', 'shown up', 'first seen', 'first appear', '首次出现', '第一次出现', '出现')
DISAPPEARANCE_QUERY_HINTS = ('disappear', 'disappears', 'disappeared', 'disappearance', 'vanish', 'vanishes', 'vanished', 'last seen', '消失', '最后出现', '最后一次出现', '不见')
NON_SEMANTIC_QUERY_TOKENS = {'a', 'an', 'and', 'appear', 'appeared', 'appears', 'appearance', 'are', 'at', 'binary', 'category', 'describe', 'did', 'disappear', 'disappeared', 'disappears', 'disappearance', 'does', 'entity', 'first', 'happen', 'happens', 'how', 'i', 'have', 'has', 'had', 'been', 'in', 'is', 'last', 'me', 'of', 'on', 'seen', 'show', 'shown', 'tell', 'the', 'this', 'text', 'time', 'up', 'was', 'we', 'were', 'what', 'when', 'where', 'who', 'long', 'duration', 'reasoning', 'spatial', 'temporal', 'between', 'before', 'after', 'during', 'while', 'until', 'with', 'from', 'for', 'to', 'into', 'you'}
ATTRIBUTE_TARGET_STOP_TOKENS = NON_SEMANTIC_QUERY_TOKENS | ATTRIBUTE_QUERY_HINTS | COLOR_TOKENS | {'around', 'binary', 'by', 'category', 'close', 'near', 'nearby', 'next', 'position', 'question', 'text', 'time', 'with'}
ATTRIBUTE_CONTEXT_CORE_TOKENS: set[str] = set()
ATTRIBUTE_CONTEXT_TARGET_TOKENS: set[str] = set()
ATTRIBUTE_CARRIER_TARGET_TOKENS: set[str] = set()
PRIMARY_INTENT_QUERY_FAMILIES = {'attribute', 'position', 'existence'}
CONSTRAINT_CLAUSE_PATTERN = re.compile('\\b(?:near|nearby|around|beside|next\\s+to|close\\s+to|in\\s+front\\s+of|behind|after|before|during|while|since|until|between)\\b[^,?.;]*')
REFERENCE_EVENT_CLAUSE_PATTERN = re.compile('$^')
LOW_VALUE_PREDICATES = {'UNMAPPED_RELATION', 'NEAR', 'NEXT_TO'}
TEMPORAL_COMPOSITION_MIN_BOUNDARY_CHANGE = 0.35
_PROJECTION_CACHE: dict[str, _CanonicalProjection] = {}
PROJECTION_CACHE_SCHEMA_VERSION = 6

class CanonicalHybridRetrievalSession:
    """Reusable canonical retrieval state for batch QA."""

    def __init__(self, graph: CanonicalTemporalMemoryGraph, *, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None, domain_profile: str | None=None, vector_recall_cache_path: str | Path | None=None, vector_recall_model_name: str=DEFAULT_VECTOR_MODEL, vector_recall_index: CanonicalRecordVectorIndex | None=None) -> None:
        self.graph = graph
        self.temporal_document = temporal_document
        self.include_base_delta_projection = include_base_delta_projection
        self.projection_cache_path = projection_cache_path
        self.lexicon = get_domain_lexicon(domain_profile)
        self.vector_recall_cache_path = vector_recall_cache_path
        self.vector_recall_model_name = vector_recall_model_name
        self._vector_recall_index = vector_recall_index
        self.projection = _get_projection(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, projection_cache_path=projection_cache_path)

    def retrieve(self, query: MemoryQuery, *, mode: CanonicalRetrievalMode, limit: int) -> CanonicalHybridRetrievalOutput:
        graph_retrieval_profile = _graph_retrieval_profile_from_query(query)
        if graph_retrieval_profile in CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES and self._vector_recall_index is None:
            self._vector_recall_index = CanonicalRecordVectorIndex.build_or_load(self.projection, cache_path=self.vector_recall_cache_path, model_name=self.vector_recall_model_name)
        return _run_with_projection(self.projection, query, mode=mode, limit=limit, include_base_delta_projection=self.include_base_delta_projection, has_temporal_document=self.temporal_document is not None, lexicon=self.lexicon, vector_recall_index=self._vector_recall_index)

def run_canonical_graph_hybrid_retrieval(graph: CanonicalTemporalMemoryGraph, query: MemoryQuery, *, mode: CanonicalRetrievalMode, limit: int, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None, domain_profile: str | None=None, vector_recall_cache_path: str | Path | None=None, vector_recall_model_name: str=DEFAULT_VECTOR_MODEL, vector_recall_index: CanonicalRecordVectorIndex | None=None) -> CanonicalHybridRetrievalOutput:
    session = CanonicalHybridRetrievalSession(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, projection_cache_path=projection_cache_path, domain_profile=domain_profile, vector_recall_cache_path=vector_recall_cache_path, vector_recall_model_name=vector_recall_model_name, vector_recall_index=vector_recall_index)
    return session.retrieve(query, mode=mode, limit=limit)

def _run_with_projection(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, limit: int, include_base_delta_projection: bool=False, has_temporal_document: bool=False, lexicon: DomainLexicon | None=None, vector_recall_index: CanonicalRecordVectorIndex | None=None) -> CanonicalHybridRetrievalOutput:
    lexicon = lexicon or get_domain_lexicon(DEFAULT_DOMAIN_PROFILE)
    graph_retrieval_profile = _graph_retrieval_profile_from_query(query)
    query_text = (query.query_text or '').strip().lower()
    query_tokens = _tokenize(query_text)
    query_intent = _infer_query_intent(query_text, query_tokens, lexicon=lexicon, answer_type=query.answer_type)
    query_intent = _apply_question_ir_to_intent(query_intent, _question_ir_from_query(query), query_text=query_text, lexicon=lexicon)
    query_plan = _plan_query(projection, query, mode=mode, limit=limit, query_text=query_text, query_tokens=query_tokens, query_intent=query_intent)
    vector_hits: list[VectorRecallHit] = []
    vector_similarity_by_record_id: dict[str, float] = {}
    vector_rank_by_record_id: dict[str, int] = {}
    vector_added_candidate_ids: set[str] = set()
    vector_empty_result_fallback = False
    if graph_retrieval_profile in CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES and vector_recall_index is not None:
        vector_query_text = canonical_vector_query_text(query)
        vector_hits = vector_recall_index.search(vector_query_text)
        if not vector_hits:
            vector_hits = vector_recall_index.search(vector_query_text, min_similarity=-1.0)
            vector_empty_result_fallback = bool(vector_hits)
        vector_similarity_by_record_id = {hit.record_id: hit.similarity for hit in vector_hits if hit.record_id in projection.records}
        vector_rank_by_record_id = {hit.record_id: hit.rank for hit in vector_hits if hit.record_id in projection.records}
        lexical_candidate_ids = set(query_plan.candidate_record_ids) if query_plan.candidate_record_ids is not None else set()
        vector_added_candidate_ids = set(vector_similarity_by_record_id) - lexical_candidate_ids if query_plan.candidate_record_ids is not None else set(vector_similarity_by_record_id)
        if vector_similarity_by_record_id:
            if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES and query_plan.candidate_record_ids is not None:
                query_plan = replace(query_plan, candidate_record_ids=frozenset(set(query_plan.candidate_record_ids) | set(vector_similarity_by_record_id)))
            vector_strategy = 'canonical_vector_union' if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else 'canonical_vector_supplement'
            query_plan = replace(query_plan, candidate_strategy=f'{query_plan.candidate_strategy}+{vector_strategy}')
    scored_records = _score_records(projection, query, mode=mode, query_text=query_text, query_tokens=query_tokens, query_intent=query_intent, candidate_record_ids=query_plan.candidate_record_ids, lexicon=lexicon, vector_similarity_by_record_id=vector_similarity_by_record_id if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None, vector_rank_by_record_id=vector_rank_by_record_id if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None, vector_score_record_ids=set(vector_similarity_by_record_id) if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None)
    if not scored_records:
        fallback_plan = query_plan
        if query_plan.candidate_record_ids is not None:
            fallback_plan = _QueryPlan(global_scope_score=query_plan.global_scope_score, retrieval_budget=query_plan.retrieval_budget, seed_limit=query_plan.seed_limit, max_hops=query_plan.max_hops, candidate_record_ids=None, candidate_strategy=f'{query_plan.candidate_strategy}+global_fallback', global_fallback=True, scene_seed_count=query_plan.scene_seed_count, query_family=query_plan.query_family, core_terms=query_plan.core_terms, constraint_tokens=query_plan.constraint_tokens, hard_constraint_token_groups=query_plan.hard_constraint_token_groups, compiler_answer_type=query_plan.compiler_answer_type, compiler_retrieval_strategy=query_plan.compiler_retrieval_strategy, compiler_current_window_required=query_plan.compiler_current_window_required, compiler_target_entities=query_plan.compiler_target_entities, compiler_attributes=query_plan.compiler_attributes, compiler_events=query_plan.compiler_events, compiler_temporal_constraints=query_plan.compiler_temporal_constraints, compiler_spatial_constraints=query_plan.compiler_spatial_constraints, compiler_ranking_policy=query_plan.compiler_ranking_policy, compiler_risk_notes=query_plan.compiler_risk_notes, domain_profile=query_plan.domain_profile)
            scored_records = _score_records(projection, query, mode=mode, query_text=query_text, query_tokens=query_tokens, query_intent=query_intent, candidate_record_ids=None, lexicon=lexicon, vector_similarity_by_record_id=vector_similarity_by_record_id if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None, vector_rank_by_record_id=vector_rank_by_record_id if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None, vector_score_record_ids=set(vector_similarity_by_record_id) if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES else None)
        query_plan = fallback_plan
    vector_scored_records: list[_ScoredRecord] = []
    vector_score_candidate_ids = set(vector_added_candidate_ids)
    if not scored_records and vector_similarity_by_record_id:
        vector_score_candidate_ids = set(vector_similarity_by_record_id)
    if vector_score_candidate_ids and graph_retrieval_profile != CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE:
        vector_scored_records = _score_records(projection, query, mode=mode, query_text=query_text, query_tokens=query_tokens, query_intent=query_intent, candidate_record_ids=frozenset(vector_score_candidate_ids), lexicon=lexicon, vector_similarity_by_record_id=vector_similarity_by_record_id, vector_rank_by_record_id=vector_rank_by_record_id, vector_score_record_ids=vector_score_candidate_ids)
    used_vector_fallback = False
    if not scored_records and vector_scored_records:
        scored_records = vector_scored_records
        used_vector_fallback = True
        query_plan = replace(query_plan, candidate_record_ids=frozenset(vector_score_candidate_ids), candidate_strategy=f'{query_plan.candidate_strategy}+vector_fallback')
    if not scored_records:
        raw_payload: dict[str, Any] = {'mode': mode, 'retrieval_mode': 'canonical_graph_hybrid', 'query_plan': _serialize_query_plan(query_plan), 'question_route': _question_route_from_query(query) or {}, 'seed_candidates': [], 'fragments': [], 'items': []}
        if graph_retrieval_profile in CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES:
            raw_payload['vector_recall'] = _serialize_vector_recall(vector_recall_index, vector_hits, query=canonical_vector_query_text(query), added_candidate_ids=vector_added_candidate_ids)
            raw_payload['vector_recall']['empty_result_fallback'] = vector_empty_result_fallback
        if include_base_delta_projection:
            raw_payload['include_base_delta_projection'] = has_temporal_document
        return CanonicalHybridRetrievalOutput(items=[], raw_payload=raw_payload)
    seed_limit = min(query_plan.seed_limit, len(scored_records))
    seed_records = scored_records[:seed_limit]
    score_lookup = {item.record.source_record_id: item for item in scored_records}
    seed_records = _promote_temporal_axis_seed_records(seed_records, scored_records=scored_records, query_intent=query_intent, seed_limit=seed_limit)
    fragments = _build_fragments(projection, score_lookup=score_lookup, seed_records=seed_records, query=query, mode=mode, limit=limit, max_hops=query_plan.max_hops, lexicon=lexicon)
    items = _rank_evidence_items(projection, score_lookup=score_lookup, fragments=fragments, limit=limit, mode=mode, query_intent=query_intent, query=query)
    primary_item_ids = [str(item.get('source_record_id') or '') for item in items if str(item.get('source_record_id') or '')]
    supplemental_items = _supplemental_context_items(projection, score_lookup=score_lookup, primary_items=items, primary_item_ids=primary_item_ids, query_intent=query_intent, query=query, mode=mode, limit=limit)
    if supplemental_items:
        items = [*items, *supplemental_items]
    vector_supplemental_items: list[dict[str, Any]] = []
    vector_fragments: list[_GraphFragment] = []
    if vector_scored_records and (not used_vector_fallback) and (graph_retrieval_profile != CANONICAL_VECTOR_UNION_GRAPH_RETRIEVAL_PROFILE):
        vector_supplement_limit = 3 if graph_retrieval_profile == CANONICAL_VECTOR_GRAPH_HYBRID_RETRIEVAL_PROFILE else 4 if graph_retrieval_profile == CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE else 1
        vector_supplemental_items, vector_fragments = _canonical_vector_supplemental_items(projection, query=query, mode=mode, limit=limit, max_hops=query_plan.max_hops, query_intent=query_intent, lexicon=lexicon, vector_scored_records=vector_scored_records, existing_items=items, max_items=vector_supplement_limit)
        if vector_supplemental_items:
            insertion_index = min(2, len(items)) if graph_retrieval_profile == CANONICAL_VECTOR_EVIDENCE_GRAPH_RETRIEVAL_PROFILE else min(5, len(items))
            items = [*items[:insertion_index], *vector_supplemental_items, *items[insertion_index:]]
            primary_item_ids.extend((str(item.get('source_record_id') or '') for item in vector_supplemental_items if str(item.get('source_record_id') or '')))
    items = _apply_graph_retrieval_profile_to_evidence_items(items, query=query)
    raw_payload = {'mode': mode, 'retrieval_mode': 'canonical_graph_hybrid', 'query_plan': _serialize_query_plan(query_plan), 'question_route': _question_route_from_query(query) or {}, 'seed_candidates': [_serialize_seed(item) for item in seed_records], 'fragments': [_serialize_fragment(fragment) for fragment in fragments[:max(limit * 3, 10)]], 'primary_item_ids': primary_item_ids, 'supplemental_item_ids': [str(item.get('source_record_id') or '') for item in supplemental_items if str(item.get('source_record_id') or '')], 'items': items}
    if graph_retrieval_profile in CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES:
        raw_payload['vector_supplemental_item_ids'] = [str(item.get('source_record_id') or '') for item in vector_supplemental_items if str(item.get('source_record_id') or '')]
        raw_payload['vector_fragments'] = [_serialize_fragment(fragment) for fragment in vector_fragments]
        raw_payload['vector_recall'] = _serialize_vector_recall(vector_recall_index, vector_hits, query=canonical_vector_query_text(query), added_candidate_ids=vector_added_candidate_ids)
        raw_payload['vector_recall']['empty_result_fallback'] = vector_empty_result_fallback
        if graph_retrieval_profile in CANONICAL_VECTOR_UNION_LIKE_RETRIEVAL_PROFILES:
            raw_payload['vector_union_candidate_ids'] = sorted(vector_similarity_by_record_id)
    if include_base_delta_projection:
        raw_payload['include_base_delta_projection'] = has_temporal_document
    return CanonicalHybridRetrievalOutput(items=items, raw_payload=raw_payload)

def _build_projection(graph: CanonicalTemporalMemoryGraph, *, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False) -> _CanonicalProjection:
    projection = _CanonicalProjection()
    scene_ids_by_entity: dict[str, set[str]] = defaultdict(set)
    entity_ids_by_scene: dict[str, set[str]] = defaultdict(set)
    entity_ids_by_state_value: dict[tuple[str, str], set[str]] = defaultdict(set)
    entity_ids_by_predicate: dict[str, set[str]] = defaultdict(set)
    entity_node_id_by_entity_id = {node.entity_id: node.node_id for node in graph.entity_nodes}
    scene_node_id_by_scene_id = {node.scene_id: node.node_id for node in graph.scene_nodes}
    interaction_node_id_by_predicate = {node.predicate: node.node_id for node in graph.interaction_type_nodes}
    entity_label_by_entity_id = {node.entity_id: node.label for node in graph.entity_nodes}
    for episode in graph.episode_nodes:
        if episode.scene_id:
            for entity_id in episode.participant_entity_ids:
                if entity_id:
                    scene_ids_by_entity[entity_id].add(episode.scene_id)
                    entity_ids_by_scene[episode.scene_id].add(entity_id)
        if episode.attr_key and episode.attr_value:
            entity_ids_by_state_value[episode.attr_key, episode.attr_value].update((entity_id for entity_id in episode.participant_entity_ids if entity_id))
        if episode.predicate:
            entity_ids_by_predicate[episode.predicate].update((entity_id for entity_id in episode.participant_entity_ids if entity_id))
    for node in graph.scene_nodes:
        scene_entity_ids = set(entity_ids_by_scene.get(node.scene_id, set()))
        scene_entity_ids.update(node.representative_static_entity_ids)
        metadata = {'node_id': node.node_id, 'scene_id': node.scene_id, 'scene_index': node.scene_index, 'scene_count': node.scene_count, 'label': node.label, 'scene_kind_hint': node.scene_kind_hint, 'representative_static_labels': list(node.representative_static_labels), 'representative_dynamic_labels': list(node.representative_dynamic_labels), 'entry_static_labels': list(node.entry_static_labels), 'exit_static_labels': list(node.exit_static_labels), 'semantic_alias_terms': list(node.semantic_alias_terms), 'salient_labels': list(node.salient_labels), 'boundary_from_previous_reason': node.boundary_from_previous_reason, 'boundary_to_next_reason': node.boundary_to_next_reason, 'boundary_from_previous_entity_similarity': node.boundary_from_previous_entity_similarity, 'boundary_to_next_entity_similarity': node.boundary_to_next_entity_similarity, 'boundary_from_previous_label_similarity': node.boundary_from_previous_label_similarity, 'boundary_to_next_label_similarity': node.boundary_to_next_label_similarity, 'time_range': node.time_range.model_dump(mode='json')}
        projection.add_record(_make_record(source_kind='scene', source_record_id=node.node_id, summary=f'Scene {node.scene_id} ({node.label}) spans {_format_time_range(node.time_range)}.', metadata=metadata, fields=[node.node_id, node.scene_id, node.label, node.scene_kind_hint or '', *node.representative_static_labels, *node.representative_dynamic_labels, *node.entry_static_labels, *node.exit_static_labels, *[str(item) for item in node.semantic_alias_terms], *[str(item) for item in node.salient_labels]], video_id=node.video_id, scene_ids=(node.scene_id,), entity_ids=tuple(sorted(scene_entity_ids)), start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
    for node in graph.entity_nodes:
        metadata = {'node_id': node.node_id, 'entity_id': node.entity_id, 'label': node.label, 'node_role': node.node_role, 'is_static': node.is_static, 'is_noise': node.is_noise, 'is_fragment': node.is_fragment, 'stable_identity_id': node.stable_identity_id, 'stable_identity_display_name': node.stable_identity_display_name, 'stable_identity_canonical_label': node.stable_identity_canonical_label, 'canonical_label': node.attributes.get('canonical_label'), 'semantic_group': node.attributes.get('semantic_group'), 'semantic_alias_terms': list(node.attributes.get('semantic_alias_terms') or []), 'is_salient_label': bool(node.attributes.get('is_salient_label')), 'attributes': dict(node.attributes), 'raw_visual_attributes': list(node.attributes.get('raw_visual_attributes') or []), 'time_range': node.time_range.model_dump(mode='json')}
        member_count = int(node.attributes.get('member_count') or 0)
        raw_visual_attributes = [str(item) for item in node.attributes.get('raw_visual_attributes', []) if str(item).strip()]
        compact_suffix = f' aggregating {member_count} source entities' if member_count > 1 else ''
        projection.add_record(_make_record(source_kind='entity', source_record_id=node.node_id, summary=f'Entity {node.entity_id} is labeled {node.label}{compact_suffix} from {_format_time_range(node.time_range)}.', metadata=metadata, fields=[node.node_id, node.entity_id, node.label, node.node_role, *node.labels_seen, *node.states_seen, *raw_visual_attributes, *node.parent_labels_seen, *[str(item) for item in node.attributes.get('semantic_alias_terms', [])], str(node.attributes.get('canonical_label') or ''), str(node.attributes.get('semantic_group') or ''), *[str(item) for item in node.attributes.get('member_entity_ids', [])]], video_id=node.video_id, scene_ids=tuple(sorted(scene_ids_by_entity.get(node.entity_id, set()))), entity_ids=(node.entity_id,), start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
        for attribute_index, attribute_value in enumerate(raw_visual_attributes, start=1):
            record_id = f'{node.node_id}:raw_visual_attribute:{attribute_index:04d}'
            projection.add_record(_make_record(source_kind='entity_attribute', source_record_id=record_id, summary=f'Entity {node.entity_id} ({node.label}) has visual attribute {attribute_value} during {_format_time_range(node.time_range)}.', metadata={'node_id': record_id, 'source_entity_node_id': node.node_id, 'entity_id': node.entity_id, 'label': node.label, 'canonical_label': node.attributes.get('canonical_label'), 'semantic_group': node.attributes.get('semantic_group'), 'semantic_alias_terms': list(node.attributes.get('semantic_alias_terms') or []), 'attr_key': 'visual_attribute', 'attr_value': attribute_value, 'raw_visual_attribute': attribute_value, 'time_range': node.time_range.model_dump(mode='json')}, fields=[node.node_id, node.entity_id, node.label, str(node.attributes.get('canonical_label') or ''), str(node.attributes.get('semantic_group') or ''), *[str(item) for item in node.attributes.get('semantic_alias_terms', [])], 'visual attribute', attribute_value], video_id=node.video_id, scene_ids=tuple(sorted(scene_ids_by_entity.get(node.entity_id, set()))), entity_ids=(node.entity_id,), start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
            projection.add_link(source_record_id=node.node_id, target_record_id=record_id, edge_type='DESCRIBES_ENTITY', edge_id=f'projection:DESCRIBES_ENTITY:{node.node_id}:{record_id}')
    for node in graph.stable_identity_nodes:
        metadata = {'node_id': node.node_id, 'stable_id': node.stable_id, 'identity_type': node.identity_type, 'canonical_label': node.canonical_label, 'display_name': node.display_name, 'member_entity_ids': list(node.member_entity_ids), 'labels_seen': list(node.labels_seen), 'signature_tokens': list(node.signature_tokens), 'scene_ids': list(node.scene_ids), 'observation_count': node.observation_count, 'attributes': dict(node.attributes), 'raw_visual_attributes': list(node.attributes.get('raw_visual_attributes') or []), 'time_range': node.time_range.model_dump(mode='json')}
        raw_visual_attributes = [str(item) for item in node.attributes.get('raw_visual_attributes', []) if str(item).strip()]
        member_preview = ', '.join(node.member_entity_ids[:3]) or 'no members'
        projection.add_record(_make_record(source_kind='stable_identity', source_record_id=node.node_id, summary=f'Stable identity {node.display_name} ({node.canonical_label}) covers {member_preview} during {_format_time_range(node.time_range)}.', metadata=metadata, fields=[node.node_id, node.stable_id, node.identity_type, node.canonical_label, node.display_name, *node.member_entity_ids, *node.labels_seen, *node.signature_tokens, *raw_visual_attributes, *node.scene_ids], video_id=node.video_id, scene_ids=tuple(node.scene_ids), entity_ids=tuple(node.member_entity_ids), start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
    for node in graph.state_value_nodes:
        metadata = {'node_id': node.node_id, 'attr_key': node.attr_key, 'attr_value': node.attr_value, 'label': node.label}
        projection.add_record(_make_record(source_kind='state_value', source_record_id=node.node_id, summary=f'State value {node.attr_key}={node.attr_value}.', metadata=metadata, fields=[node.node_id, node.attr_key, node.attr_value, node.label], video_id=node.video_id, scene_ids=(), entity_ids=tuple(sorted(entity_ids_by_state_value.get((node.attr_key, node.attr_value), set()))), start_ts=None, end_ts=None))
    for node in graph.interaction_type_nodes:
        metadata = {'node_id': node.node_id, 'predicate': node.predicate, 'label': node.label, 'raw_predicates_seen': list(node.raw_predicates_seen)}
        projection.add_record(_make_record(source_kind='interaction_type', source_record_id=node.node_id, summary=f'Interaction type {node.predicate}.', metadata=metadata, fields=[node.node_id, node.predicate, node.label, *node.raw_predicates_seen], video_id=node.video_id, scene_ids=(), entity_ids=tuple(sorted(entity_ids_by_predicate.get(node.predicate, set()))), start_ts=None, end_ts=None))
    for node in graph.episode_nodes:
        metadata = {'node_id': node.node_id, 'episode_id': node.episode_id, 'episode_kind': node.episode_kind, 'label': node.label, 'scene_id': node.scene_id, 'primary_entity_id': node.primary_entity_id, 'primary_entity_label': entity_label_by_entity_id.get(node.primary_entity_id or ''), 'subject_entity_id': node.subject_entity_id, 'subject_entity_label': entity_label_by_entity_id.get(node.subject_entity_id or ''), 'object_entity_id': node.object_entity_id, 'object_entity_label': entity_label_by_entity_id.get(node.object_entity_id or ''), 'participant_entity_ids': list(node.participant_entity_ids), 'attr_key': node.attr_key, 'attr_value': node.attr_value, 'predicate': node.predicate, 'raw_predicate': node.raw_predicate, 'segment_index': node.segment_index, 'segment_count': node.segment_count, 'source_ref_id': node.source_ref_id, 'attributes': dict(node.attributes), 'time_range': node.time_range.model_dump(mode='json')}
        projection.add_record(_make_record(source_kind='episode', source_record_id=node.node_id, summary=_episode_summary(node), metadata=metadata, fields=[node.node_id, node.episode_id, node.episode_kind, node.label, node.scene_id or '', node.primary_entity_id or '', node.subject_entity_id or '', node.object_entity_id or '', node.attr_key or '', node.attr_value or '', node.predicate or '', node.raw_predicate or '', *node.participant_entity_ids], video_id=node.video_id, scene_ids=(node.scene_id,) if node.scene_id else (), entity_ids=tuple((entity_id for entity_id in dict.fromkeys(node.participant_entity_ids) if entity_id)), start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
    _add_relation_observed_burst_records(projection, graph=graph, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id, interaction_node_id_by_predicate=interaction_node_id_by_predicate)
    for node in graph.transition_event_nodes:
        if node.event_type == 'RELATION_OBSERVED':
            continue
        entity_ids = tuple((entity_id for entity_id in dict.fromkeys([node.entity_id, node.subject_entity_id, node.object_entity_id]) if entity_id))
        metadata = {'node_id': node.node_id, 'event_id': node.event_id, 'event_type': node.event_type, 'label': node.label, 'scene_id': node.scene_id, 'entity_id': node.entity_id, 'entity_label': entity_label_by_entity_id.get(node.entity_id or ''), 'subject_entity_id': node.subject_entity_id, 'subject_entity_label': entity_label_by_entity_id.get(node.subject_entity_id or ''), 'object_entity_id': node.object_entity_id, 'object_entity_label': entity_label_by_entity_id.get(node.object_entity_id or ''), 'predicate': node.predicate, 'raw_predicate': node.raw_predicate, 'attr_key': node.attr_key, 'previous_state_value': node.previous_state_value, 'current_state_value': node.current_state_value, 'source_ref_id': node.source_ref_id, 'time_range': node.time_range.model_dump(mode='json')}
        projection.add_record(_make_record(source_kind='transition_event', source_record_id=node.node_id, summary=f'Transition event {node.event_type} ({node.label}) at {_format_time_range(node.time_range)}.', metadata=metadata, fields=[node.node_id, node.event_id, node.event_type, node.label, node.scene_id or '', node.entity_id or '', node.subject_entity_id or '', node.object_entity_id or '', node.predicate or '', node.raw_predicate or '', node.attr_key or '', node.previous_state_value or '', node.current_state_value or ''], video_id=node.video_id, scene_ids=(node.scene_id,) if node.scene_id else (), entity_ids=entity_ids, start_ts=node.time_range.start_ts, end_ts=node.time_range.end_ts))
    if include_base_delta_projection and temporal_document is not None:
        _add_base_delta_projection_records(projection, graph=graph, document=temporal_document, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id)
    _add_temporal_composition_records(projection, graph=graph, scene_node_id_by_scene_id=scene_node_id_by_scene_id)
    for edge in graph.edges:
        projection.add_edge(edge)
    _add_virtual_structural_links(projection, graph=graph, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id)
    _add_virtual_previous_episode_links(projection, graph=graph)
    _add_virtual_derived_temporal_links(projection, graph=graph)
    return projection

def _get_projection(graph: CanonicalTemporalMemoryGraph, *, temporal_document: TemporalGraphDocument | None=None, include_base_delta_projection: bool=False, projection_cache_path: str | Path | None=None) -> _CanonicalProjection:
    fingerprint = _projection_cache_fingerprint(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection)
    cache_key = _projection_cache_key(fingerprint)
    cached = _PROJECTION_CACHE.get(cache_key)
    if cached is not None:
        if projection_cache_path is not None and (not Path(projection_cache_path).exists()):
            _write_projection_cache(projection_cache_path, projection=cached, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, fingerprint=fingerprint)
        return cached
    if projection_cache_path is not None:
        disk_cached = _load_projection_cache(projection_cache_path, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, expected_fingerprint=fingerprint)
        if disk_cached is not None:
            _PROJECTION_CACHE[cache_key] = disk_cached
            return disk_cached
    projection = _build_projection(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection)
    _PROJECTION_CACHE[cache_key] = projection
    if projection_cache_path is not None:
        _write_projection_cache(projection_cache_path, projection=projection, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, fingerprint=fingerprint)
    if len(_PROJECTION_CACHE) > 8:
        oldest_key = next(iter(_PROJECTION_CACHE))
        _PROJECTION_CACHE.pop(oldest_key, None)
    return projection

def _projection_cache_key(fingerprint: dict[str, Any]) -> str:
    return json.dumps(fingerprint, ensure_ascii=True, sort_keys=True, separators=(',', ':'))

def _projection_cache_fingerprint(graph: CanonicalTemporalMemoryGraph, *, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool) -> dict[str, Any]:
    graph_signature = _stable_digest({'scenes': [(node.node_id, node.scene_id, node.label, node.time_range.model_dump(mode='json'), node.scene_kind_hint, node.representative_static_labels, node.representative_dynamic_labels, node.entry_static_labels, node.exit_static_labels, node.salient_labels, node.semantic_alias_terms) for node in graph.scene_nodes], 'entities': [(node.node_id, node.entity_id, node.label, node.time_range.model_dump(mode='json'), node.observation_count, node.stable_identity_id, node.labels_seen, node.states_seen) for node in graph.entity_nodes], 'stable_identities': [(node.node_id, node.stable_id, node.canonical_label, node.display_name, node.time_range.model_dump(mode='json'), node.member_entity_ids, node.labels_seen, node.signature_tokens, node.scene_ids) for node in graph.stable_identity_nodes], 'episodes': [(node.node_id, node.episode_id, node.episode_kind, node.label, node.time_range.model_dump(mode='json'), node.scene_id, node.primary_entity_id, node.subject_entity_id, node.object_entity_id, node.participant_entity_ids, node.attr_key, node.attr_value, node.predicate, node.raw_predicate, node.source_ref_id) for node in graph.episode_nodes], 'events': [(node.node_id, node.event_id, node.event_type, node.label, node.time_range.model_dump(mode='json'), node.scene_id, node.entity_id, node.subject_entity_id, node.object_entity_id, node.predicate, node.raw_predicate, node.attr_key, node.previous_state_value, node.current_state_value, node.source_ref_id) for node in graph.transition_event_nodes], 'relation_observed_samples': [(node.sample_id, node.event_id, node.label, node.time_range.model_dump(mode='json'), node.scene_id, node.subject_entity_id, node.predicate, node.raw_predicate, node.object_entity_id, node.source_ref_id) for node in graph.relation_observed_samples], 'edges': [(edge.edge_id, edge.edge_type, edge.source_node_id, edge.target_node_id, edge.time_range.model_dump(mode='json') if edge.time_range else None) for edge in graph.edges]})
    return {'schema_version': PROJECTION_CACHE_SCHEMA_VERSION, 'graph_schema_version': graph.schema_version, 'video_id': graph.video_id, 'graph_signature': graph_signature, 'node_count': graph.node_count, 'edge_count': graph.edge_count, 'scene_count': len(graph.scene_nodes), 'entity_count': len(graph.entity_nodes), 'stable_identity_count': len(graph.stable_identity_nodes), 'state_value_count': len(graph.state_value_nodes), 'interaction_type_count': len(graph.interaction_type_nodes), 'episode_count': len(graph.episode_nodes), 'transition_event_count': len(graph.transition_event_nodes), 'relation_observed_sample_count': len(graph.relation_observed_samples), 'include_base_delta_projection': include_base_delta_projection, 'temporal_document_video_id': temporal_document.video_id if temporal_document is not None else None, 'temporal_document_frame_count': temporal_document.frame_count if temporal_document is not None else None, 'temporal_document_scene_count': len(temporal_document.scene_segments) if temporal_document is not None else 0, 'temporal_document_entity_count': len(temporal_document.entities) if temporal_document is not None else 0, 'temporal_document_relation_episode_count': len(temporal_document.relation_episodes) if temporal_document is not None else 0, 'temporal_document_state_episode_count': len(temporal_document.state_episodes) if temporal_document is not None else 0}

def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
    return hashlib.md5(payload.encode('utf-8'), usedforsecurity=False).hexdigest()

def _load_projection_cache(path: str | Path, *, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, expected_fingerprint: dict[str, Any] | None=None) -> _CanonicalProjection | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    if _projection_cache_uses_pickle(cache_path):
        return _load_pickle_projection_cache(cache_path, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, expected_fingerprint=expected_fingerprint)
    return _load_light_projection_cache(cache_path, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, expected_fingerprint=expected_fingerprint)

def _load_pickle_projection_cache(cache_path: Path, *, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, expected_fingerprint: dict[str, Any] | None=None) -> _CanonicalProjection | None:
    try:
        with cache_path.open('rb') as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = expected_fingerprint or _projection_cache_fingerprint(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection)
    if payload.get('fingerprint') != expected:
        return None
    projection = payload.get('projection')
    if not isinstance(projection, _CanonicalProjection):
        return None
    return projection

def _load_light_projection_cache(cache_path: Path, *, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, expected_fingerprint: dict[str, Any] | None=None) -> _CanonicalProjection | None:
    try:
        payload = json.loads(_read_projection_cache_text(cache_path))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = expected_fingerprint or _projection_cache_fingerprint(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection)
    if payload.get('fingerprint') != expected:
        return None
    try:
        return _projection_from_light_payload(payload)
    except (KeyError, TypeError, ValueError):
        return None

def _write_projection_cache(path: str | Path, *, projection: _CanonicalProjection, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, fingerprint: dict[str, Any] | None=None) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if _projection_cache_uses_pickle(cache_path):
        _write_pickle_projection_cache(cache_path, projection=projection, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, fingerprint=fingerprint)
        return
    _write_light_projection_cache(cache_path, projection=projection, graph=graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection, fingerprint=fingerprint)

def _write_pickle_projection_cache(cache_path: Path, *, projection: _CanonicalProjection, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, fingerprint: dict[str, Any] | None=None) -> None:
    payload = {'fingerprint': fingerprint or _projection_cache_fingerprint(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection), 'projection': projection}
    tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
    with tmp_path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)

def _write_light_projection_cache(cache_path: Path, *, projection: _CanonicalProjection, graph: CanonicalTemporalMemoryGraph, temporal_document: TemporalGraphDocument | None, include_base_delta_projection: bool, fingerprint: dict[str, Any] | None=None) -> None:
    payload = {'fingerprint': fingerprint or _projection_cache_fingerprint(graph, temporal_document=temporal_document, include_base_delta_projection=include_base_delta_projection), 'projection': _projection_to_light_payload(projection)}
    tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
    _write_projection_cache_text(tmp_path, json.dumps(payload, ensure_ascii=False, separators=(',', ':')), compression_path=cache_path)
    tmp_path.replace(cache_path)

def _projection_cache_uses_pickle(cache_path: Path) -> bool:
    return cache_path.suffix.lower() in {'.pkl', '.pickle'}

def _read_projection_cache_text(cache_path: Path) -> str:
    if cache_path.suffix.lower() == '.gz':
        with gzip.open(cache_path, 'rt', encoding='utf-8') as handle:
            return handle.read()
    return cache_path.read_text(encoding='utf-8')

def _write_projection_cache_text(cache_path: Path, text: str, *, compression_path: Path | None=None) -> None:
    resolved_path = compression_path or cache_path
    if resolved_path.suffix.lower() == '.gz':
        with gzip.open(cache_path, 'wt', encoding='utf-8') as handle:
            handle.write(text)
        return
    cache_path.write_text(text, encoding='utf-8')

def _projection_to_light_payload(projection: _CanonicalProjection) -> dict[str, Any]:
    return {'records': [{'source_kind': record.source_kind, 'source_record_id': record.source_record_id, 'summary': record.summary, 'metadata': record.metadata, 'index_text': record.index_text, 'tokens': sorted(record.tokens), 'video_id': record.video_id, 'scene_ids': list(record.scene_ids), 'entity_ids': list(record.entity_ids), 'start_ts': record.start_ts, 'end_ts': record.end_ts, 'priority': record.priority} for record in sorted(projection.records.values(), key=lambda item: item.source_record_id)], 'links': [{'source_record_id': source_record_id, 'neighbor_id': link.neighbor_id, 'edge_type': link.edge_type, 'edge_id': link.edge_id} for source_record_id, links in sorted(projection.adjacency.items()) for link in links if source_record_id <= link.neighbor_id]}

def _projection_from_light_payload(payload: dict[str, Any]) -> _CanonicalProjection:
    projection_payload = payload.get('projection')
    if not isinstance(projection_payload, dict):
        raise TypeError('projection payload must be a dict')
    projection = _CanonicalProjection()
    for item in projection_payload.get('records', []):
        if not isinstance(item, dict):
            continue
        record = _GraphRecordView(source_kind=item['source_kind'], source_record_id=str(item['source_record_id']), summary=str(item.get('summary') or ''), metadata=dict(item.get('metadata') or {}), index_text=str(item.get('index_text') or ''), tokens=frozenset((str(token) for token in item.get('tokens', []))), video_id=str(item.get('video_id') or ''), scene_ids=tuple((str(value) for value in item.get('scene_ids', []))), entity_ids=tuple((str(value) for value in item.get('entity_ids', []))), start_ts=item.get('start_ts'), end_ts=item.get('end_ts'), priority=int(item.get('priority') or PRIORITY_BY_KIND[item['source_kind']]))
        projection.add_record(record)
    for item in projection_payload.get('links', []):
        if not isinstance(item, dict):
            continue
        projection.add_link(source_record_id=str(item.get('source_record_id') or ''), target_record_id=str(item.get('neighbor_id') or ''), edge_type=str(item.get('edge_type') or ''), edge_id=str(item.get('edge_id') or ''))
    return projection

def _add_base_delta_projection_records(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, document: TemporalGraphDocument, entity_node_id_by_entity_id: dict[str, str], scene_node_id_by_scene_id: dict[str, str]) -> None:
    scene_segments_by_id = {scene.scene_id: scene for scene in document.scene_segments}
    delta_observations_by_node_id: dict[str, list[Any]] = defaultdict(list)
    for observation in document.scene_delta_observations:
        delta_observations_by_node_id[observation.scene_delta_node_id].append(observation)
    for observations in delta_observations_by_node_id.values():
        observations.sort(key=lambda item: (item.frame_index, item.delta_observation_id))
    base_record_id_by_scene_entity: dict[tuple[str, str], str] = {}
    for node in sorted(document.scene_base_nodes, key=lambda item: (item.scene_id, item.entity_id, item.node_id)):
        record_id = _scene_base_projection_record_id(node.node_id)
        scene_segment = scene_segments_by_id.get(node.scene_id)
        start_ts = _timestamp_for_scene_frame(scene_segment, node.first_frame)
        end_ts = _timestamp_for_scene_frame(scene_segment, node.last_frame)
        time_range = {'start_frame': node.first_frame, 'end_frame': node.last_frame, 'start_ts': start_ts, 'end_ts': end_ts, 'support_frames': node.support_frames}
        summary = f'Scene base anchor {node.entity_id} ({node.label}) is stable context in {node.scene_id} across frames {node.first_frame}-{node.last_frame} with coverage {node.coverage_ratio:.2f}.'
        metadata = {'node_id': record_id, 'source_ref_id': node.node_id, 'projection_kind': 'scene_base', 'scene_id': node.scene_id, 'entity_id': node.entity_id, 'label': node.label, 'coverage_ratio': node.coverage_ratio, 'support_frames': node.support_frames, 'parent_labels_seen': list(node.parent_labels_seen), 'states_seen': list(node.states_seen), 'time_range': time_range}
        projection.add_record(_make_record(source_kind='scene_base', source_record_id=record_id, summary=summary, metadata=metadata, fields=[record_id, node.node_id, 'scene base static stable background anchor', node.scene_id, node.entity_id, node.label, *node.parent_labels_seen, *node.states_seen], video_id=node.video_id, scene_ids=(node.scene_id,), entity_ids=(node.entity_id,), start_ts=start_ts, end_ts=end_ts))
        base_record_id_by_scene_entity[node.scene_id, node.entity_id] = record_id
        _link_projection_record_to_scene_and_entity(projection, record_id=record_id, scene_id=node.scene_id, entity_id=node.entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_edge_type='SCENE_CONTAINS', entity_edge_type='HAS_MEMBER_ENTITY')
    for node in sorted(document.scene_delta_nodes, key=lambda item: (item.scene_id, item.start_frame, item.entity_id, item.node_id)):
        record_id = _scene_delta_projection_record_id(node.node_id)
        observations = delta_observations_by_node_id.get(node.node_id, [])
        observed_frames = [int(item.frame_index) for item in observations]
        observed_timestamps = [float(item.timestamp_sec) for item in observations]
        trigger_event_types = list(dict.fromkeys((event_type for observation in observations for event_type in observation.trigger_event_types if event_type)))
        trigger_event_ids = list(dict.fromkeys((event_id for observation in observations for event_id in observation.trigger_event_ids if event_id)))
        time_range = {'start_frame': node.start_frame, 'end_frame': node.end_frame, 'start_ts': node.start_ts, 'end_ts': node.end_ts, 'support_frames': node.support_frames}
        summary = f'Scene delta {node.entity_id} ({node.label}) updates {node.scene_id} from {_format_time_range(time_range)} with {node.transition_event_count} transition event(s), {node.relation_episode_count} relation episode(s), {node.state_episode_count} state episode(s), and {len(observed_frames)} sampled delta observation(s).'
        metadata = {'node_id': record_id, 'source_ref_id': node.node_id, 'projection_kind': 'scene_delta', 'scene_id': node.scene_id, 'entity_id': node.entity_id, 'label': node.label, 'transition_event_count': node.transition_event_count, 'relation_episode_count': node.relation_episode_count, 'state_episode_count': node.state_episode_count, 'observed_frames': observed_frames, 'observed_timestamps': observed_timestamps, 'trigger_event_types': trigger_event_types, 'trigger_event_ids': trigger_event_ids, 'time_range': time_range}
        projection.add_record(_make_record(source_kind='scene_delta', source_record_id=record_id, summary=summary, metadata=metadata, fields=[record_id, node.node_id, 'scene delta dynamic change update transition', node.scene_id, node.entity_id, node.label, *trigger_event_types, *[str(frame) for frame in observed_frames[:8]]], video_id=node.video_id, scene_ids=(node.scene_id,), entity_ids=(node.entity_id,), start_ts=node.start_ts, end_ts=node.end_ts))
        _link_projection_record_to_scene_and_entity(projection, record_id=record_id, scene_id=node.scene_id, entity_id=node.entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_edge_type='SCENE_UPDATES', entity_edge_type='HAS_EPISODE')
        base_record_id = base_record_id_by_scene_entity.get((node.scene_id, node.entity_id))
        if base_record_id is not None:
            projection.add_link(source_record_id=base_record_id, target_record_id=record_id, edge_type='SCENE_UPDATES', edge_id=f'projection:SCENE_UPDATES:{base_record_id}:{record_id}')

def _add_temporal_composition_records(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, scene_node_id_by_scene_id: dict[str, str]) -> None:
    ordered_scenes = sorted(graph.scene_nodes, key=lambda item: (item.time_range.start_frame if item.time_range.start_frame is not None else math.inf, item.scene_index, item.scene_id))
    if len(ordered_scenes) < 2:
        return
    for chain_size in (2, 3):
        if len(ordered_scenes) < chain_size:
            continue
        for start_index in range(0, len(ordered_scenes) - chain_size + 1):
            scenes = ordered_scenes[start_index:start_index + chain_size]
            start_scene = scenes[0]
            end_scene = scenes[-1]
            if start_scene.video_id != end_scene.video_id:
                continue
            record_id = _temporal_composition_record_id(scenes)
            scene_ids = [scene.scene_id for scene in scenes]
            label_sequence = [_scene_signature_text(scene) for scene in scenes]
            context_sequence = [_scene_context_hint_text(scene) for scene in scenes]
            transition_labels = _scene_transition_labels(scenes)
            transition_tokens = _scene_transition_tokens(transition_labels)
            scene_context_transition_labels = _scene_context_transition_labels(scenes)
            scene_context_transition_tokens = _scene_transition_tokens(scene_context_transition_labels)
            boundary_change_score = _scene_chain_boundary_change_score(scenes)
            start_frame = start_scene.time_range.start_frame
            end_frame = end_scene.time_range.end_frame
            start_ts = start_scene.time_range.start_ts
            end_ts = end_scene.time_range.end_ts
            support_frames = sum((int(scene.time_range.support_frames or 0) for scene in scenes)) or None
            time_range = {'start_frame': start_frame, 'end_frame': end_frame, 'start_ts': start_ts, 'end_ts': end_ts, 'support_frames': support_frames}
            summary = f"Temporal scene chain {' -> '.join(scene_ids)} spans {_format_time_values(start_frame, end_frame, start_ts, end_ts)} with scene signatures {' -> '.join(label_sequence)}."
            metadata = {'node_id': record_id, 'composition_type': 'scene_sequence', 'chain_size': chain_size, 'scene_ids': scene_ids, 'scene_sequence': [_serialize_temporal_composition_scene(scene) for scene in scenes], 'context_sequence': context_sequence, 'transition_labels': transition_labels, 'transition_tokens': sorted(transition_tokens), 'scene_context_transition_labels': scene_context_transition_labels, 'scene_context_transition_tokens': sorted(scene_context_transition_tokens), 'boundary_change_score': round(boundary_change_score, 4), 'time_range': time_range}
            projection.add_record(_make_record(source_kind='temporal_composition', source_record_id=record_id, summary=summary, metadata=metadata, fields=[record_id, 'temporal composition scene sequence transition duration timeline', *scene_ids, *label_sequence, *context_sequence, *transition_labels, *sorted(transition_tokens), *scene_context_transition_labels, *sorted(scene_context_transition_tokens)], video_id=start_scene.video_id, scene_ids=tuple(scene_ids), entity_ids=(), start_ts=start_ts, end_ts=end_ts))
            for scene in scenes:
                scene_node_id = scene_node_id_by_scene_id.get(scene.scene_id)
                if scene_node_id is None:
                    continue
                projection.add_link(source_record_id=record_id, target_record_id=scene_node_id, edge_type='TEMPORALLY_NEAR', edge_id=f'projection:TEMPORAL_COMPOSITION:{record_id}:{scene_node_id}')

def _scene_base_projection_record_id(node_id: str) -> str:
    return f'scene_base:{node_id}'

def _scene_delta_projection_record_id(node_id: str) -> str:
    return f'scene_delta:{node_id}'

def _temporal_composition_record_id(scenes: list[Any]) -> str:
    scene_ids = [str(scene.scene_id) for scene in scenes]
    digest = hashlib.md5('||'.join(scene_ids).encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
    return f'temporal_composition:scene_sequence:{digest}'

def _serialize_temporal_composition_scene(scene: Any) -> dict[str, Any]:
    return {'scene_id': scene.scene_id, 'scene_index': scene.scene_index, 'scene_count': scene.scene_count, 'label': scene.label, 'scene_kind_hint': scene.scene_kind_hint, 'scene_context_hints': _scene_context_hints(scene), 'start_frame': scene.time_range.start_frame, 'end_frame': scene.time_range.end_frame, 'start_ts': scene.time_range.start_ts, 'end_ts': scene.time_range.end_ts, 'representative_static_labels': list(scene.representative_static_labels), 'representative_dynamic_labels': list(scene.representative_dynamic_labels), 'entry_static_labels': list(scene.entry_static_labels), 'exit_static_labels': list(scene.exit_static_labels), 'salient_labels': list(getattr(scene, 'salient_labels', []) or []), 'semantic_alias_terms': list(getattr(scene, 'semantic_alias_terms', []) or []), 'previous_scene_id': scene.previous_scene_id, 'next_scene_id': scene.next_scene_id, 'boundary_from_previous_reason': scene.boundary_from_previous_reason, 'boundary_to_next_reason': scene.boundary_to_next_reason}

def _scene_context_hint_text(scene: Any) -> str:
    return ' '.join(_scene_context_hints(scene))

def _scene_context_hints(scene: Any) -> list[str]:
    scene_tokens = _scene_signature_tokens(scene)
    hints: list[str] = []
    for context, cues in SCENE_CONTEXT_CUES.items():
        if scene_tokens & cues:
            hints.extend(SCENE_CONTEXT_ALIASES[context])
    return list(dict.fromkeys(hints))

def _scene_signature_text(scene: Any) -> str:
    labels = _scene_signature_labels(scene, limit=4)
    return ' / '.join(labels) if labels else str(scene.label or scene.scene_id)

def _scene_signature_labels(scene: Any, *, limit: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in [scene.scene_kind_hint, scene.label, *list(scene.representative_static_labels), *list(scene.representative_dynamic_labels), *list(scene.entry_static_labels), *list(scene.exit_static_labels), *list(getattr(scene, 'salient_labels', []) or [])]:
        for normalized in _iter_scene_label_atoms(value):
            normalized_key = normalized.lower()
            if normalized_key in seen:
                continue
            ordered.append(normalized)
            seen.add(normalized_key)
            if len(ordered) >= limit:
                return ordered
    return ordered

def _iter_scene_label_atoms(value: Any) -> list[str]:
    text = str(value or '').strip()
    if not text:
        return []
    atoms = [part.strip() for part in re.split('\\s*/\\s*|\\s*\\|\\s*', text) if part.strip()]
    return atoms or [text]

def _scene_transition_labels(scenes: list[Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for left, right in zip(scenes, scenes[1:], strict=False):
        left_labels = {label.lower() for label in _scene_signature_labels(left, limit=8)}
        right_labels = {label.lower() for label in _scene_signature_labels(right, limit=8)}
        entered = sorted(right_labels - left_labels)
        exited = sorted(left_labels - right_labels)
        for prefix, values in (('entered', entered), ('exited', exited)):
            for value in values[:4]:
                label = f'{prefix}:{value}'
                if label in seen:
                    continue
                labels.append(label)
                seen.add(label)
    return labels

def _scene_transition_tokens(transition_labels: list[str]) -> set[str]:
    tokens: set[str] = set()
    for label in transition_labels:
        prefix, _, value = str(label).partition(':')
        tokens.update(_tokenize(prefix))
        tokens.update(_tokenize(value or label))
    return tokens

def _scene_context_transition_labels(scenes: list[Any]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for left, right in zip(scenes, scenes[1:], strict=False):
        left_hints = set(_scene_context_hints(left))
        right_hints = set(_scene_context_hints(right))
        for label in _ordered_scene_context_transition_labels(left_hints, right_hints):
            if label in seen:
                continue
            labels.append(label)
            seen.add(label)
    return labels

def _ordered_scene_context_transition_labels(left_hints: set[str], right_hints: set[str]) -> list[str]:
    labels: list[str] = []
    left_indoor = bool(left_hints & {'indoor', 'inside'})
    left_outdoor = bool(left_hints & {'outdoor', 'outside'})
    right_indoor = bool(right_hints & {'indoor', 'inside'})
    right_outdoor = bool(right_hints & {'outdoor', 'outside'})
    if left_outdoor and (not left_indoor) and right_indoor and (not right_outdoor):
        labels.append('entered')
    if left_indoor and (not left_outdoor) and right_outdoor and (not right_indoor):
        labels.append('exited')
    return labels

def _scene_chain_boundary_change_score(scenes: list[Any]) -> float:
    if len(scenes) < 2:
        return 0.0
    scores: list[float] = []
    for left, right in zip(scenes, scenes[1:], strict=False):
        left_tokens = _scene_signature_tokens(left)
        right_tokens = _scene_signature_tokens(right)
        union = left_tokens | right_tokens
        if not union:
            continue
        scores.append(1.0 - len(left_tokens & right_tokens) / len(union))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def _scene_signature_tokens(scene: Any) -> set[str]:
    tokens: set[str] = set()
    for label in _scene_signature_labels(scene, limit=12):
        tokens.update(_tokenize(label))
    return tokens

def _timestamp_for_scene_frame(scene_segment: Any | None, frame_index: int) -> float | None:
    if scene_segment is None:
        return None
    start_frame = getattr(scene_segment, 'start_frame', None)
    end_frame = getattr(scene_segment, 'end_frame', None)
    start_ts = getattr(scene_segment, 'start_ts', None)
    end_ts = getattr(scene_segment, 'end_ts', None)
    if start_frame is None or end_frame is None or start_ts is None or (end_ts is None):
        return None
    if int(end_frame) == int(start_frame):
        return float(start_ts)
    position = (int(frame_index) - int(start_frame)) / (int(end_frame) - int(start_frame))
    position = min(1.0, max(0.0, position))
    return float(start_ts) + position * (float(end_ts) - float(start_ts))

def _link_projection_record_to_scene_and_entity(projection: _CanonicalProjection, *, record_id: str, scene_id: str, entity_id: str, scene_node_id_by_scene_id: dict[str, str], entity_node_id_by_entity_id: dict[str, str], scene_edge_type: str, entity_edge_type: str) -> None:
    scene_node_id = scene_node_id_by_scene_id.get(scene_id)
    if scene_node_id is not None:
        projection.add_link(source_record_id=scene_node_id, target_record_id=record_id, edge_type=scene_edge_type)
    entity_node_id = entity_node_id_by_entity_id.get(entity_id)
    if entity_node_id is not None:
        projection.add_link(source_record_id=entity_node_id, target_record_id=record_id, edge_type=entity_edge_type)

def _add_relation_observed_burst_records(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, entity_node_id_by_entity_id: dict[str, str], scene_node_id_by_scene_id: dict[str, str], interaction_node_id_by_predicate: dict[str, str]) -> None:
    grouped: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for node in _iter_relation_observed_sources(graph):
        if not node.subject_entity_id or not node.predicate or (not node.object_entity_id):
            continue
        scene_key = node.scene_id or '__scene_unknown__'
        grouped[scene_key, node.subject_entity_id, node.predicate, node.object_entity_id].append(node)
    for (scene_key, subject_entity_id, predicate, object_entity_id), nodes in sorted(grouped.items()):
        ordered = sorted(nodes, key=lambda item: (item.time_range.start_frame if item.time_range.start_frame is not None else math.inf, item.time_range.start_ts if item.time_range.start_ts is not None else math.inf, item.event_id))
        scene_id = None if scene_key == '__scene_unknown__' else scene_key
        observed_frames = [int(node.time_range.start_frame) for node in ordered if node.time_range.start_frame is not None]
        observed_timestamps = [float(node.time_range.start_ts) for node in ordered if node.time_range.start_ts is not None]
        raw_predicates = sorted({str(node.raw_predicate or '').strip() for node in ordered if str(node.raw_predicate or '').strip()})
        start_frame = min(observed_frames) if observed_frames else None
        end_frame = max(observed_frames) if observed_frames else None
        start_ts = min(observed_timestamps) if observed_timestamps else None
        end_ts = max(observed_timestamps) if observed_timestamps else None
        support_frames = len(observed_frames)
        source_event_ids = [node.event_id for node in ordered]
        source_record_ids = [_relation_observed_source_record_id(node) for node in ordered]
        sampled_frame_text = ', '.join((str(frame) for frame in observed_frames[:3]))
        if len(observed_frames) > 3:
            sampled_frame_text += ', ...'
        record_id = _relation_observed_burst_record_id(scene_id=scene_id, subject_entity_id=subject_entity_id, predicate=predicate, object_entity_id=object_entity_id)
        summary = f'Observed {subject_entity_id} {predicate} {object_entity_id} at {support_frames} sampled moment(s) (frames {sampled_frame_text}) between {_format_time_values(start_frame, end_frame, start_ts, end_ts)}.'
        metadata = {'node_id': record_id, 'episode_id': record_id, 'episode_kind': RELATION_OBSERVED_BURST_KIND, 'label': f'{subject_entity_id} {predicate} {object_entity_id}', 'scene_id': scene_id, 'primary_entity_id': subject_entity_id, 'subject_entity_id': subject_entity_id, 'object_entity_id': object_entity_id, 'participant_entity_ids': [subject_entity_id, object_entity_id], 'predicate': predicate, 'raw_predicate': ' | '.join(raw_predicates) if raw_predicates else None, 'source_ref_id': record_id, 'source_event_ids': list(source_event_ids), 'source_record_ids': list(source_record_ids), 'observed_frames': list(observed_frames), 'observed_timestamps': list(observed_timestamps), 'support_frames': support_frames, 'attributes': {'aggregation_kind': 'relation_observed_burst', 'raw_event_count': len(source_event_ids)}, 'time_range': {'start_frame': start_frame, 'end_frame': end_frame, 'start_ts': start_ts, 'end_ts': end_ts, 'support_frames': support_frames}}
        projection.add_record(_make_record(source_kind='episode', source_record_id=record_id, summary=summary, metadata=metadata, fields=[record_id, RELATION_OBSERVED_BURST_KIND, subject_entity_id, predicate, object_entity_id, scene_id or '', *raw_predicates, *[str(frame) for frame in observed_frames[:6]]], video_id=ordered[0].video_id, scene_ids=(scene_id,) if scene_id else (), entity_ids=(subject_entity_id, object_entity_id), start_ts=start_ts, end_ts=end_ts))
        scene_node_id = scene_node_id_by_scene_id.get(scene_id or '')
        if scene_node_id is not None:
            projection.add_link(source_record_id=record_id, target_record_id=scene_node_id, edge_type='IN_SCENE')
            projection.add_link(source_record_id=scene_node_id, target_record_id=record_id, edge_type='HAS_EPISODE')
        for entity_id, role in ((subject_entity_id, 'subject'), (object_entity_id, 'object')):
            entity_node_id = entity_node_id_by_entity_id.get(entity_id)
            if entity_node_id is None:
                continue
            projection.add_link(source_record_id=entity_node_id, target_record_id=record_id, edge_type='HAS_EPISODE')
        interaction_node_id = interaction_node_id_by_predicate.get(predicate)
        if interaction_node_id is not None:
            projection.add_link(source_record_id=record_id, target_record_id=interaction_node_id, edge_type='HAS_INTERACTION_TYPE')

def _iter_relation_observed_sources(graph: CanonicalTemporalMemoryGraph) -> list[Any]:
    sources: list[Any] = [node for node in graph.transition_event_nodes if node.event_type == 'RELATION_OBSERVED']
    sources.extend(graph.relation_observed_samples)
    return sources

def _relation_observed_source_record_id(node: Any) -> str:
    return str(getattr(node, 'node_id', None) or getattr(node, 'sample_id', ''))

def _add_virtual_structural_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, entity_node_id_by_entity_id: dict[str, str], scene_node_id_by_scene_id: dict[str, str]) -> None:
    materialized_edge_types = {edge.edge_type for edge in graph.edges}
    missing_edge_types = DERIVED_STRUCTURAL_EDGE_TYPES - materialized_edge_types
    if not missing_edge_types:
        return
    scene_segment_node_id_by_scene_id = {node.scene_id: node.node_id for node in graph.episode_nodes if node.episode_kind == 'SCENE_SEGMENT' and node.scene_id}
    entity_segment_nodes_by_entity_id: dict[str, list[_GraphRecordView]] = defaultdict(list)
    for episode in graph.episode_nodes:
        if episode.episode_kind != 'ENTITY_SEGMENT':
            continue
        if not episode.primary_entity_id:
            continue
        record = projection.records.get(episode.node_id)
        if record is None:
            continue
        entity_segment_nodes_by_entity_id[episode.primary_entity_id].append(record)
    for records in entity_segment_nodes_by_entity_id.values():
        records.sort(key=lambda item: (_record_start_frame(item) if _record_start_frame(item) is not None else math.inf, _record_end_frame(item) if _record_end_frame(item) is not None else math.inf, item.source_record_id))
    if 'IN_SCENE' in missing_edge_types:
        _add_virtual_in_scene_links(projection, graph=graph, scene_node_id_by_scene_id=scene_node_id_by_scene_id)
    if 'HAS_EPISODE' in missing_edge_types:
        _add_virtual_has_episode_links(projection, graph=graph, entity_node_id_by_entity_id=entity_node_id_by_entity_id, scene_node_id_by_scene_id=scene_node_id_by_scene_id)
    if 'BELONGS_TO_SEGMENT' in missing_edge_types:
        _add_virtual_belongs_to_segment_links(projection, graph=graph, scene_segment_node_id_by_scene_id=scene_segment_node_id_by_scene_id, entity_segment_nodes_by_entity_id=entity_segment_nodes_by_entity_id)

def _add_virtual_in_scene_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, scene_node_id_by_scene_id: dict[str, str]) -> None:
    for node in graph.episode_nodes:
        if not node.scene_id:
            continue
        scene_node_id = scene_node_id_by_scene_id.get(node.scene_id)
        if scene_node_id is None:
            continue
        projection.add_link(source_record_id=node.node_id, target_record_id=scene_node_id, edge_type='IN_SCENE', edge_id=f'projection:IN_SCENE:{node.node_id}:{scene_node_id}:from_episode_scene_id')
    for node in graph.transition_event_nodes:
        if not node.scene_id:
            continue
        scene_node_id = scene_node_id_by_scene_id.get(node.scene_id)
        if scene_node_id is None:
            continue
        projection.add_link(source_record_id=node.node_id, target_record_id=scene_node_id, edge_type='IN_SCENE', edge_id=f'projection:IN_SCENE:{node.node_id}:{scene_node_id}:from_event_scene_id')
    for node in graph.stable_identity_nodes:
        for scene_id in node.scene_ids:
            scene_node_id = scene_node_id_by_scene_id.get(scene_id)
            if scene_node_id is None:
                continue
            projection.add_link(source_record_id=node.node_id, target_record_id=scene_node_id, edge_type='IN_SCENE', edge_id=f'projection:IN_SCENE:{node.node_id}:{scene_node_id}:from_stable_identity_scene_id')

def _add_virtual_has_episode_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, entity_node_id_by_entity_id: dict[str, str], scene_node_id_by_scene_id: dict[str, str]) -> None:
    for episode in graph.episode_nodes:
        if episode.scene_id:
            scene_node_id = scene_node_id_by_scene_id.get(episode.scene_id)
            if scene_node_id is not None:
                projection.add_link(source_record_id=scene_node_id, target_record_id=episode.node_id, edge_type='HAS_EPISODE', edge_id=f'projection:HAS_EPISODE:{scene_node_id}:{episode.node_id}:from_episode_scene_id')
        for entity_id, role in _episode_has_episode_entity_roles(episode):
            entity_node_id = entity_node_id_by_entity_id.get(entity_id)
            if entity_node_id is None:
                continue
            projection.add_link(source_record_id=entity_node_id, target_record_id=episode.node_id, edge_type='HAS_EPISODE', edge_id=f'projection:HAS_EPISODE:{entity_node_id}:{episode.node_id}:from_episode_{role}')

def _episode_has_episode_entity_roles(episode: Any) -> list[tuple[str, str]]:
    if getattr(episode, 'episode_kind', None) == 'RELATION':
        pairs = [(getattr(episode, 'subject_entity_id', None), 'subject'), (getattr(episode, 'object_entity_id', None), 'object')]
    else:
        pairs = [(getattr(episode, 'primary_entity_id', None), 'primary'), *[(entity_id, 'participant') for entity_id in getattr(episode, 'participant_entity_ids', [])]]
    results: list[tuple[str, str]] = []
    seen: set[str] = set()
    for entity_id, role in pairs:
        if not entity_id or entity_id in seen:
            continue
        seen.add(entity_id)
        results.append((entity_id, role))
    return results

def _add_virtual_belongs_to_segment_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph, scene_segment_node_id_by_scene_id: dict[str, str], entity_segment_nodes_by_entity_id: dict[str, list[_GraphRecordView]]) -> None:
    for episode in graph.episode_nodes:
        if episode.episode_kind == 'SCENE_SEGMENT':
            continue
        if episode.scene_id:
            scene_segment_node_id = scene_segment_node_id_by_scene_id.get(episode.scene_id)
            if scene_segment_node_id is not None and scene_segment_node_id != episode.node_id:
                projection.add_link(source_record_id=episode.node_id, target_record_id=scene_segment_node_id, edge_type='BELONGS_TO_SEGMENT', edge_id=f'projection:BELONGS_TO_SEGMENT:{episode.node_id}:{scene_segment_node_id}:from_scene_id')
        if episode.episode_kind == 'ENTITY_SEGMENT':
            continue
        for entity_id, role in _episode_segment_entity_roles(episode):
            segment_node_id = _best_projection_entity_segment_record_id(episode, entity_segment_nodes_by_entity_id.get(entity_id, []))
            if segment_node_id is None or segment_node_id == episode.node_id:
                continue
            projection.add_link(source_record_id=episode.node_id, target_record_id=segment_node_id, edge_type='BELONGS_TO_SEGMENT', edge_id=f'projection:BELONGS_TO_SEGMENT:{episode.node_id}:{segment_node_id}:from_{role}_entity_id')

def _episode_segment_entity_roles(episode: Any) -> list[tuple[str, str]]:
    if getattr(episode, 'episode_kind', None) == 'RELATION':
        return _episode_has_episode_entity_roles(episode)
    primary_entity_id = getattr(episode, 'primary_entity_id', None)
    return [(primary_entity_id, 'primary')] if primary_entity_id else []

def _best_projection_entity_segment_record_id(episode: Any, candidate_records: list[_GraphRecordView]) -> str | None:
    if not candidate_records:
        return None
    episode_start = getattr(episode.time_range, 'start_frame', None)
    episode_end = getattr(episode.time_range, 'end_frame', None)
    best_record: _GraphRecordView | None = None
    best_overlap = -1
    for record in candidate_records:
        record_start = _record_start_frame(record)
        record_end = _record_end_frame(record)
        if episode_start is None or episode_end is None or record_start is None or (record_end is None):
            overlap = 0
        else:
            overlap = min(episode_end, record_end) - max(episode_start, record_start) + 1
        if overlap > best_overlap:
            best_overlap = overlap
            best_record = record
    if best_record is None or best_overlap <= 0:
        return None
    return best_record.source_record_id

def _add_virtual_previous_episode_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph) -> None:
    edge_types = {edge.edge_type for edge in graph.edges}
    if DERIVED_PREVIOUS_EPISODE_EDGE_TYPE in edge_types or 'NEXT_EPISODE' not in edge_types:
        return
    for edge in graph.edges:
        if edge.edge_type != 'NEXT_EPISODE':
            continue
        projection.add_link(source_record_id=edge.target_node_id, target_record_id=edge.source_node_id, edge_type=DERIVED_PREVIOUS_EPISODE_EDGE_TYPE, edge_id=f'projection:PREVIOUS_EPISODE:{edge.target_node_id}:{edge.source_node_id}:from_next_episode')

def _add_virtual_derived_temporal_links(projection: _CanonicalProjection, *, graph: CanonicalTemporalMemoryGraph) -> None:
    missing_edge_types = DERIVED_TEMPORAL_EDGE_TYPES - {edge.edge_type for edge in graph.edges}
    if not missing_edge_types:
        return
    semantic_episodes = [record for record in projection.records.values() if record.source_kind == 'episode' and str(record.metadata.get('episode_kind') or '') in TEMPORAL_SEMANTIC_EPISODE_KINDS]
    transition_events = [record for record in projection.records.values() if record.source_kind == 'transition_event' and str(record.metadata.get('event_type') or '') != 'RELATION_OBSERVED']
    if 'OVERLAPS_WITH' in missing_edge_types:
        _add_virtual_event_overlap_links(projection, episodes=semantic_episodes, transition_events=transition_events)
    _add_virtual_episode_temporal_links(projection, episodes=semantic_episodes, include_overlap='OVERLAPS_WITH' in missing_edge_types, include_near='TEMPORALLY_NEAR' in missing_edge_types)

def _add_virtual_event_overlap_links(projection: _CanonicalProjection, *, episodes: list[_GraphRecordView], transition_events: list[_GraphRecordView]) -> None:
    for event in transition_events:
        event_frame = _record_start_frame(event)
        if event_frame is None:
            continue
        event_entities = set(event.entity_ids)
        if not event_entities:
            continue
        for episode in episodes:
            if not _same_known_record_scene(episode, event):
                continue
            shared_entities = set(episode.entity_ids) & event_entities
            if not shared_entities:
                continue
            start_frame = _record_start_frame(episode)
            end_frame = _record_end_frame(episode)
            if start_frame is None or end_frame is None:
                continue
            if start_frame < event_frame < end_frame:
                projection.add_link(source_record_id=episode.source_record_id, target_record_id=event.source_record_id, edge_type='OVERLAPS_WITH', edge_id=f'projection:OVERLAPS_WITH:{episode.source_record_id}:{event.source_record_id}:temporal_semantic:event_inside_interval')

def _add_virtual_episode_temporal_links(projection: _CanonicalProjection, *, episodes: list[_GraphRecordView], include_overlap: bool, include_near: bool) -> None:
    if not include_overlap and (not include_near):
        return
    for left_index, left in enumerate(episodes):
        for right in episodes[left_index + 1:]:
            if left.metadata.get('episode_kind') == right.metadata.get('episode_kind'):
                continue
            if not _same_known_record_scene(left, right):
                continue
            if not set(left.entity_ids) & set(right.entity_ids):
                continue
            if _records_overlap(left, right):
                if include_overlap:
                    source, target = _ordered_temporal_records(left, right)
                    projection.add_link(source_record_id=source.source_record_id, target_record_id=target.source_record_id, edge_type='OVERLAPS_WITH', edge_id=f'projection:OVERLAPS_WITH:{source.source_record_id}:{target.source_record_id}:temporal_semantic:episode_overlap')
                continue
            gap = _record_frame_ts_gap(left, right)
            if gap is None or not include_near:
                continue
            gap_frames, gap_ts = gap
            if gap_frames <= TEMPORAL_NEAR_MAX_FRAME_GAP or gap_ts <= TEMPORAL_NEAR_MAX_TS_GAP:
                source, target = _ordered_temporal_records(left, right)
                projection.add_link(source_record_id=source.source_record_id, target_record_id=target.source_record_id, edge_type='TEMPORALLY_NEAR', edge_id=f'projection:TEMPORALLY_NEAR:{source.source_record_id}:{target.source_record_id}:temporal_semantic:episode_near')

def _same_known_record_scene(left: _GraphRecordView, right: _GraphRecordView) -> bool:
    if not left.scene_ids or not right.scene_ids:
        return True
    return bool(set(left.scene_ids) & set(right.scene_ids))

def _records_overlap(left: _GraphRecordView, right: _GraphRecordView) -> bool:
    left_start = _record_start_frame(left)
    left_end = _record_end_frame(left)
    right_start = _record_start_frame(right)
    right_end = _record_end_frame(right)
    if left_start is None or left_end is None or right_start is None or (right_end is None):
        return False
    return max(left_start, right_start) <= min(left_end, right_end)

def _record_frame_ts_gap(left: _GraphRecordView, right: _GraphRecordView) -> tuple[int, float] | None:
    left_start = _record_start_frame(left)
    left_end = _record_end_frame(left)
    right_start = _record_start_frame(right)
    right_end = _record_end_frame(right)
    if left_start is None or left_end is None or right_start is None or (right_end is None):
        return None
    if left_end < right_start:
        return (right_start - left_end, _optional_gap(left.end_ts, right.start_ts))
    if right_end < left_start:
        return (left_start - right_end, _optional_gap(right.end_ts, left.start_ts))
    return (0, 0.0)

def _ordered_temporal_records(left: _GraphRecordView, right: _GraphRecordView) -> tuple[_GraphRecordView, _GraphRecordView]:
    left_key = (_sort_frame(_record_start_frame(left)), _sort_frame(_record_end_frame(left)), left.source_record_id)
    right_key = (_sort_frame(_record_start_frame(right)), _sort_frame(_record_end_frame(right)), right.source_record_id)
    return (left, right) if left_key <= right_key else (right, left)

def _record_start_frame(record: _GraphRecordView) -> int | None:
    return _record_time_range_int(record, 'start_frame')

def _record_end_frame(record: _GraphRecordView) -> int | None:
    return _record_time_range_int(record, 'end_frame')

def _record_time_range_int(record: _GraphRecordView, key: str) -> int | None:
    time_range = record.metadata.get('time_range')
    if not isinstance(time_range, dict):
        return None
    value = time_range.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _sort_frame(value: int | None) -> int:
    return value if value is not None else 10 ** 12

def _optional_gap(left_end: float | None, right_start: float | None) -> float:
    if left_end is None or right_start is None:
        return float('inf')
    return max(0.0, right_start - left_end)

def _make_record(*, source_kind: CanonicalSourceKind, source_record_id: str, summary: str, metadata: dict[str, Any], fields: list[str], video_id: str, scene_ids: tuple[str, ...], entity_ids: tuple[str, ...], start_ts: float | None, end_ts: float | None) -> _GraphRecordView:
    index_text = ' '.join((normalized for field in fields for normalized in (_normalize_index_field(field, video_id=video_id),) if normalized))
    return _GraphRecordView(source_kind=source_kind, source_record_id=source_record_id, summary=summary, metadata=metadata, index_text=index_text, tokens=frozenset(_tokenize(index_text)), video_id=video_id, scene_ids=scene_ids, entity_ids=entity_ids, start_ts=start_ts, end_ts=end_ts, priority=PRIORITY_BY_KIND[source_kind])

def _episode_summary(node: Any) -> str:
    if node.episode_kind == 'SCENE_SEGMENT':
        return f'Scene segment {node.scene_id} spans {_format_time_range(node.time_range)}.'
    if node.episode_kind == 'ENTITY_SEGMENT':
        return f"Entity segment for {node.primary_entity_id} in {node.scene_id or 'unknown_scene'} spans {_format_time_range(node.time_range)}."
    if node.episode_kind == 'STATE':
        return f'{node.primary_entity_id} has {node.attr_key}={node.attr_value} from {_format_time_range(node.time_range)}.'
    return f'{node.subject_entity_id} {node.predicate} {node.object_entity_id} from {_format_time_range(node.time_range)}.'

def _relation_observed_burst_record_id(*, scene_id: str | None, subject_entity_id: str, predicate: str, object_entity_id: str) -> str:
    digest = hashlib.md5('||'.join([scene_id or '', subject_entity_id, predicate, object_entity_id]).encode('utf-8'), usedforsecurity=False).hexdigest()[:12]
    return f'episode:relation_observed_burst:{digest}'

def _score_records(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, query_text: str | None=None, query_tokens: set[str] | None=None, query_intent: _QueryIntent | None=None, candidate_record_ids: frozenset[str] | None=None, lexicon: DomainLexicon | None=None, vector_similarity_by_record_id: dict[str, float] | None=None, vector_rank_by_record_id: dict[str, int] | None=None, vector_score_record_ids: set[str] | None=None) -> list[_ScoredRecord]:
    scored_records: list[_ScoredRecord] = []
    query_text = query_text if query_text is not None else (query.query_text or '').strip().lower()
    query_tokens = query_tokens if query_tokens is not None else _tokenize(query_text)
    query_entities = set(query.entity_ids)
    query_scenes = set(query.scene_ids)
    query_intent = query_intent or _infer_query_intent(query_text, query_tokens, lexicon=lexicon or get_domain_lexicon(DEFAULT_DOMAIN_PROFILE))
    ranking_query_tokens = _ranking_query_tokens(query_tokens, query_intent)
    base_delta_intent = _query_wants_base_delta_projection(query_text, query_tokens)
    graph_retrieval_profile = _graph_retrieval_profile_from_query(query)
    vector_similarity_by_record_id = vector_similarity_by_record_id or {}
    vector_rank_by_record_id = vector_rank_by_record_id or {}
    vector_score_record_ids = vector_score_record_ids or set()
    if candidate_record_ids is None:
        records = projection.records.values()
    else:
        records = (projection.records[record_id] for record_id in candidate_record_ids if record_id in projection.records)
    for record in records:
        if query.video_id and record.video_id != query.video_id:
            continue
        if record.source_kind in BASE_DELTA_SOURCE_KINDS and (not base_delta_intent):
            continue
        if record.source_kind == 'temporal_composition':
            if mode != 'temporal' or not query_intent.temporal_composition:
                continue
            if _temporal_composition_is_only_constraint_context(query_intent):
                continue
            if not _temporal_composition_relevant(record, query_intent):
                continue
        lexical_overlap = tuple(sorted(ranking_query_tokens & set(record.tokens)))
        primary_overlap = tuple(sorted(set(_candidate_focus_tokens(query_intent)) & _record_primary_tokens(record)))
        core_overlap = tuple(sorted(set(query_intent.core_terms) & _record_core_tokens(record)))
        evidence_strength = _record_evidence_strength(record, query_intent=query_intent, core_overlap=core_overlap, lexical_overlap=lexical_overlap)
        shared_entities = tuple(sorted(query_entities & set(record.entity_ids)))
        shared_scenes = tuple(sorted(query_scenes & set(record.scene_ids)))
        exact_substring = bool(query_text and query_text in record.index_text)
        query_window_overlap = _query_window_overlap(record, query)
        low_value_relation = _record_is_low_value_relation(record)
        semantic_similarity = vector_similarity_by_record_id.get(record.source_record_id)
        semantic_rank = vector_rank_by_record_id.get(record.source_record_id)
        score = 0.0
        if lexical_overlap:
            score += 1.6 * len(lexical_overlap)
        if primary_overlap:
            score += 2.8 + 0.4 * max(0, len(primary_overlap) - 1)
        if mode == 'temporal' and core_overlap:
            score += 7.0 + 1.2 * max(0, len(core_overlap) - 1)
        if evidence_strength:
            score += evidence_strength
        if exact_substring:
            score += 2.4
        if shared_entities:
            score += 5.2 + 0.5 * max(0, len(shared_entities) - 1)
        if shared_scenes:
            score += 4.2 + 0.5 * max(0, len(shared_scenes) - 1)
        if query_window_overlap:
            score += 3.0
        elif query.start_ts is not None or query.end_ts is not None:
            gap = _query_window_gap(record, query)
            if gap is not None and gap <= 1.0:
                score += max(0.5, 1.5 - gap)
            elif _query_window_should_penalize_far_record(record, query, query_intent=query_intent):
                score -= _query_window_far_record_penalty(record, query)
        if graph_retrieval_profile in CANONICAL_VECTOR_GRAPH_RETRIEVAL_PROFILES and semantic_similarity is not None and (record.source_record_id in vector_score_record_ids):
            rank_score = max(4.5, 18.0 - 0.25 * max(0, semantic_rank - 1)) if semantic_rank is not None else 0.0
            similarity_score = min(3.0, max(0.0, semantic_similarity - 0.2) * 6.0)
            score += rank_score + similarity_score
        if record.source_kind == 'episode':
            episode_kind = str(record.metadata.get('episode_kind') or '')
            if mode == 'temporal' and episode_kind in {'STATE', 'RELATION', RELATION_OBSERVED_BURST_KIND} and lexical_overlap:
                score += 0.8
            if mode == 'entity' and episode_kind == 'ENTITY_SEGMENT' and shared_entities:
                score += 0.5
            if mode == 'temporal' and low_value_relation and (not core_overlap):
                score -= 4.5
            elif mode == 'temporal' and low_value_relation and core_overlap:
                score -= 1.2
        if record.source_kind == 'entity' and mode == 'entity' and shared_entities:
            score += 1.0
        if record.source_kind == 'entity_attribute' and mode == 'entity' and shared_entities:
            score += 0.4
        if record.source_kind == 'stable_identity' and mode == 'entity' and shared_entities:
            score += 1.2
        if record.source_kind == 'transition_event' and lexical_overlap:
            score += 0.6
        if record.source_kind == 'temporal_composition':
            if query_intent.temporal_composition:
                score += 3.0
            if query_intent.duration:
                score += 1.8
            if query_intent.temporal_order:
                score += 1.2
            chain_size = _record_chain_size(record)
            if (query_intent.temporal_order or query_intent.duration) and chain_size >= 3:
                score += 1.0
            transition_overlap = tuple(sorted(_temporal_composition_focus_terms(query_intent) & _record_transition_tokens(record)))
            if transition_overlap:
                score += 6.0 + 0.8 * max(0, len(transition_overlap) - 1)
            boundary_change_score = _record_boundary_change_score(record)
            if boundary_change_score >= TEMPORAL_COMPOSITION_MIN_BOUNDARY_CHANGE:
                score += 2.5 * boundary_change_score
            order_bonus_allowed = bool(transition_overlap or not _candidate_focus_tokens(query_intent))
            if query_intent.earliest:
                start_frame = _record_start_frame(record)
                if start_frame is not None and order_bonus_allowed:
                    score += max(0.0, 8.0 - start_frame / 120.0)
            if query_intent.latest:
                start_frame = _record_start_frame(record)
                if start_frame is not None and order_bonus_allowed:
                    score += min(8.0, start_frame / 120.0)
            if lexical_overlap:
                score += 0.6
        if query_intent.query_family == 'attribute':
            constraint_overlap = _record_constraint_overlap(record, query_intent)
            if constraint_overlap:
                score += 18.0 + 4.0 * max(0, len(constraint_overlap) - 1)
                if record.source_kind == 'entity_attribute':
                    score += 12.0
                elif record.source_kind == 'episode':
                    score += 5.0
            hard_constraint_hits = _record_hard_constraint_group_hits(record, query_intent)
            if hard_constraint_hits:
                score += _hard_constraint_hit_score(hard_constraint_hits, record=record, query_intent=query_intent)
        score += _query_intent_bonus(record, query_intent=query_intent, lexical_overlap=lexical_overlap, primary_overlap=primary_overlap, shared_entities=shared_entities)
        if mode == 'temporal':
            score += _query_family_score_adjustment(record, query_intent=query_intent, core_overlap=core_overlap, lexical_overlap=lexical_overlap)
            score += _attribute_binding_score_adjustment(record, query_intent=query_intent)
        if graph_retrieval_profile in CANONICAL_VECTOR_STRUCTURED_RETRIEVAL_PROFILES and record.source_record_id in vector_score_record_ids:
            score += _canonical_vector_structural_score_adjustment(record, query=query, query_intent=query_intent)
        if score > 0.0:
            score += SEED_KIND_BOOST[mode][record.source_kind]
            scored_records.append(_ScoredRecord(record=record, score=score, lexical_overlap=lexical_overlap, exact_substring=exact_substring, shared_entities=shared_entities, shared_scenes=shared_scenes, query_window_overlap=query_window_overlap, semantic_similarity=semantic_similarity))
    return sorted(scored_records, key=lambda item: (-item.score, item.record.priority, _sort_ts(item.record), item.record.source_record_id))

def _plan_query(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, limit: int, query_text: str, query_tokens: set[str], query_intent: _QueryIntent) -> _QueryPlan:
    global_scope_score = _global_scope_score(query, query_text=query_text, query_tokens=query_tokens, query_intent=query_intent)
    retrieval_budget = _retrieval_budget_for_scope(global_scope_score, limit=limit)
    seed_limit = _seed_limit_for_scope(global_scope_score, limit=limit)
    max_hops = _max_hops_for_scope(global_scope_score, mode=mode)
    scene_seed_ids = _scene_seed_record_ids_for_query(projection, query, mode=mode, query_tokens=query_tokens, query_intent=query_intent, global_scope_score=global_scope_score, retrieval_budget=retrieval_budget)
    candidates = _candidate_record_ids_for_query(projection, query, mode=mode, query_tokens=query_tokens, query_intent=query_intent, retrieval_budget=_auxiliary_temporal_candidate_budget(retrieval_budget) if query_intent.query_family in TEMPORAL_AXIS_QUERY_FAMILIES else retrieval_budget, scene_seed_ids=scene_seed_ids)
    temporal_axis_candidate_ids = _temporal_axis_candidate_record_ids_for_query(projection, query, mode=mode, query_tokens=query_tokens, query_intent=query_intent, scene_seed_ids=scene_seed_ids, retrieval_budget=_temporal_axis_candidate_budget(retrieval_budget, limit=limit, query_intent=query_intent))
    if temporal_axis_candidate_ids:
        if candidates is None:
            candidates = set(temporal_axis_candidate_ids)
        else:
            candidates.update(temporal_axis_candidate_ids)
            max_candidates = max(_temporal_axis_candidate_budget(retrieval_budget, limit=limit, query_intent=query_intent), len(scene_seed_ids), limit * 8)
            if len(candidates) > max_candidates:
                candidates = _trim_candidate_record_ids(projection, candidates, budget=max_candidates, query_intent=query_intent)
    candidate_strategy = 'scene_first_local_seed_index' if scene_seed_ids else 'local_seed_index'
    if temporal_axis_candidate_ids:
        candidate_strategy = f'{candidate_strategy}+temporal_axis'
    if candidates is None:
        candidate_strategy = 'global_scan'
    elif len(candidates) >= len(projection.records):
        candidate_strategy = 'scene_first_global_index_equivalent' if scene_seed_ids else 'global_index_equivalent'
        if temporal_axis_candidate_ids:
            candidate_strategy = f'{candidate_strategy}+temporal_axis'
    return _QueryPlan(global_scope_score=global_scope_score, retrieval_budget=retrieval_budget, seed_limit=seed_limit, max_hops=max_hops, domain_profile=query_intent.domain_profile, candidate_record_ids=frozenset(candidates) if candidates is not None else None, candidate_strategy=candidate_strategy, scene_seed_count=len(scene_seed_ids), query_family=query_intent.query_family, core_terms=query_intent.core_terms, constraint_tokens=query_intent.constraint_tokens, hard_constraint_token_groups=query_intent.hard_constraint_token_groups, compiler_answer_type=query_intent.compiler_answer_type, compiler_retrieval_strategy=query_intent.compiler_retrieval_strategy, compiler_current_window_required=query_intent.compiler_current_window_required, compiler_target_entities=query_intent.compiler_target_entities, compiler_attributes=query_intent.compiler_attributes, compiler_events=query_intent.compiler_events, compiler_temporal_constraints=query_intent.compiler_temporal_constraints, compiler_spatial_constraints=query_intent.compiler_spatial_constraints, compiler_ranking_policy=query_intent.compiler_ranking_policy, compiler_risk_notes=query_intent.compiler_risk_notes)

def _global_scope_score(query: MemoryQuery, *, query_text: str, query_tokens: set[str], query_intent: _QueryIntent) -> int:
    score = 1
    if query.start_ts is not None or query.end_ts is not None or query.scene_ids or query.entity_ids:
        score = 0
    if query_intent.compiler_current_window_required:
        return min(score, 1)
    if query_intent.temporal_composition or query_intent.temporal_order:
        score = max(score, 2)
    if query_intent.duration:
        score = max(score, 3)
    if query_intent.earliest or query_intent.latest:
        score = max(score, 3)
    if bool({'ever', 'any', 'all', 'count', 'closest', 'nearest', 'where'} & query_tokens):
        score = max(score, 3)
    if any((phrase in query_text for phrase in ('did you see', 'have you seen', 'where did you see', 'how many', 'right now'))):
        score = max(score, 3)
    if not _candidate_focus_tokens(query_intent) and (not query.entity_ids) and (not query.scene_ids):
        score = max(score, 4)
    if query.start_ts is not None or query.end_ts is not None:
        score = min(score, _max_scope_for_query_window(query, query_intent=query_intent))
    return min(4, score)

def _max_scope_for_query_window(query: MemoryQuery, *, query_intent: _QueryIntent) -> int:
    if query_intent.compiler_current_window_required:
        return 0
    duration = _query_window_duration(query)
    if duration is not None:
        if duration <= 45.0:
            return 1
        if duration <= 180.0:
            return 2
    if query_intent.query_family in {'position', 'attribute', 'existence', 'ego_direction'}:
        return 2
    return 3

def _query_window_duration(query: MemoryQuery) -> float | None:
    if query.start_ts is None or query.end_ts is None:
        return None
    return max(0.0, query.end_ts - query.start_ts)

def _retrieval_budget_for_scope(global_scope_score: int, *, limit: int) -> int:
    base = max(limit * 8, 40)
    if global_scope_score <= 0:
        return min(base, 80)
    if global_scope_score == 1:
        return min(max(limit * 12, 80), 180)
    if global_scope_score == 2:
        return min(max(limit * 18, 140), 320)
    if global_scope_score == 3:
        return min(max(limit * 30, 260), 700)
    return 0

def _seed_limit_for_scope(global_scope_score: int, *, limit: int) -> int:
    if global_scope_score <= 1:
        return min(max(limit + 2, 6), 10)
    if global_scope_score == 2:
        return min(max(limit * 2, 8), 14)
    if global_scope_score == 3:
        return min(max(limit * 3, 10), 24)
    return min(max(limit * 4, 14), 32)

def _max_hops_for_scope(global_scope_score: int, *, mode: CanonicalRetrievalMode) -> int:
    if global_scope_score <= 1:
        return 1
    if global_scope_score == 2:
        return 2
    if mode == 'temporal':
        return 2
    return 2

def _auxiliary_temporal_candidate_budget(retrieval_budget: int) -> int:
    return min(retrieval_budget, max(24, retrieval_budget // 3))

def _temporal_axis_candidate_budget(retrieval_budget: int, *, limit: int, query_intent: _QueryIntent) -> int:
    if retrieval_budget <= 0:
        return 0
    multiplier = 18 if query_intent.duration else 14
    if query_intent.earliest or query_intent.latest:
        multiplier = max(multiplier, 16)
    budget = max(limit * multiplier, retrieval_budget // 2, 80)
    return min(retrieval_budget, budget)

def _scene_seed_record_ids_for_query(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, query_tokens: set[str], query_intent: _QueryIntent, global_scope_score: int, retrieval_budget: int) -> set[str]:
    if mode != 'temporal' or global_scope_score < 2 or retrieval_budget <= 0:
        return set()
    scene_ids = set(projection.source_kind_index.get('scene', set()))
    if not scene_ids:
        return set()
    focus_tokens = _scene_seed_focus_tokens(query_intent)
    if query.scene_ids:
        explicit = {record_id for scene_id in query.scene_ids for record_id in projection.scene_index.get(scene_id, set()) if projection.records.get(record_id) and projection.records[record_id].source_kind == 'scene'}
        return explicit
    scored: list[tuple[float, str]] = []
    for record_id in scene_ids:
        record = projection.records[record_id]
        lexical_overlap = focus_tokens & set(record.tokens)
        primary_overlap = focus_tokens & _record_primary_tokens(record)
        score = 0.0
        if lexical_overlap:
            score += 3.0 * len(lexical_overlap)
        if primary_overlap:
            score += 4.0 + 0.5 * max(0, len(primary_overlap) - 1)
        if _query_window_overlap(record, query):
            score += 5.0
        elif query.start_ts is not None or query.end_ts is not None:
            gap = _query_window_gap(record, query)
            if gap is not None and gap <= 3.0:
                score += max(0.5, 3.0 - gap)
        if query_intent.duration or query_intent.temporal_order:
            score += _scene_boundary_intent_score(record, query_intent, focus_tokens)
        if query_intent.earliest:
            start_frame = _record_start_frame(record)
            if start_frame is not None:
                score += max(0.0, 4.0 - start_frame / 180.0)
        if query_intent.latest:
            start_frame = _record_start_frame(record)
            if start_frame is not None:
                score += min(4.0, start_frame / 180.0)
        if score <= 0.0:
            continue
        scored.append((score, record_id))
    if not scored:
        return set()
    scene_limit = min(max(8, retrieval_budget // 24), 32)
    ranked = sorted(scored, key=lambda item: (-item[0], _sort_ts(projection.records[item[1]]), item[1]))
    selected = {record_id for _, record_id in ranked[:scene_limit]}
    return _expand_scene_neighbors(projection, selected, max_neighbors=2 if global_scope_score >= 3 else 1, budget=min(len(scene_ids), scene_limit + 12))

def _scene_boundary_intent_score(record: _GraphRecordView, query_intent: _QueryIntent, focus_tokens: set[str]) -> float:
    metadata = record.metadata
    fields: list[Any] = [metadata.get('label'), metadata.get('scene_kind_hint'), *(metadata.get('representative_static_labels') or []), *(metadata.get('representative_dynamic_labels') or []), *(metadata.get('entry_static_labels') or []), *(metadata.get('exit_static_labels') or [])]
    tokens: set[str] = set()
    for field in fields:
        tokens.update(_tokenize(str(field or '')))
    overlap = focus_tokens & tokens
    if not overlap:
        return 0.0
    score = 2.0 + 0.5 * max(0, len(overlap) - 1)
    if query_intent.duration:
        score += 1.0
    if query_intent.temporal_order:
        score += 0.8
    return score

def _scene_seed_focus_tokens(query_intent: _QueryIntent) -> set[str]:
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        return {token for token in set(query_intent.core_terms) | set(query_intent.constraint_tokens) if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    return {token for token in query_intent.focus_tokens if token not in NON_SEMANTIC_QUERY_TOKENS}

def _expand_scene_neighbors(projection: _CanonicalProjection, scene_record_ids: set[str], *, max_neighbors: int, budget: int) -> set[str]:
    expanded = set(scene_record_ids)
    if max_neighbors <= 0:
        return expanded
    ordered_scenes = sorted((record for record_id in projection.source_kind_index.get('scene', set()) for record in [projection.records[record_id]]), key=lambda record: (_sort_ts(record), record.source_record_id))
    index_by_id = {record.source_record_id: index for index, record in enumerate(ordered_scenes)}
    for record_id in sorted(scene_record_ids):
        index = index_by_id.get(record_id)
        if index is None:
            continue
        for offset in range(1, max_neighbors + 1):
            for neighbor_index in (index - offset, index + offset):
                if neighbor_index < 0 or neighbor_index >= len(ordered_scenes):
                    continue
                expanded.add(ordered_scenes[neighbor_index].source_record_id)
                if len(expanded) >= budget:
                    return expanded
    return expanded

def _candidate_record_ids_for_query(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, query_tokens: set[str], query_intent: _QueryIntent, retrieval_budget: int, scene_seed_ids: set[str] | None=None) -> set[str] | None:
    if retrieval_budget <= 0:
        return None
    candidate_ids: set[str] = set()
    for entity_id in query.entity_ids:
        candidate_ids.update(projection.entity_index.get(entity_id, set()))
    for scene_id in query.scene_ids:
        candidate_ids.update(projection.scene_index.get(scene_id, set()))
    if scene_seed_ids:
        candidate_ids.update(scene_seed_ids)
        if query_intent.query_family not in TEMPORAL_AXIS_QUERY_FAMILIES:
            scene_budget = max(retrieval_budget, len(candidate_ids))
            scene_expanded = _expand_candidate_record_ids(projection, candidate_ids, max_hops=1, budget=scene_budget)
            candidate_ids.update(scene_expanded)
    focus_tokens = [token for token in _candidate_focus_tokens(query_intent) if token not in NON_SEMANTIC_QUERY_TOKENS]
    if mode == 'temporal':
        for token in query_intent.core_terms:
            if token not in focus_tokens and token not in NON_SEMANTIC_QUERY_TOKENS:
                focus_tokens.append(token)
    if not focus_tokens and (not candidate_ids):
        return None
    token_hit_counts: Counter[str] = Counter()
    for token in focus_tokens:
        hits = projection.token_index.get(token)
        if not hits:
            continue
        for record_id in hits:
            token_hit_counts[record_id] += 1
    if token_hit_counts:
        preferred_kinds = _preferred_source_kinds_for_query(mode=mode, query_intent=query_intent)
        ranked_ids = sorted(token_hit_counts, key=lambda record_id: (-token_hit_counts[record_id], projection.records[record_id].priority, _candidate_window_rank(projection.records[record_id], query), _sort_ts(projection.records[record_id]), record_id))
        for record_id in ranked_ids:
            record = projection.records[record_id]
            if preferred_kinds and record.source_kind not in preferred_kinds:
                continue
            candidate_ids.add(record_id)
            if len(candidate_ids) >= retrieval_budget:
                break
        if len(candidate_ids) < max(10, retrieval_budget // 3):
            for record_id in ranked_ids:
                candidate_ids.add(record_id)
                if len(candidate_ids) >= retrieval_budget:
                    break
    constraint_hit_counts: Counter[str] = Counter()
    for token in _constraint_candidate_terms(query_intent):
        hits = projection.token_index.get(token)
        if not hits:
            continue
        for record_id in hits:
            constraint_hit_counts[record_id] += 1
    if constraint_hit_counts:
        constraint_budget = max(10, retrieval_budget // 4)
        ranked_constraint_ids = sorted(constraint_hit_counts, key=lambda record_id: (-constraint_hit_counts[record_id], projection.records[record_id].priority, _candidate_window_rank(projection.records[record_id], query), _sort_ts(projection.records[record_id]), record_id))
        for record_id in ranked_constraint_ids[:constraint_budget]:
            candidate_ids.add(record_id)
            if len(candidate_ids) >= retrieval_budget:
                break
    primary_intent_ids = _primary_intent_candidate_record_ids_for_query(projection, query, query_intent=query_intent, budget=max(20, retrieval_budget // 2))
    if primary_intent_ids:
        candidate_ids.update(primary_intent_ids)
    expanded = _expand_candidate_record_ids(projection, candidate_ids, max_hops=1, budget=max(retrieval_budget, len(candidate_ids)))
    return expanded or None

def _primary_intent_candidate_record_ids_for_query(projection: _CanonicalProjection, query: MemoryQuery, *, query_intent: _QueryIntent, budget: int) -> set[str]:
    if query_intent.query_family not in PRIMARY_INTENT_QUERY_FAMILIES or budget <= 0:
        return set()
    target_terms = _primary_target_terms(query_intent)
    if not target_terms:
        return set()
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for token in sorted(target_terms):
        for record_id in projection.token_index.get(token, set()):
            if record_id in seen:
                continue
            record = projection.records.get(record_id)
            if record is None:
                continue
            if query.video_id and record.video_id != query.video_id:
                continue
            if query_intent.query_family == 'attribute' and record.source_kind == 'scene':
                continue
            if record.source_kind == 'temporal_composition':
                continue
            if not (_record_matches_primary_target(record, query_intent) or _record_is_bound_attribute_answer(record, query_intent) or _record_is_bound_attribute_existence_answer(record, query_intent)):
                continue
            if _timed_record(record) and (not _record_in_or_near_query_window(record, query)):
                continue
            role_rank = _primary_answer_role_rank(record, query_intent=query_intent, query=query)
            overlap_count = len(target_terms & _record_core_tokens(record))
            score = 10.0 * role_rank + overlap_count
            if _record_is_bound_attribute_existence_answer(record, query_intent):
                score += 28.0
            constraint_overlap = _record_constraint_overlap(record, query_intent)
            if constraint_overlap:
                score += 20.0 + 4.0 * max(0, len(constraint_overlap) - 1)
                if record.source_kind == 'entity_attribute':
                    score += 8.0
            hard_constraint_hits = _record_hard_constraint_group_hits(record, query_intent)
            if hard_constraint_hits:
                score += _hard_constraint_hit_score(hard_constraint_hits, record=record, query_intent=query_intent)
            if _query_window_overlap(record, query):
                score += 4.0
            scored.append((score, record_id))
            seen.add(record_id)
    if not scored:
        return set()
    scored.sort(key=lambda item: (-item[0], _candidate_window_rank(projection.records[item[1]], query), projection.records[item[1]].priority, _sort_ts(projection.records[item[1]]), item[1]))
    return {record_id for _, record_id in scored[:budget]}

def _candidate_focus_tokens(query_intent: _QueryIntent) -> tuple[str, ...]:
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        terms = set(query_intent.focus_tokens) | set(query_intent.core_terms)
        if not terms:
            terms.update(query_intent.constraint_tokens)
        return tuple(sorted(terms))
    return query_intent.focus_tokens

def _constraint_candidate_terms(query_intent: _QueryIntent) -> tuple[str, ...]:
    terms: set[str] = set(query_intent.constraint_tokens)
    for group in query_intent.hard_constraint_token_groups:
        terms.update(group)
    return tuple(sorted((token for token in terms if token and token not in NON_SEMANTIC_QUERY_TOKENS)))

def _candidate_window_rank(record: _GraphRecordView, query: MemoryQuery) -> int:
    if query.start_ts is None and query.end_ts is None:
        return 0
    if _query_window_overlap(record, query):
        return 0
    gap = _query_window_gap(record, query)
    if gap is not None and gap <= 3.0:
        return 1
    return 2

def _temporal_axis_candidate_record_ids_for_query(projection: _CanonicalProjection, query: MemoryQuery, *, mode: CanonicalRetrievalMode, query_tokens: set[str], query_intent: _QueryIntent, scene_seed_ids: set[str], retrieval_budget: int) -> set[str]:
    if mode != 'temporal' or query_intent.query_family not in TEMPORAL_AXIS_QUERY_FAMILIES or retrieval_budget <= 0:
        return set()
    axis_scene_ids = _temporal_axis_scene_record_ids(projection, query, scene_seed_ids=scene_seed_ids, query_tokens=query_tokens, query_intent=query_intent, budget=min(max(12, retrieval_budget // 8), 80))
    if not axis_scene_ids:
        return set()
    candidate_ids: set[str] = set(axis_scene_ids)
    axis_scene_keys = {scene_id for record_id in axis_scene_ids for scene_id in projection.records[record_id].scene_ids}
    candidate_ids.update(_temporal_axis_scene_detail_record_ids(projection, axis_scene_keys=axis_scene_keys, query_intent=query_intent, budget=max(retrieval_budget - len(candidate_ids), len(axis_scene_ids))))
    for record_id in projection.source_kind_index.get('temporal_composition', set()):
        record = projection.records[record_id]
        if axis_scene_keys & set(record.scene_ids):
            candidate_ids.add(record_id)
    expanded = _expand_candidate_record_ids(projection, candidate_ids, max_hops=1, budget=max(retrieval_budget, len(candidate_ids)))
    candidate_ids.update(expanded)
    return _trim_candidate_record_ids(projection, candidate_ids, budget=max(retrieval_budget, len(axis_scene_ids)), query_intent=query_intent)

def _temporal_axis_scene_detail_record_ids(projection: _CanonicalProjection, *, axis_scene_keys: set[str], query_intent: _QueryIntent, budget: int) -> set[str]:
    if not axis_scene_keys or budget <= 0:
        return set()
    focus_terms = {token for token in set(_candidate_focus_tokens(query_intent)) | set(query_intent.core_terms) if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    ranked: list[tuple[float, str]] = []
    for scene_id in sorted(axis_scene_keys):
        for record_id in projection.scene_index.get(scene_id, set()):
            record = projection.records.get(record_id)
            if record is None:
                continue
            if record.source_kind in {'scene', 'temporal_composition'}:
                if query_intent.duration:
                    score = _duration_temporal_axis_context_score(record, query_intent, focus_terms)
                    if score <= 0.0:
                        continue
                    ranked.append((score, record_id))
                else:
                    ranked.append((100.0, record_id))
                continue
            score = _temporal_axis_detail_score(record, query_intent, focus_terms)
            if score <= 0.0:
                continue
            ranked.append((score, record_id))
    if not ranked:
        return set()
    ranked.sort(key=lambda item: (-item[0], _temporal_axis_source_kind_priority(projection.records[item[1]].source_kind, query_intent=query_intent), _sort_ts(projection.records[item[1]]), item[1]))
    return {record_id for _, record_id in ranked[:budget]}

def _temporal_axis_detail_score(record: _GraphRecordView, query_intent: _QueryIntent, focus_terms: set[str]) -> float:
    tokens = _record_core_tokens(record)
    overlap = focus_terms & tokens
    episode_kind = str(record.metadata.get('episode_kind') or '').upper()
    event_type = str(record.metadata.get('event_type') or '').upper()
    score = 0.0
    if overlap:
        score += 5.0 + 0.5 * max(0, len(overlap) - 1)
    if record.source_kind == 'transition_event':
        score += 2.0
        if query_intent.appearance and event_type == 'APPEAR':
            score += 3.0
        if query_intent.disappearance and event_type == 'DISAPPEAR':
            score += 3.0
        if query_intent.compiler_events and overlap:
            score += 1.5
    elif record.source_kind == 'episode':
        if episode_kind in {'STATE', 'ENTITY_SEGMENT'}:
            score += 1.2
        elif episode_kind in {'RELATION', RELATION_OBSERVED_BURST_KIND}:
            score += 0.5 if overlap else -1.0
    elif record.source_kind == 'scene_delta':
        score += 1.0
    elif record.source_kind in {'entity', 'stable_identity'}:
        score += 0.8 if overlap else -0.5
    elif record.source_kind in {'state_value', 'interaction_type'}:
        score -= 1.0
    if query_intent.duration and record.source_kind in {'transition_event', 'episode'}:
        score += 0.8
    if query_intent.temporal_order and record.source_kind == 'transition_event':
        score += 0.8
    return score

def _duration_temporal_axis_context_score(record: _GraphRecordView, query_intent: _QueryIntent, focus_terms: set[str]) -> float:
    tokens = _record_core_tokens(record) | _record_transition_tokens(record)
    overlap = focus_terms & tokens
    if not overlap:
        return 0.0
    score = 1.0 + 0.25 * max(0, len(overlap) - 1)
    if record.source_kind == 'temporal_composition':
        boundary_change = _record_boundary_change_score(record)
        if boundary_change > 0.0:
            score += min(2.0, boundary_change)
    return score

def _temporal_axis_scene_record_ids(projection: _CanonicalProjection, query: MemoryQuery, *, scene_seed_ids: set[str], query_tokens: set[str], query_intent: _QueryIntent, budget: int) -> set[str]:
    if scene_seed_ids:
        max_neighbors = 1 if query_intent.duration else 2
        return _expand_scene_neighbors(projection, scene_seed_ids, max_neighbors=max_neighbors, budget=budget)
    if query.scene_ids:
        explicit = {record_id for scene_id in query.scene_ids for record_id in projection.scene_index.get(scene_id, set()) if projection.records.get(record_id) and projection.records[record_id].source_kind == 'scene'}
        return _expand_scene_neighbors(projection, explicit, max_neighbors=2, budget=budget)
    if query.start_ts is not None or query.end_ts is not None:
        window_scene_ids = {record_id for record_id in projection.source_kind_index.get('scene', set()) if _query_window_overlap(projection.records[record_id], query) or ((gap := _query_window_gap(projection.records[record_id], query)) is not None and gap <= 5.0)}
        return _expand_scene_neighbors(projection, window_scene_ids, max_neighbors=2, budget=budget)
    if query_intent.temporal_composition or query_intent.temporal_order:
        return _temporal_boundary_scene_record_ids(projection, query_tokens=query_tokens, query_intent=query_intent, budget=budget)
    return set()

def _temporal_boundary_scene_record_ids(projection: _CanonicalProjection, *, query_tokens: set[str], query_intent: _QueryIntent, budget: int) -> set[str]:
    scored: list[tuple[float, str]] = []
    focus_tokens = set(_candidate_focus_tokens(query_intent)) | set(query_intent.core_terms)
    for record_id in projection.source_kind_index.get('temporal_composition', set()):
        record = projection.records[record_id]
        transition_tokens = _record_transition_tokens(record)
        overlap = focus_tokens & (transition_tokens | _record_core_tokens(record))
        score = 0.0
        if overlap:
            score += 3.0 + 0.4 * max(0, len(overlap) - 1)
        boundary_change = _record_boundary_change_score(record)
        if boundary_change >= TEMPORAL_COMPOSITION_MIN_BOUNDARY_CHANGE:
            score += 1.5 * boundary_change
        if query_intent.earliest:
            start_frame = _record_start_frame(record)
            if start_frame is not None:
                score += max(0.0, 3.0 - start_frame / 240.0)
        if query_intent.latest:
            start_frame = _record_start_frame(record)
            if start_frame is not None:
                score += min(3.0, start_frame / 240.0)
        if score <= 0.0:
            continue
        scored.append((score, record_id))
    if not scored:
        return set()
    ranked = sorted(scored, key=lambda item: (-item[0], _sort_ts(projection.records[item[1]]), item[1]))
    scene_ids = {scene_record_id for _, record_id in ranked[:max(4, budget // 4)] for scene_id in projection.records[record_id].scene_ids for scene_record_id in projection.scene_index.get(scene_id, set()) if projection.records.get(scene_record_id) and projection.records[scene_record_id].source_kind == 'scene'}
    return _expand_scene_neighbors(projection, scene_ids, max_neighbors=1, budget=budget)

def _trim_candidate_record_ids(projection: _CanonicalProjection, candidate_ids: set[str], *, budget: int, query_intent: _QueryIntent | None=None) -> set[str]:
    if len(candidate_ids) <= budget:
        return candidate_ids
    ranked = sorted(candidate_ids, key=lambda record_id: (_temporal_axis_source_kind_priority(projection.records[record_id].source_kind, query_intent=query_intent), _sort_ts(projection.records[record_id]), record_id))
    return set(ranked[:budget])

def _temporal_axis_source_kind_priority(source_kind: CanonicalSourceKind, *, query_intent: _QueryIntent | None) -> int:
    priorities = TEMPORAL_AXIS_DURATION_SOURCE_KIND_PRIORITY if query_intent is not None and query_intent.duration else TEMPORAL_AXIS_SOURCE_KIND_PRIORITY
    return priorities.get(source_kind, 99)

def _preferred_source_kinds_for_query(*, mode: CanonicalRetrievalMode, query_intent: _QueryIntent) -> set[CanonicalSourceKind]:
    if mode == 'entity':
        return {'entity', 'entity_attribute', 'stable_identity', 'episode', 'transition_event', 'scene_delta'}
    if query_intent.query_family in {'position', 'attribute', 'existence'}:
        return {'entity', 'entity_attribute', 'stable_identity', 'episode', 'scene', 'transition_event'}
    if query_intent.temporal_composition:
        return {'temporal_composition', 'episode', 'transition_event', 'scene', 'scene_delta'}
    if query_intent.appearance or query_intent.disappearance:
        return {'transition_event', 'episode', 'entity', 'entity_attribute', 'stable_identity'}
    return {'episode', 'transition_event', 'scene', 'entity', 'entity_attribute', 'stable_identity', 'temporal_composition'}

def _expand_candidate_record_ids(projection: _CanonicalProjection, seed_ids: set[str], *, max_hops: int, budget: int) -> set[str]:
    if not seed_ids:
        return set()
    expanded = set(seed_ids)
    frontier = set(seed_ids)
    for _ in range(max_hops):
        next_frontier: set[str] = set()
        for record_id in sorted(frontier):
            for link in projection.adjacency.get(record_id, []):
                if link.neighbor_id in expanded:
                    continue
                expanded.add(link.neighbor_id)
                next_frontier.add(link.neighbor_id)
                if len(expanded) >= budget:
                    return expanded
        frontier = next_frontier
        if not frontier:
            break
    return expanded

def _build_fragments(projection: _CanonicalProjection, *, score_lookup: dict[str, _ScoredRecord], seed_records: list[_ScoredRecord], query: MemoryQuery, mode: CanonicalRetrievalMode, limit: int, max_hops: int | None=None, lexicon: DomainLexicon | None=None) -> list[_GraphFragment]:
    lexicon = lexicon or get_domain_lexicon(DEFAULT_DOMAIN_PROFILE)
    query_text = str(query.query_text or '').lower()
    query_intent = _infer_query_intent(query_text, _tokenize(query_text), lexicon=lexicon, answer_type=query.answer_type)
    fragments: dict[tuple[str, ...], _GraphFragment] = {}
    per_seed_limit = max(5, min(limit + 2, 8))
    resolved_max_hops = max_hops if max_hops is not None else 2
    for seed in seed_records:
        self_score = seed.score + RESULT_KIND_BOOST[mode][seed.record.source_kind]
        self_fragment = _GraphFragment(seed_record_id=seed.record.source_record_id, record_ids=(seed.record.source_record_id,), score=self_score, relation_types=('seed_match',))
        fragments[self_fragment.record_ids] = self_fragment
        related_paths = _collect_related_paths(projection, seed.record.source_record_id, max_hops=resolved_max_hops, mode=mode)
        neighbor_fragments: list[_GraphFragment] = []
        for record_id, path in related_paths.items():
            if record_id == seed.record.source_record_id:
                continue
            neighbor = projection.records[record_id]
            relation = _relation_between(seed.record, neighbor, path=path, query=query, mode=mode, lexicon=lexicon)
            if relation.score <= 0.0:
                continue
            neighbor_score = score_lookup.get(record_id)
            fragment_score = seed.score * 1.15 + (neighbor_score.score if neighbor_score else 0.0) * 0.6 + relation.score + RESULT_KIND_BOOST[mode][neighbor.source_kind]
            neighbor_fragments.append(_GraphFragment(seed_record_id=seed.record.source_record_id, record_ids=tuple(sorted((seed.record.source_record_id, neighbor.source_record_id))), score=fragment_score, relation_types=relation.relation_types, shared_entities=relation.shared_entities, shared_scenes=relation.shared_scenes, path_record_ids=relation.path_record_ids, path_edge_types=relation.path_edge_types, hop_count=relation.hop_count))
        neighbor_fragments.sort(key=lambda item: (-item.score, item.hop_count, len(item.record_ids), item.record_ids))
        for fragment in neighbor_fragments[:per_seed_limit]:
            existing = fragments.get(fragment.record_ids)
            if existing is None or fragment.score > existing.score:
                fragments[fragment.record_ids] = fragment
        for fragment in _attribute_constraint_neighbor_fragments(projection, seed=seed, score_lookup=score_lookup, query_intent=query_intent, mode=mode, limit=limit):
            existing = fragments.get(fragment.record_ids)
            if existing is None or fragment.score > existing.score:
                fragments[fragment.record_ids] = fragment
    return sorted(fragments.values(), key=lambda item: (-item.score, item.hop_count, len(item.record_ids), item.record_ids))

def _attribute_constraint_neighbor_fragments(projection: _CanonicalProjection, *, seed: _ScoredRecord, score_lookup: dict[str, _ScoredRecord], query_intent: _QueryIntent, mode: CanonicalRetrievalMode, limit: int) -> list[_GraphFragment]:
    if mode != 'temporal' or query_intent.query_family not in {'attribute', 'existence'}:
        return []
    if not _record_constraint_overlap(seed.record, query_intent):
        return []
    attribute_key_terms, attribute_value_terms, target_terms = _attribute_binding_terms(query_intent)
    if not (attribute_key_terms or attribute_value_terms) or not target_terms:
        return []
    component_target_terms = _attribute_component_target_terms(target_terms)
    scored: list[tuple[float, _GraphRecordView, tuple[str, ...], tuple[str, ...]]] = []
    for record_id in _nearby_attribute_candidate_ids(projection, seed.record):
        if record_id == seed.record.source_record_id:
            continue
        candidate = projection.records.get(record_id)
        if candidate is None or candidate.source_kind != 'entity_attribute':
            continue
        if not _record_matches_attribute_request(candidate, attribute_key_terms, attribute_value_terms):
            continue
        candidate_tokens = _record_core_tokens(candidate)
        target_overlap = tuple(sorted(candidate_tokens & target_terms))
        if not target_overlap:
            continue
        component_overlap = tuple(sorted(candidate_tokens & component_target_terms))
        if component_target_terms and (not component_overlap):
            continue
        temporal_gap = _record_gap(seed.record, candidate)
        if temporal_gap > 5.0:
            continue
        score = seed.score * 1.1 + 18.0 + RESULT_KIND_BOOST[mode][candidate.source_kind] + 2.0 * max(0, len(target_overlap) - 1) + 6.0 * len(component_overlap) + max(0.0, 5.0 - temporal_gap)
        candidate_score = score_lookup.get(candidate.source_record_id)
        if candidate_score is not None:
            score += candidate_score.score * 0.4
        scored.append((score, candidate, target_overlap, component_overlap))
    scored.sort(key=lambda item: (-item[0], _record_gap(seed.record, item[1]), item[1].priority, _sort_ts(item[1]), item[1].source_record_id))
    fragments: list[_GraphFragment] = []
    for score, candidate, target_overlap, component_overlap in scored[:max(2, min(limit, 6))]:
        record_ids = tuple(sorted((seed.record.source_record_id, candidate.source_record_id)))
        shared_scenes = tuple(sorted(set(seed.record.scene_ids) & set(candidate.scene_ids)))
        relation_types = ['attribute_event_constraint_nearby', 'shared_scene']
        if _record_gap(seed.record, candidate) == 0.0:
            relation_types.append('temporal_overlap')
        else:
            relation_types.append('temporal_near')
        displayed_targets = component_overlap or target_overlap
        relation_types.extend((f'target:{token}' for token in displayed_targets[:3]))
        fragments.append(_GraphFragment(seed_record_id=seed.record.source_record_id, record_ids=record_ids, score=score, relation_types=tuple(dict.fromkeys(relation_types)), shared_scenes=shared_scenes, path_record_ids=(seed.record.source_record_id, candidate.source_record_id), path_edge_types=('ATTRIBUTE_NEAR_CONSTRAINED_EVENT',), hop_count=1))
    return fragments

def _attribute_component_target_terms(target_terms: set[str]) -> set[str]:
    component_terms = {token for token in target_terms if token not in ATTRIBUTE_CARRIER_TARGET_TOKENS}
    return component_terms or set(target_terms)

def _nearby_attribute_candidate_ids(projection: _CanonicalProjection, seed: _GraphRecordView) -> set[str]:
    candidate_ids: set[str] = set()
    for scene_id in seed.scene_ids:
        candidate_ids.update(projection.scene_index.get(scene_id, set()))
    for entity_id in seed.entity_ids:
        candidate_ids.update(projection.entity_index.get(entity_id, set()))
    return candidate_ids

def _collect_related_paths(projection: _CanonicalProjection, seed_record_id: str, *, max_hops: int, mode: CanonicalRetrievalMode) -> dict[str, _GraphPath]:
    best_paths: dict[str, _GraphPath] = {}

    def walk(current_record_id: str, *, edge_types: tuple[str, ...], record_ids: tuple[str, ...], visited: set[str]) -> None:
        if len(edge_types) >= max_hops:
            return
        for link in projection.adjacency.get(current_record_id, []):
            if link.neighbor_id in visited:
                continue
            next_record_ids = record_ids + (link.neighbor_id,)
            next_edge_types = edge_types + (link.edge_type,)
            path = _GraphPath(record_ids=next_record_ids, edge_types=next_edge_types)
            existing = best_paths.get(link.neighbor_id)
            if existing is None or _path_rank(path, mode=mode) > _path_rank(existing, mode=mode):
                best_paths[link.neighbor_id] = path
            walk(link.neighbor_id, edge_types=next_edge_types, record_ids=next_record_ids, visited=visited | {link.neighbor_id})
    walk(seed_record_id, edge_types=(), record_ids=(seed_record_id,), visited={seed_record_id})
    return best_paths

def _path_rank(path: _GraphPath, *, mode: CanonicalRetrievalMode) -> float:
    weight = sum((EDGE_TYPE_WEIGHT[mode].get(edge_type, 0.5) for edge_type in path.edge_types))
    return weight - 0.7 * max(0, path.hop_count - 1)

def _relation_between(seed: _GraphRecordView, neighbor: _GraphRecordView, *, path: _GraphPath, query: MemoryQuery, mode: CanonicalRetrievalMode, lexicon: DomainLexicon | None=None) -> _RelationSignal:
    lexicon = lexicon or get_domain_lexicon(DEFAULT_DOMAIN_PROFILE)
    if (seed.source_kind in BASE_DELTA_SOURCE_KINDS or neighbor.source_kind in BASE_DELTA_SOURCE_KINDS) and (not _query_wants_base_delta_projection(str(query.query_text or '').lower(), _tokenize(str(query.query_text or '').lower()))):
        return _RelationSignal(score=0.0, relation_types=())
    if seed.source_kind == 'temporal_composition' or neighbor.source_kind == 'temporal_composition':
        query_text = str(query.query_text or '').lower()
        query_intent = _infer_query_intent(query_text, _tokenize(query_text), lexicon=lexicon)
        if mode != 'temporal' or not query_intent.temporal_composition:
            return _RelationSignal(score=0.0, relation_types=())
    score = 0.0
    relation_types: list[str] = list(path.edge_types)
    shared_entities = tuple(sorted(set(seed.entity_ids) & set(neighbor.entity_ids)))
    shared_scenes = tuple(sorted(set(seed.scene_ids) & set(neighbor.scene_ids)))
    if path.hop_count == 1:
        score += 2.4
    elif path.hop_count == 2:
        score += 1.5
    else:
        score += 0.8
    for edge_type in path.edge_types:
        score += EDGE_TYPE_WEIGHT[mode].get(edge_type, 0.5)
    if shared_entities:
        score += 2.8 + 0.3 * max(0, len(shared_entities) - 1)
        relation_types.append('shared_entity')
    if shared_scenes:
        score += 2.1 + 0.2 * max(0, len(shared_scenes) - 1)
        relation_types.append('shared_scene')
    temporal_gap = _record_gap(seed, neighbor)
    if temporal_gap == 0.0:
        score += 1.4
        relation_types.append('temporal_overlap')
    elif temporal_gap <= 1.0:
        score += max(0.3, 1.0 - temporal_gap)
        relation_types.append('temporal_near')
    if query.entity_ids and shared_entities and set(shared_entities) & set(query.entity_ids):
        score += 1.4
        relation_types.append('query_entity_focus')
    if query.scene_ids and shared_scenes and set(shared_scenes) & set(query.scene_ids):
        score += 1.0
        relation_types.append('query_scene_focus')
    if _query_window_overlap(neighbor, query):
        score += 0.8
        relation_types.append('query_window_overlap')
    if mode == 'temporal' and any((edge in {'NEXT_EPISODE', 'PREVIOUS_EPISODE'} for edge in path.edge_types)):
        score += 0.8
    if mode == 'entity' and shared_entities:
        score += 0.5
    return _RelationSignal(score=score, relation_types=tuple(dict.fromkeys(relation_types)), shared_entities=shared_entities, shared_scenes=shared_scenes, path_record_ids=path.record_ids, path_edge_types=path.edge_types, hop_count=path.hop_count)

def _rank_evidence_items(projection: _CanonicalProjection, *, score_lookup: dict[str, _ScoredRecord], fragments: list[_GraphFragment], limit: int, mode: CanonicalRetrievalMode, query_intent: _QueryIntent, query: MemoryQuery) -> list[dict[str, Any]]:
    best_items: dict[str, dict[str, Any]] = {}
    for fragment in fragments:
        for record_id in fragment.record_ids:
            record = projection.records[record_id]
            seed_bonus = 0.3 if record_id == fragment.seed_record_id else 0.0
            query_score = score_lookup.get(record_id)
            item_score = fragment.score + seed_bonus + RESULT_KIND_BOOST[mode][record.source_kind]
            if query_score is not None:
                item_score += query_score.score * 0.1
            item_score += _final_evidence_score_adjustment(record, query_intent=query_intent, mode=mode, query=query)
            item_score += _primary_role_score_adjustment(record, query_intent=query_intent, query=query)
            if _graph_retrieval_profile_from_query(query) in CANONICAL_VECTOR_STRUCTURED_RETRIEVAL_PROFILES and query_score is not None and (query_score.semantic_similarity is not None):
                item_score += _canonical_vector_structural_score_adjustment(record, query=query, query_intent=query_intent)
            support = {'seed_record_id': fragment.seed_record_id, 'record_ids': list(fragment.record_ids), 'relation_types': list(fragment.relation_types), 'shared_entities': list(fragment.shared_entities), 'shared_scenes': list(fragment.shared_scenes), 'fragment_score': round(fragment.score, 4), 'path_record_ids': list(fragment.path_record_ids), 'path_edge_types': list(fragment.path_edge_types), 'hop_count': fragment.hop_count}
            candidate = {'source_kind': record.source_kind, 'source_record_id': record.source_record_id, 'summary': record.summary, 'score': round(item_score, 4), 'metadata': {**record.metadata, 'hybrid_support': support}, 'sort': {'answer_role_rank': _primary_answer_role_rank(record, query_intent=query_intent, query=query), 'priority': record.priority, 'start_ts': _sort_ts(record)}}
            existing = best_items.get(record_id)
            if existing is None or candidate['score'] > existing['score']:
                best_items[record_id] = candidate
    ranked_items = sorted(best_items.values(), key=lambda item: (-int(item['sort'].get('answer_role_rank') or 0), -float(item['score'] or 0.0), item['sort']['priority'], item['sort']['start_ts'], item['source_record_id']))
    return [{'source_kind': item['source_kind'], 'source_record_id': item['source_record_id'], 'summary': item['summary'], 'score': item['score'], 'metadata': item['metadata']} for item in ranked_items[:limit]]

def _supplemental_context_items(projection: _CanonicalProjection, *, score_lookup: dict[str, _ScoredRecord], primary_items: list[dict[str, Any]], primary_item_ids: list[str], query_intent: _QueryIntent, query: MemoryQuery, mode: CanonicalRetrievalMode, limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not primary_items:
        return []
    wanted_terms = _supplemental_wanted_terms(query_intent)
    seed_ids = {record_id for record_id in primary_item_ids if record_id in projection.records}
    if not seed_ids:
        return []
    primary_set = set(seed_ids)
    primary_entity_ids = _primary_anchor_entity_ids(projection, primary_set, query_intent=query_intent, query=query)
    primary_scene_ids = _primary_scene_ids(projection, primary_set)
    candidate_ids: set[str] = set()
    for seed_id in seed_ids:
        seed = projection.records[seed_id]
        for entity_id in seed.entity_ids:
            candidate_ids.update(projection.entity_index.get(entity_id, set()))
        for scene_id in seed.scene_ids:
            candidate_ids.update(projection.scene_index.get(scene_id, set()))
        for link in projection.adjacency.get(seed_id, []):
            candidate_ids.add(link.neighbor_id)
    for token in wanted_terms:
        candidate_ids.update(projection.token_index.get(token, set()))
    scored: list[tuple[float, _GraphRecordView, tuple[str, ...]]] = []
    for record_id in candidate_ids:
        if record_id in primary_set:
            continue
        record = projection.records.get(record_id)
        if record is None:
            continue
        if query.video_id and record.video_id != query.video_id:
            continue
        if record.source_kind in BASE_DELTA_SOURCE_KINDS:
            continue
        if record.source_kind == 'temporal_composition' and query_intent.query_family not in TEMPORAL_AXIS_QUERY_FAMILIES:
            continue
        if _query_window_should_penalize_far_record(record, query, query_intent=query_intent):
            continue
        target_overlap = tuple(sorted(wanted_terms & set(_record_core_tokens(record))))
        shared_entities = tuple(sorted(set(record.entity_ids) & primary_entity_ids))
        shared_scenes = tuple(sorted(set(record.scene_ids) & primary_scene_ids))
        primary_target_match = _record_matches_primary_target(record, query_intent)
        if query_intent.query_family in {'attribute', 'existence'} and (not shared_entities) and (not primary_target_match) and (record.source_kind != 'scene'):
            continue
        if not target_overlap and (not shared_entities) and (not shared_scenes):
            continue
        score = _supplemental_context_score(record, score_lookup=score_lookup, target_overlap=target_overlap, shared_entities=shared_entities, shared_scenes=shared_scenes, query_intent=query_intent, query=query, mode=mode)
        if score <= 0.0:
            continue
        reasons = _supplemental_context_reasons(target_overlap=target_overlap, shared_entities=shared_entities, shared_scenes=shared_scenes, record=record, query_intent=query_intent)
        scored.append((score, record, reasons))
    scored.sort(key=lambda item: (-item[0], -_primary_answer_role_rank(item[1], query_intent=query_intent, query=query), item[1].priority, _sort_ts(item[1]), item[1].source_record_id))
    max_supplemental = max(2, min(limit, 8))
    return [_record_to_evidence_item(record, score=round(score, 4), support={'supplemental_context': True, 'relation_types': list(reasons), 'shared_entities': list(sorted(set(record.entity_ids) & primary_entity_ids)), 'shared_scenes': list(sorted(set(record.scene_ids) & primary_scene_ids)), 'target_overlap': sorted(wanted_terms & set(_record_core_tokens(record)))}) for score, record, reasons in scored[:max_supplemental]]

def _record_to_evidence_item(record: _GraphRecordView, *, score: float, support: dict[str, Any]) -> dict[str, Any]:
    return {'source_kind': record.source_kind, 'source_record_id': record.source_record_id, 'summary': record.summary, 'score': score, 'metadata': {**record.metadata, 'hybrid_support': support}}

def _supplemental_wanted_terms(query_intent: _QueryIntent) -> set[str]:
    terms: set[str] = set()
    for value in (*query_intent.core_terms, *query_intent.focus_tokens, *query_intent.constraint_tokens, *query_intent.compiler_target_entities, *query_intent.compiler_attributes, *query_intent.compiler_events, *query_intent.compiler_temporal_constraints, *query_intent.compiler_spatial_constraints):
        for token in _semantic_tokens(str(value)):
            if token and token not in ATTRIBUTE_QUERY_HINTS:
                terms.add(token)
    if query_intent.query_family in {'existence', 'temporal', 'duration'}:
        for token in query_intent.core_terms:
            if token and token not in NON_SEMANTIC_QUERY_TOKENS:
                terms.add(token)
    return terms

def _primary_anchor_entity_ids(projection: _CanonicalProjection, primary_record_ids: set[str], *, query_intent: _QueryIntent, query: MemoryQuery) -> set[str]:
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        anchored_entity_ids: set[str] = set()
        for record_id in primary_record_ids:
            record = projection.records.get(record_id)
            if record is None or record.source_kind == 'scene':
                continue
            role_rank = _primary_answer_role_rank(record, query_intent=query_intent, query=query)
            if role_rank >= 2 or _record_matches_primary_target(record, query_intent):
                anchored_entity_ids.update(record.entity_ids)
        if anchored_entity_ids:
            return anchored_entity_ids
    entity_ids: set[str] = set()
    for record_id in primary_record_ids:
        record = projection.records.get(record_id)
        if record is not None:
            entity_ids.update(record.entity_ids)
    return entity_ids

def _primary_scene_ids(projection: _CanonicalProjection, primary_record_ids: set[str]) -> set[str]:
    scene_ids: set[str] = set()
    for record_id in primary_record_ids:
        record = projection.records.get(record_id)
        if record is not None:
            scene_ids.update(record.scene_ids)
    return scene_ids

def _supplemental_context_score(record: _GraphRecordView, *, score_lookup: dict[str, _ScoredRecord], target_overlap: tuple[str, ...], shared_entities: tuple[str, ...], shared_scenes: tuple[str, ...], query_intent: _QueryIntent, query: MemoryQuery, mode: CanonicalRetrievalMode) -> float:
    score = RESULT_KIND_BOOST[mode][record.source_kind]
    score += 5.0 * len(target_overlap)
    score += 4.0 * len(shared_entities)
    score += 2.0 * len(shared_scenes)
    query_score = score_lookup.get(record.source_record_id)
    if query_score is not None:
        score += query_score.score * 0.35
    score += _final_evidence_score_adjustment(record, query_intent=query_intent, mode=mode, query=query)
    score += _primary_role_score_adjustment(record, query_intent=query_intent, query=query)
    if record.source_kind in {'entity_attribute', 'episode', 'transition_event'}:
        score += 4.0
    elif record.source_kind in {'entity', 'stable_identity'}:
        score += 2.0
    elif record.source_kind == 'scene':
        score -= 2.0
    return score

def _supplemental_context_reasons(*, target_overlap: tuple[str, ...], shared_entities: tuple[str, ...], shared_scenes: tuple[str, ...], record: _GraphRecordView, query_intent: _QueryIntent) -> tuple[str, ...]:
    reasons: list[str] = ['supplemental_context']
    if target_overlap:
        reasons.append('matches_question_terms')
    if shared_entities:
        reasons.append('same_entity')
    if shared_scenes:
        reasons.append('same_scene')
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        reasons.append('primary_intent_context')
    if record.source_kind in {'entity_attribute', 'episode', 'transition_event'}:
        reasons.append('direct_fact_record')
    return tuple(dict.fromkeys(reasons))

def _primary_answer_role_rank(record: _GraphRecordView, *, query_intent: _QueryIntent, query: MemoryQuery) -> int:
    if not _record_in_or_near_query_window(record, query):
        return 0
    graph_retrieval_profile = _graph_retrieval_profile_from_query(query)
    unified_hm3d_profile = False
    if unified_hm3d_profile:
        binding = 0
        binding_rank = int(binding.get('answer_role_rank') or 0)
        if binding_rank:
            return binding_rank
    if query_intent.query_family == 'attribute':
        if _record_is_bound_attribute_answer(record, query_intent):
            return 4
        if _record_matches_primary_target(record, query_intent):
            if record.source_kind == 'entity_attribute':
                return 4
            if record.source_kind == 'state_value':
                return 3
            if record.source_kind == 'episode':
                return 3
            if record.source_kind in {'entity', 'stable_identity'}:
                return 2
        if record.source_kind == 'scene':
            return 1
        return 0
    if query_intent.query_family == 'existence':
        if _record_is_bound_attribute_existence_answer(record, query_intent):
            return 4
        if _record_matches_primary_target(record, query_intent):
            if record.source_kind == 'entity_attribute':
                return 4
            if record.source_kind in {'episode', 'entity', 'stable_identity'}:
                return 3
            if record.source_kind == 'scene':
                return 1
    if query_intent.query_family == 'position':
        if _record_matches_primary_target(record, query_intent):
            if record.source_kind in {'entity', 'stable_identity'}:
                return 3
            if record.source_kind == 'episode':
                return 2
            if record.source_kind == 'scene':
                return 1
    return 0

def _primary_role_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent, query: MemoryQuery) -> float:
    if query_intent.query_family not in PRIMARY_INTENT_QUERY_FAMILIES:
        return 0.0
    role_rank = _primary_answer_role_rank(record, query_intent=query_intent, query=query)
    if query_intent.query_family == 'attribute':
        if role_rank >= 4:
            return 32.0
        if role_rank == 3:
            return 18.0
        if role_rank == 2:
            return 12.0
        if record.source_kind == 'scene':
            return -36.0
        if record.source_kind in {'interaction_type', 'temporal_composition'}:
            return -24.0
        return -8.0
    if query_intent.query_family == 'existence':
        if _record_is_bound_attribute_existence_answer(record, query_intent):
            return 32.0
        if role_rank >= 3:
            return 18.0
        if record.source_kind == 'scene':
            return -8.0
    if query_intent.query_family == 'position':
        if role_rank >= 3:
            return 16.0
        if role_rank == 2:
            return 10.0
        if record.source_kind == 'scene':
            return -4.0
    return 0.0

def _record_in_or_near_query_window(record: _GraphRecordView, query: MemoryQuery) -> bool:
    if query.start_ts is None and query.end_ts is None:
        return True
    if _query_window_overlap(record, query):
        return True
    gap = _query_window_gap(record, query)
    return gap is not None and gap <= 3.0

def _final_evidence_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent, mode: CanonicalRetrievalMode, query: MemoryQuery) -> float:
    score = 0.0
    if _query_window_should_penalize_far_record(record, query, query_intent=query_intent):
        score -= _query_window_far_record_penalty(record, query) * 2.0
    if query_intent.query_family == 'attribute':
        _, _, target_terms = _attribute_binding_terms(query_intent)
        if not target_terms:
            target_terms = {token for token in query_intent.core_terms if token not in ATTRIBUTE_TARGET_STOP_TOKENS}
        record_tokens = _record_core_tokens(record)
        target_match = bool(record_tokens & target_terms)
        if target_match:
            if record.source_kind == 'entity_attribute':
                score += 34.0
            elif record.source_kind in {'entity', 'stable_identity'}:
                score += 4.0
            elif record.source_kind == 'episode':
                episode_kind = str(record.metadata.get('episode_kind') or '').upper()
                if episode_kind == 'STATE':
                    score += 36.0
                elif episode_kind == 'ENTITY_SEGMENT':
                    score += 6.0
                elif episode_kind in {'RELATION', RELATION_OBSERVED_BURST_KIND}:
                    score += 8.0
            elif record.source_kind == 'state_value':
                score += 14.0
            elif record.source_kind == 'scene':
                score += 0.0
        elif record.source_kind in {'stable_identity', 'interaction_type'}:
            score -= 14.0
        elif record.source_kind == 'episode':
            episode_kind = str(record.metadata.get('episode_kind') or '').upper()
            if episode_kind in {'STATE', 'RELATION', RELATION_OBSERVED_BURST_KIND}:
                score -= 8.0
        elif record.source_kind == 'scene':
            score -= 4.0
    if mode == 'temporal' and record.source_kind == 'interaction_type' and _record_is_low_value_relation(record):
        score -= 10.0
    elif mode == 'temporal' and record.source_kind == 'episode' and _record_is_low_value_relation(record) and (not set(query_intent.core_terms) & _record_primary_tokens(record)):
        score -= 4.0
    elif mode == 'temporal' and record.source_kind == 'interaction_type':
        score -= 6.0
    semantic_overlap = tuple((token for token in query_intent.core_terms if token not in NON_SEMANTIC_QUERY_TOKENS and token in _record_semantic_label_tokens(record)))
    if semantic_overlap:
        semantic_overlap_count = len(set(semantic_overlap))
        if record.source_kind == 'scene':
            score += 9.0
            if query_intent.query_family != 'attribute':
                score += 3.0 * max(0, semantic_overlap_count - 1)
        elif record.source_kind == 'entity' and query_intent.query_family == 'existence':
            score += 8.0 + 2.0 * max(0, semantic_overlap_count - 1)
        elif record.source_kind == 'temporal_composition':
            score += 4.0 + 1.5 * max(0, semantic_overlap_count - 1)
    score += _ranking_policy_score_adjustment(record, query_intent=query_intent, query=query)
    score += _primary_intent_evidence_score_adjustment(record, query_intent=query_intent)
    return score

def _query_window_should_penalize_far_record(record: _GraphRecordView, query: MemoryQuery, *, query_intent: _QueryIntent) -> bool:
    if query.start_ts is None and query.end_ts is None:
        return False
    if not _timed_record(record) or _query_window_overlap(record, query):
        return False
    if query_intent.compiler_current_window_required:
        return True
    if query_intent.query_family in TEMPORAL_AXIS_QUERY_FAMILIES:
        return True
    if query_intent.query_family in {'ego_direction', 'existence'}:
        return True
    return False

def _query_window_far_record_penalty(record: _GraphRecordView, query: MemoryQuery) -> float:
    gap = _query_window_gap(record, query)
    if gap is None:
        return 0.0
    if gap <= 1.0:
        return 0.0
    return min(32.0, 6.0 + gap / 5.0)

def _primary_intent_evidence_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent) -> float:
    family = query_intent.query_family
    if family == 'attribute':
        if _record_is_bound_attribute_answer(record, query_intent):
            return 14.0
        if record.source_kind == 'entity_attribute':
            return 8.0
        if record.source_kind == 'temporal_composition':
            return -18.0
        if record.source_kind == 'scene':
            return -5.0
    elif family == 'position':
        if query_intent.compiler_answer_type != 'position' and (not query_intent.compiler_target_entities):
            return 0.0
        if _record_matches_primary_target(record, query_intent):
            if record.source_kind in {'entity', 'stable_identity'}:
                return 8.0
            if record.source_kind == 'scene':
                return 4.0
            if record.source_kind == 'episode':
                return 3.0
        if record.source_kind == 'temporal_composition':
            return -10.0
    elif family == 'existence':
        if _record_is_bound_attribute_existence_answer(record, query_intent):
            return 18.0
        if _record_matches_primary_target(record, query_intent):
            if record.source_kind in {'entity', 'stable_identity'}:
                return 8.0
            if record.source_kind == 'entity_attribute':
                return 8.0
            if record.source_kind == 'episode':
                return 5.0
            if record.source_kind == 'scene':
                return 2.0
        if record.source_kind == 'temporal_composition':
            return -8.0
    return 0.0

def _ranking_policy_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent, query: MemoryQuery) -> float:
    score = 0.0
    if 'earliest_event' in query_intent.compiler_ranking_policy:
        score += _earliest_within_query_window_score(record, query=query)
        if query_intent.query_family in TEMPORAL_AXIS_QUERY_FAMILIES:
            if record.source_kind == 'temporal_composition':
                if _record_starts_within_query_window(record, query=query):
                    score += 7.0
                elif _record_overlaps_query_window(record, query=query):
                    score += 1.0
            elif record.source_kind == 'scene':
                score -= 2.0
    if 'distance_or_recency_proxy' in query_intent.compiler_ranking_policy:
        score += _closest_proxy_score(record, query_intent=query_intent, query=query)
    return score

def _earliest_within_query_window_score(record: _GraphRecordView, *, query: MemoryQuery) -> float:
    if record.start_ts is None:
        return 0.0
    if query.start_ts is not None and record.end_ts is not None and (record.end_ts < query.start_ts):
        return -16.0
    if query.end_ts is not None and record.start_ts > query.end_ts:
        return -16.0
    if query.start_ts is not None and record.start_ts < query.start_ts:
        return -6.0
    if query.start_ts is None or query.end_ts is None:
        start_frame = _record_start_frame(record)
        return max(0.0, 6.0 - (start_frame or 0) / 160.0)
    window = max(1e-06, query.end_ts - query.start_ts)
    position = min(1.0, max(0.0, (record.start_ts - query.start_ts) / window))
    return 8.0 * (1.0 - position)

def _record_starts_within_query_window(record: _GraphRecordView, *, query: MemoryQuery) -> bool:
    if record.start_ts is None:
        return False
    tolerance = 1e-06
    if query.start_ts is not None and record.start_ts < query.start_ts - tolerance:
        return False
    if query.end_ts is not None and record.start_ts > query.end_ts + tolerance:
        return False
    return True

def _record_overlaps_query_window(record: _GraphRecordView, *, query: MemoryQuery) -> bool:
    if record.start_ts is None or record.end_ts is None:
        return False
    if query.start_ts is not None and record.end_ts < query.start_ts:
        return False
    if query.end_ts is not None and record.start_ts > query.end_ts:
        return False
    return True

def _closest_proxy_score(record: _GraphRecordView, *, query_intent: _QueryIntent, query: MemoryQuery) -> float:
    score = 0.0
    tokens = _record_core_tokens(record)
    target_terms = {token for token in set(query_intent.core_terms) | set(_candidate_focus_tokens(query_intent)) if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    if tokens & target_terms:
        if record.source_kind in {'entity', 'stable_identity'}:
            score += 5.0
        elif record.source_kind == 'episode':
            score += 3.5
        elif record.source_kind == 'scene':
            score += 2.0
    if query.end_ts is not None and record.start_ts is not None:
        gap = abs(record.start_ts - query.end_ts)
        score += max(0.0, 4.0 - min(gap / 30.0, 4.0))
    return score

def _direct_semantic_overlap(record: _GraphRecordView, overlap_terms: tuple[str, ...]) -> bool:
    if not overlap_terms:
        return False
    return bool(set(overlap_terms) & _record_semantic_label_tokens(record))

def _record_semantic_label_tokens(record: _GraphRecordView) -> frozenset[str]:
    metadata = record.metadata
    fields: list[Any] = []
    if record.source_kind == 'entity':
        fields.extend([metadata.get('label'), metadata.get('canonical_label'), metadata.get('semantic_group'), *(metadata.get('semantic_alias_terms') or [])])
    elif record.source_kind == 'scene':
        fields.extend([metadata.get('label'), metadata.get('scene_kind_hint'), *(metadata.get('representative_static_labels') or []), *(metadata.get('representative_dynamic_labels') or []), *(metadata.get('salient_labels') or []), *(metadata.get('semantic_alias_terms') or [])])
    elif record.source_kind == 'temporal_composition':
        for scene in metadata.get('scene_sequence') or []:
            if not isinstance(scene, dict):
                continue
            fields.extend([scene.get('label'), scene.get('scene_kind_hint'), *(scene.get('representative_static_labels') or []), *(scene.get('representative_dynamic_labels') or []), *(scene.get('salient_labels') or []), *(scene.get('semantic_alias_terms') or [])])
    tokens: set[str] = set()
    for field in fields:
        normalized = _normalize_index_field(field, video_id=record.video_id)
        if normalized:
            tokens.update(_tokenize(normalized))
    return frozenset(tokens)

def _promote_temporal_axis_seed_records(seed_records: list[_ScoredRecord], *, scored_records: list[_ScoredRecord], query_intent: _QueryIntent, seed_limit: int) -> list[_ScoredRecord]:
    if query_intent.query_family not in TEMPORAL_AXIS_QUERY_FAMILIES:
        return seed_records
    if seed_limit <= 0:
        return seed_records
    temporal_axis = [item for item in scored_records if item.record.source_kind in {'temporal_composition', 'scene'}]
    if not temporal_axis:
        return seed_records
    keep_count = min(3 if query_intent.duration else 2, seed_limit, len(temporal_axis))
    promoted = temporal_axis[:keep_count]
    seen = {item.record.source_record_id for item in promoted}
    merged = promoted + [item for item in seed_records if item.record.source_record_id not in seen]
    return merged[:seed_limit]

def _serialize_seed(item: _ScoredRecord) -> dict[str, Any]:
    payload = {'source_record_id': item.record.source_record_id, 'source_kind': item.record.source_kind, 'score': round(item.score, 4), 'lexical_overlap': list(item.lexical_overlap), 'exact_substring': item.exact_substring, 'shared_entities': list(item.shared_entities), 'shared_scenes': list(item.shared_scenes), 'query_window_overlap': item.query_window_overlap}
    if item.semantic_similarity is not None:
        payload['semantic_similarity'] = round(item.semantic_similarity, 6)
    return payload

def _serialize_vector_recall(index: CanonicalRecordVectorIndex | None, hits: list[VectorRecallHit], *, query: str, added_candidate_ids: set[str]) -> dict[str, Any]:
    if index is None:
        return {'enabled': False, 'hits': []}
    return {'enabled': True, 'model_name': index.model_name, 'cache_status': index.cache_status, 'query': query, 'hit_count': len(hits), 'hits': [{'record_id': hit.record_id, 'source_kind': hit.source_kind, 'similarity': round(hit.similarity, 6), 'rank': hit.rank, 'added_to_lexical_candidates': hit.record_id in added_candidate_ids} for hit in hits]}

def _canonical_vector_supplemental_items(projection: _CanonicalProjection, *, query: MemoryQuery, mode: CanonicalRetrievalMode, limit: int, max_hops: int, query_intent: _QueryIntent, lexicon: DomainLexicon, vector_scored_records: list[_ScoredRecord], existing_items: list[dict[str, Any]], max_items: int=1) -> tuple[list[dict[str, Any]], list[_GraphFragment]]:
    if not vector_scored_records or limit <= 0 or max_items <= 0:
        return ([], [])
    vector_seed_records = vector_scored_records[:min(max(4, max_items * 3), len(vector_scored_records))]
    vector_score_lookup = {item.record.source_record_id: item for item in vector_scored_records}
    fragments = _build_fragments(projection, score_lookup=vector_score_lookup, seed_records=vector_seed_records, query=query, mode=mode, limit=min(max(3, max_items * 2), limit), max_hops=max_hops, lexicon=lexicon)
    candidates = _rank_evidence_items(projection, score_lookup=vector_score_lookup, fragments=fragments, limit=min(max(3, max_items * 2), limit), mode=mode, query_intent=query_intent, query=query)
    existing_ids = {str(item.get('source_record_id') or '') for item in existing_items if str(item.get('source_record_id') or '')}
    supplements: list[dict[str, Any]] = []
    for item in candidates:
        record_id = str(item.get('source_record_id') or '')
        if not record_id or record_id in existing_ids:
            continue
        updated = dict(item)
        updated['metadata'] = {**dict(item.get('metadata') or {}), 'retrieval_channel': 'canonical_vector_supplement'}
        supplements.append(updated)
        existing_ids.add(record_id)
        if len(supplements) >= max_items:
            break
    return (supplements, fragments[:6])

def _serialize_fragment(fragment: _GraphFragment) -> dict[str, Any]:
    return {'seed_record_id': fragment.seed_record_id, 'record_ids': list(fragment.record_ids), 'score': round(fragment.score, 4), 'relation_types': list(fragment.relation_types), 'shared_entities': list(fragment.shared_entities), 'shared_scenes': list(fragment.shared_scenes), 'path_record_ids': list(fragment.path_record_ids), 'path_edge_types': list(fragment.path_edge_types), 'hop_count': fragment.hop_count}

def _serialize_query_plan(plan: _QueryPlan) -> dict[str, Any]:
    candidate_count = None
    if plan.candidate_record_ids is not None:
        candidate_count = len(plan.candidate_record_ids)
    return {'global_scope_score': plan.global_scope_score, 'retrieval_budget': plan.retrieval_budget, 'seed_limit': plan.seed_limit, 'max_hops': plan.max_hops, 'candidate_strategy': plan.candidate_strategy, 'candidate_count': candidate_count, 'global_fallback': plan.global_fallback, 'scene_seed_count': plan.scene_seed_count, 'domain_profile': plan.domain_profile, 'query_family': plan.query_family, 'core_terms': list(plan.core_terms), 'constraint_tokens': list(plan.constraint_tokens), 'hard_constraint_token_groups': [list(group) for group in plan.hard_constraint_token_groups], 'question_compiler': {'answer_type': plan.compiler_answer_type, 'retrieval_strategy': plan.compiler_retrieval_strategy, 'current_window_required': plan.compiler_current_window_required, 'target_entities': list(plan.compiler_target_entities), 'attributes': list(plan.compiler_attributes), 'events': list(plan.compiler_events), 'temporal_constraints': list(plan.compiler_temporal_constraints), 'spatial_constraints': list(plan.compiler_spatial_constraints), 'ranking_policy': list(plan.compiler_ranking_policy), 'risk_notes': list(plan.compiler_risk_notes)}}

def _compiler_family_to_retrieval_family(family: str, *, answer_type: str) -> str:
    if answer_type in {'text', 'binary', 'time', 'duration'} and family == 'position_target':
        return 'general'
    if family == 'position_target':
        return 'position'
    if family == 'temporal_event':
        return 'temporal'
    if family == 'ego_motion':
        return 'ego_direction'
    if family == 'current_state':
        return 'existence'
    if family in {'attribute', 'duration', 'existence'}:
        return family
    if answer_type == 'position':
        return 'position'
    if answer_type == 'time':
        return 'temporal'
    return 'general'

def _merge_compiler_terms(current_terms: tuple[str, ...], compiler_terms: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    for token in (*current_terms, *compiler_terms):
        normalized = str(token or '').strip().lower()
        if normalized and normalized not in NON_SEMANTIC_QUERY_TOKENS and (normalized not in merged):
            merged.append(normalized)
    return tuple(sorted(merged))

def _compiler_focus_tokens(plan: Any) -> tuple[str, ...]:
    tokens: list[str] = []
    for group in (plan.target_entities, plan.target_aliases, plan.attributes, plan.events, plan.temporal_constraints, plan.spatial_constraints):
        for value in group:
            for token in _tokenize(str(value)):
                if token and token not in NON_SEMANTIC_QUERY_TOKENS and (token not in tokens):
                    tokens.append(token)
    return tuple(sorted(tokens))

def _constraint_tokens_for_plan(plan: Any, *, normalized_query_text: str, query_tokens: set[str]) -> tuple[str, ...]:
    tokens: set[str] = set()
    for group in (plan.events, plan.temporal_constraints, plan.spatial_constraints):
        for value in group:
            tokens.update(_semantic_tokens(str(value)))
    for match in CONSTRAINT_CLAUSE_PATTERN.finditer(normalized_query_text):
        tokens.update(_semantic_tokens(match.group(0)))
    for match in REFERENCE_EVENT_CLAUSE_PATTERN.finditer(normalized_query_text):
        tokens.update(_semantic_tokens(match.group(0)))
    return tuple(sorted(tokens))

def _semantic_tokens(text: str) -> set[str]:
    return {token for token in _tokenize(text) if token and token not in NON_SEMANTIC_QUERY_TOKENS}

def _compiler_primary_terms_for_family(plan: Any, *, query_family: str) -> tuple[str, ...]:
    if query_family == 'attribute':
        groups = (_attribute_primary_targets(plan.target_entities), plan.target_aliases, plan.attributes)
    elif query_family in {'position', 'existence'}:
        groups = (plan.target_entities, plan.target_aliases, plan.attributes)
    else:
        groups = (plan.target_entities, plan.target_aliases, plan.attributes, plan.events)
    tokens: list[str] = []
    for group in groups:
        for value in group:
            for token in _tokenize(str(value)):
                if query_family == 'attribute' and token in ATTRIBUTE_CONTEXT_TARGET_TOKENS:
                    continue
                if token and token not in NON_SEMANTIC_QUERY_TOKENS and (token not in tokens):
                    tokens.append(token)
    if query_family not in {'attribute', 'position', 'existence'}:
        for token in plan.normalized_query_terms:
            if token and token not in NON_SEMANTIC_QUERY_TOKENS and (token not in tokens):
                tokens.append(token)
    return tuple(tokens)

def _primary_focus_tokens_for_query(*, normalized_query_text: str, query_tokens: set[str], compiler_plan: Any, query_family: str, constraint_tokens: tuple[str, ...], lexicon: DomainLexicon) -> tuple[str, ...]:
    if query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        tokens = set(_compiler_primary_terms_for_family(compiler_plan, query_family=query_family))
        if query_family == 'attribute':
            tokens.update(_attribute_primary_terms_from_text(normalized_query_text))
        elif query_family == 'existence':
            tokens.update(_existence_primary_terms_from_text(normalized_query_text))
        elif query_family == 'position':
            tokens.update(_position_primary_terms_from_text(normalized_query_text))
        if not tokens:
            tokens.update((token for token in query_tokens if token not in constraint_tokens and token not in NON_SEMANTIC_QUERY_TOKENS))
    else:
        tokens = set((token for token in query_tokens if token not in NON_SEMANTIC_QUERY_TOKENS))
        tokens.update(_compiler_primary_terms_for_family(compiler_plan, query_family=query_family))
    return tuple(sorted(_expand_core_aliases(tokens, lexicon=lexicon)))

def _expand_core_aliases(tokens: set[str], *, lexicon: DomainLexicon) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        normalized = str(token or '').strip().lower()
        if not normalized or normalized in NON_SEMANTIC_QUERY_TOKENS:
            continue
        expanded.add(normalized)
        expanded.update(lexicon.core_term_aliases.get(normalized, ()))
    return {token for token in expanded if token and token not in NON_SEMANTIC_QUERY_TOKENS}

def _attribute_primary_terms_from_text(normalized_query_text: str) -> set[str]:
    terms: set[str] = set()
    match = re.search('\\b(?:color|colour|appearance)\\s+(?:of\\s+)?(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    match = re.search('\\b(?:what\\s+)?(?:color|colour)\\s+(?:were|was|is|are)\\s+(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    match = re.search('\\bwhat\\s+(?:were|was|is|are)\\s+(?:the\\s+)?(?:color|colour|appearance)\\s+of\\s+(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    return {token for token in terms if token not in ATTRIBUTE_TARGET_STOP_TOKENS}

def _existence_primary_terms_from_text(normalized_query_text: str) -> set[str]:
    terms: set[str] = set()
    match = re.search('\\b(?:see|saw|seen|observe|observed|notice|noticed)\\s+(?:any\\s+)?(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    return {token for token in terms if token not in {'any', 'ever'} | set(POSITION_QUERY_HINTS)}

def _position_primary_terms_from_text(normalized_query_text: str) -> set[str]:
    terms: set[str] = set()
    match = re.search('\\bwhere\\s+(?:is|are|was|were)\\s+(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    match = re.search('\\b(?:closest|nearest)\\s+(?:the\\s+)?([a-z0-9_\\s-]+?)(?=\\s+(?:near|around|by|where|when|after|before|during|while|with|in\\s+front\\s+of)\\b|[?.;,]|$)', normalized_query_text)
    if match:
        terms.update(_semantic_tokens(match.group(1)))
    return terms

def _attribute_primary_targets(target_entities: tuple[str, ...]) -> tuple[str, ...]:
    primary: list[str] = []
    for target in target_entities:
        tokens = set(_tokenize(str(target)))
        if tokens and tokens.issubset(ATTRIBUTE_CONTEXT_TARGET_TOKENS):
            continue
        primary.append(target)
    return tuple(primary)

def _normalize_index_field(field: Any, *, video_id: str) -> str:
    text = str(field or '').strip().lower()
    if not text:
        return ''
    video_id_text = video_id.strip().lower()
    if video_id_text:
        text = text.replace(f':{video_id_text}:', ':')
        if text.startswith(f'{video_id_text}:'):
            text = text[len(video_id_text) + 1:]
        text = text.replace(f'{video_id_text}:', '')
        if text == video_id_text:
            return ''
    text = re.sub(':{2,}', ':', text).strip(' :')
    return text

def _infer_query_intent(query_text: str, query_tokens: set[str], *, lexicon: DomainLexicon | None=None, answer_type: str | None=None) -> _QueryIntent:
    lexicon = lexicon or get_domain_lexicon(DEFAULT_DOMAIN_PROFILE)
    normalized = ' '.join(query_text.strip().lower().split())
    if not normalized:
        return _QueryIntent(domain_profile=lexicon.name)
    compiler_plan = compile_question(normalized, answer_type=answer_type, domain_profile=lexicon.name)
    appearance = any((hint in normalized for hint in APPEARANCE_QUERY_HINTS))
    disappearance = any((hint in normalized for hint in DISAPPEARANCE_QUERY_HINTS))
    duration = bool(DURATION_QUERY_HINTS & query_tokens) or any((phrase in normalized for phrase in ('how long', 'how much time', 'for how long')))
    duration = duration or compiler_plan.answer_type == 'duration'
    temporal_order = bool(TEMPORAL_ORDER_QUERY_HINTS & query_tokens)
    earliest = bool({'first', 'start', 'started', 'begin', 'began'} & query_tokens)
    latest = bool({'last', 'leave', 'leaving', 'left', 'exit', 'exited', 'end', 'ended'} & query_tokens)
    temporal_composition = duration or temporal_order or compiler_plan.query_family in {'temporal_event', 'duration', 'ego_motion', 'current_state'} or bool(TEMPORAL_COMPOSITION_QUERY_HINTS & query_tokens) or any((phrase in normalized for phrase in TEMPORAL_COMPOSITION_QUERY_PHRASE_HINTS))
    query_family = _infer_query_family(normalized, query_tokens, duration=duration, temporal_composition=temporal_composition)
    compiler_family = _compiler_family_to_retrieval_family(compiler_plan.query_family, answer_type=compiler_plan.answer_type)
    if compiler_family != 'general':
        query_family = compiler_family
    elif answer_type and compiler_plan.answer_type != 'position' and (query_family == 'position'):
        query_family = 'general'
    constraint_tokens = _constraint_tokens_for_plan(compiler_plan, normalized_query_text=normalized, query_tokens=query_tokens)
    if query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        focus_tokens = _primary_focus_tokens_for_query(normalized_query_text=normalized, query_tokens=query_tokens, compiler_plan=compiler_plan, query_family=query_family, constraint_tokens=constraint_tokens, lexicon=lexicon)
        core_terms = tuple(sorted(_core_terms_for_query(normalized, focus_tokens, lexicon=lexicon)))
    else:
        focus_tokens = tuple(sorted((token for token in query_tokens if token not in NON_SEMANTIC_QUERY_TOKENS)))
        compiler_focus = _compiler_focus_tokens(compiler_plan)
        focus_tokens = _merge_compiler_terms(focus_tokens, compiler_focus)
        core_terms = tuple(sorted(_core_terms_for_query(normalized, focus_tokens, lexicon=lexicon)))
        core_terms = _merge_compiler_terms(core_terms, _compiler_primary_terms_for_family(compiler_plan, query_family=query_family))
    if query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        core_terms = tuple(sorted((token for token in core_terms if token not in constraint_tokens)))
    constraint_tokens = tuple(sorted((token for token in constraint_tokens if token not in set(core_terms))))
    return _QueryIntent(domain_profile=lexicon.name, appearance=appearance, disappearance=disappearance, temporal_composition=temporal_composition, duration=duration, temporal_order=temporal_order, earliest=earliest or 'earliest' in compiler_plan.temporal_constraints, latest=latest, query_family=query_family, core_terms=core_terms, focus_tokens=focus_tokens, constraint_tokens=constraint_tokens, hard_constraint_token_groups=(), compiler_answer_type=compiler_plan.answer_type, compiler_retrieval_strategy=compiler_plan.retrieval_strategy, compiler_current_window_required=compiler_plan.current_window_required, compiler_target_entities=compiler_plan.target_entities, compiler_attributes=compiler_plan.attributes, compiler_events=compiler_plan.events, compiler_temporal_constraints=compiler_plan.temporal_constraints, compiler_spatial_constraints=compiler_plan.spatial_constraints, compiler_ranking_policy=compiler_plan.ranking_policy, compiler_preferred_source_kinds=compiler_plan.preferred_source_kinds, compiler_risk_notes=compiler_plan.risk_notes)

def unified_graph_retrieval_question_ir(ir: dict[str, Any] | None, *, graph_retrieval_profile: str) -> dict[str, Any] | None:
    if not isinstance(ir, dict):
        return None
    if graph_retrieval_profile not in UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES:
        return ir
    return semantic_question_ir_projection(ir)

def _question_ir_from_query(query: MemoryQuery) -> dict[str, Any] | None:
    metadata = getattr(query, 'metadata', {}) or {}
    if not isinstance(metadata, dict):
        return None
    ir = metadata.get('question_ir')
    return unified_graph_retrieval_question_ir(ir if isinstance(ir, dict) else None, graph_retrieval_profile=str(metadata.get('graph_retrieval_profile') or '').strip())

def _question_route_from_query(query: MemoryQuery) -> dict[str, Any] | None:
    metadata = getattr(query, 'metadata', {}) or {}
    if not isinstance(metadata, dict):
        return None
    if str(metadata.get('graph_retrieval_profile') or '').strip() in UNIFIED_TEMPORAL_GRAPH_RETRIEVAL_PROFILES:
        return None
    route = metadata.get('question_route')
    if isinstance(route, dict):
        return route
    ir = metadata.get('question_ir')
    if isinstance(ir, dict) and isinstance(ir.get('question_route'), dict):
        return ir.get('question_route')
    return None

def _generic_event_terms(query_intent: _QueryIntent) -> set[str]:
    if query_intent.duration:
        return set()
    terms: set[str] = set()
    for value in query_intent.compiler_events:
        for token in _tokenize(str(value)):
            if token and token not in NON_SEMANTIC_QUERY_TOKENS:
                terms.add(token)
    return _generic_expand_surface_terms(terms)

def _generic_label_target_overlap(record: _GraphRecordView, target_terms: set[str]) -> set[str]:
    if not target_terms:
        return set()
    fields: list[Any] = [record.metadata.get('label'), record.metadata.get('primary_entity_label'), record.metadata.get('subject_entity_label'), record.metadata.get('object_entity_label'), record.metadata.get('canonical_label'), record.metadata.get('semantic_group'), record.metadata.get('stable_identity_canonical_label')]
    tokens: set[str] = set()
    for field in fields:
        if field:
            tokens.update(_tokenize(str(field)))
    for value in record.metadata.get('semantic_alias_terms') or []:
        tokens.update(_tokenize(str(value)))
    return _generic_expand_surface_terms(tokens) & target_terms

def _generic_expand_surface_terms(terms: set[str]) -> set[str]:
    expanded = set(terms)
    for term in terms:
        expanded.update(_english_surface_variants(term))
    return {token for token in expanded if token and token not in NON_SEMANTIC_QUERY_TOKENS}

def _english_surface_variants(token: str) -> set[str]:
    normalized = str(token or '').strip().lower()
    if len(normalized) <= 3:
        return set()
    variants: set[str] = set()
    if normalized.endswith('ies') and len(normalized) > 4:
        variants.add(f'{normalized[:-3]}y')
    if normalized.endswith('ches') or normalized.endswith('shes'):
        variants.add(normalized[:-2])
    elif normalized.endswith('xes') or normalized.endswith('ses') or normalized.endswith('zes'):
        variants.add(normalized[:-2])
    elif normalized.endswith('s') and (not normalized.endswith('ss')):
        variants.add(normalized[:-1])
    if normalized.endswith('e'):
        variants.add(f'{normalized[:-1]}ing')
        variants.add(f'{normalized}d')
    else:
        variants.add(f'{normalized}ing')
        variants.add(f'{normalized}ed')
    return variants

def _apply_question_ir_to_intent(intent: _QueryIntent, ir: dict[str, Any] | None, *, query_text: str, lexicon: DomainLexicon) -> _QueryIntent:
    if not ir:
        return intent
    operation = str(ir.get('operation') or '').strip().lower()
    focus = str(ir.get('qa_focus') or '').strip().lower()
    answer_type = str(ir.get('answer_type') or intent.compiler_answer_type or '').strip().lower()
    query_family = _query_family_from_question_ir(operation=operation, focus=focus, answer_type=answer_type, fallback=intent.query_family)
    question_route = ir.get('question_route') if isinstance(ir.get('question_route'), dict) else {}
    query_family = _query_family_from_question_route(question_route, fallback=query_family)
    target_entities = tuple(_clean_ir_strings(ir.get('target_entities')))
    attributes = tuple(_question_ir_attributes(ir, query_text=query_text))
    events = tuple(_question_ir_events(operation=operation, ir=ir))
    temporal_constraints = tuple(_question_ir_temporal_constraints(ir, operation=operation))
    spatial_constraints = tuple(_question_ir_spatial_constraints(ir))
    focus_tokens = _question_ir_focus_tokens(ir, query_family=query_family, lexicon=lexicon, fallback_text=query_text)
    core_terms = tuple(sorted(_core_terms_for_query(query_text, focus_tokens, lexicon=lexicon)))
    if not core_terms:
        core_terms = intent.core_terms
    hard_constraint_token_groups = _question_ir_hard_constraint_token_groups(ir)
    hard_constraint_tokens = {token for group in hard_constraint_token_groups for token in group}
    constraint_tokens = tuple(sorted((token for token in _question_ir_constraint_tokens(ir) | hard_constraint_tokens if token not in set(core_terms))))
    temporal_relation = str(ir.get('temporal_relation') or '').strip().lower()
    duration = query_family == 'duration' or operation == 'interval'
    temporal_order = temporal_relation in {'first', 'latest', 'last', 'before', 'after'}
    earliest = operation == 'onset' or temporal_relation == 'first'
    latest = operation == 'offset' or temporal_relation in {'latest', 'last'}
    temporal_composition = query_family in {'temporal', 'duration', 'ego_direction'} or intent.temporal_composition
    current_required = operation == 'current_state' or temporal_relation == 'current'
    return _QueryIntent(domain_profile=intent.domain_profile, appearance=intent.appearance or 'appearance' in attributes, disappearance=intent.disappearance, temporal_composition=temporal_composition, duration=duration, temporal_order=temporal_order or intent.temporal_order, earliest=earliest or intent.earliest, latest=latest or intent.latest, query_family=query_family, core_terms=core_terms, focus_tokens=focus_tokens, constraint_tokens=constraint_tokens, hard_constraint_token_groups=hard_constraint_token_groups, compiler_answer_type=answer_type or intent.compiler_answer_type, compiler_retrieval_strategy=_retrieval_strategy_from_question_ir(operation=operation, query_family=query_family, fallback=intent.compiler_retrieval_strategy), compiler_current_window_required=current_required or intent.compiler_current_window_required, compiler_target_entities=target_entities or intent.compiler_target_entities, compiler_attributes=attributes or intent.compiler_attributes, compiler_events=events or intent.compiler_events, compiler_temporal_constraints=temporal_constraints or intent.compiler_temporal_constraints, compiler_spatial_constraints=spatial_constraints or intent.compiler_spatial_constraints, compiler_ranking_policy=intent.compiler_ranking_policy, compiler_preferred_source_kinds=intent.compiler_preferred_source_kinds, compiler_risk_notes=tuple(dict.fromkeys((*intent.compiler_risk_notes, 'question_ir_experimental_override'))))

def _query_family_from_question_ir(*, operation: str, focus: str, answer_type: str, fallback: str) -> str:
    if operation in {'onset', 'offset', 'occurrence'}:
        return 'temporal'
    if operation == 'interval' or answer_type == 'duration':
        return 'duration'
    if operation == 'attribute_query':
        return 'attribute'
    if operation == 'existence_query':
        return 'existence'
    if operation == 'current_state':
        return 'existence'
    if operation in {'spatial_relation', 'navigation_direction'}:
        return 'ego_direction'
    if operation == 'position_query' or answer_type == 'position':
        return 'position'
    if focus == 'descriptive':
        return 'attribute'
    if focus in {'spatial_geometry', 'navigation_direction'}:
        return 'ego_direction'
    return fallback

def _query_family_from_question_route(question_route: dict[str, Any] | None, *, fallback: str) -> str:
    if not isinstance(question_route, dict):
        return fallback
    route_id = str(question_route.get('route_id') or '').strip()
    route_to_family = {'temporal_interval': 'duration', 'temporal_point': 'temporal', 'entity_attribute': 'attribute', 'ocr_logo_text': 'attribute', 'entity_relation_spatial': 'position', 'count_set': 'existence', 'binary_existence': 'existence', 'ego_motion_side_turn': 'ego_direction'}
    return route_to_family.get(route_id, fallback)

def _retrieval_strategy_from_question_ir(*, operation: str, query_family: str, fallback: str) -> str:
    if operation in {'onset', 'offset', 'occurrence'}:
        return 'temporal_axis_then_target_filter'
    if operation == 'interval':
        return 'temporal_axis_interval'
    if query_family == 'attribute':
        return 'target_entity_attribute_binding'
    if query_family == 'existence':
        return 'question_window_scene_first'
    if query_family == 'ego_direction':
        return 'ego_event_or_scene_transition'
    if query_family == 'position':
        return 'position_target'
    return fallback

def _question_ir_attributes(ir: dict[str, Any], *, query_text: str='') -> list[str]:
    attributes = _clean_ir_strings(ir.get('attribute_dimension'))
    target_state = str(ir.get('target_state') or '').strip().lower()
    if target_state:
        if 'busy' in target_state and 'busy' not in attributes:
            attributes.append('busy')
        if 'empty' in target_state and 'empty' not in attributes:
            attributes.append('empty')
        for color in COLOR_TOKENS:
            if color in _tokenize(target_state) and color not in attributes:
                attributes.append(color)
    if 'color' in attributes or 'colour' in attributes or 'colored' in attributes:
        color_source = ' '.join([str(query_text or ''), target_state, ' '.join(_clean_ir_strings(ir.get('target_entities')))])
        color_tokens = _tokenize(color_source) & COLOR_TOKENS
        for color in sorted(color_tokens):
            if color not in attributes:
                attributes.append(color)
    return attributes

def _question_ir_events(*, operation: str, ir: dict[str, Any]) -> list[str]:
    events: list[str] = []
    if operation == 'onset':
        events.append('enter')
    elif operation == 'offset':
        events.append('leave')
    elif operation == 'interval':
        events.append('duration')
    elif operation == 'navigation_direction':
        events.append('turn')
    elif operation == 'occurrence':
        events.append('event')
    for item in ir.get('event_constraints') or []:
        if isinstance(item, dict):
            events.extend(_clean_ir_strings(item.get('operation')))
            events.extend(_clean_ir_strings(item.get('target_state')))
            events.extend(_clean_ir_strings(item.get('target_entities')))
    return list(dict.fromkeys(events))

def _question_ir_temporal_constraints(ir: dict[str, Any], *, operation: str) -> list[str]:
    constraints = _clean_ir_strings(ir.get('temporal_relation'))
    if operation == 'onset' and 'earliest' not in constraints:
        constraints.append('earliest')
    if operation == 'offset' and 'latest' not in constraints:
        constraints.append('latest')
    if operation == 'current_state' and 'current' not in constraints:
        constraints.append('current')
    return list(dict.fromkeys(constraints))

def _question_ir_spatial_constraints(ir: dict[str, Any]) -> list[str]:
    constraints: list[str] = []
    for item in ir.get('spatial_constraints') or []:
        if isinstance(item, dict):
            constraints.extend(_clean_ir_strings(item.get('relation')))
            constraints.extend(_clean_ir_strings(item.get('target')))
        else:
            constraints.extend(_clean_ir_strings(item))
    return list(dict.fromkeys(constraints))

def _question_ir_hard_constraint_token_groups(ir: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for item in ir.get('hard_constraints') or []:
        texts = _hard_constraint_text_values(item)
        tokens: set[str] = set()
        for text in texts:
            tokens.update(_semantic_tokens(text))
        normalized = tuple(sorted(tokens))
        if normalized:
            groups.append(normalized)
    return tuple(dict.fromkeys(groups))

def _hard_constraint_text_values(item: Any) -> list[str]:
    if isinstance(item, dict):
        values: list[str] = []
        for key in ('text', 'relation', 'operation', 'target_state', 'target', 'target_entities'):
            values.extend(_clean_ir_strings(item.get(key)))
        return values
    return _clean_ir_strings(item)

def _question_ir_focus_tokens(ir: dict[str, Any], *, query_family: str, lexicon: DomainLexicon, fallback_text: str) -> tuple[str, ...]:
    raw_terms: set[str] = set()
    for value in _clean_ir_strings(ir.get('target_entities')):
        raw_terms.update(_semantic_tokens(value))
    for value in _question_ir_attributes(ir):
        raw_terms.update(_semantic_tokens(value))
    target_state = str(ir.get('target_state') or '').strip()
    if query_family in {'temporal', 'duration', 'ego_direction', 'existence'} and target_state:
        raw_terms.update(_semantic_tokens(target_state))
    if not raw_terms:
        raw_terms.update(_semantic_tokens(fallback_text))
    if query_family == 'attribute':
        raw_terms = {token for token in raw_terms if token not in ATTRIBUTE_CONTEXT_TARGET_TOKENS} or raw_terms
    return tuple(sorted(_expand_core_aliases(raw_terms, lexicon=lexicon)))

def _question_ir_constraint_tokens(ir: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ('event_constraints', 'spatial_constraints'):
        for item in ir.get(key) or []:
            if isinstance(item, dict):
                for value in item.values():
                    for text in _clean_ir_strings(value):
                        tokens.update(_semantic_tokens(text))
            else:
                tokens.update(_semantic_tokens(str(item)))
    temporal = str(ir.get('temporal_relation') or '').strip()
    if temporal:
        tokens.update(_semantic_tokens(temporal))
    for group in _question_ir_hard_constraint_token_groups(ir):
        tokens.update(group)
    return tokens

def _clean_ir_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item or '').strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip().lower()]
    return []

def _infer_query_family(normalized_query_text: str, query_tokens: set[str], *, duration: bool, temporal_composition: bool) -> str:
    if duration:
        return 'duration'
    if normalized_query_text.startswith('did you') or bool({'any', 'ever'} & query_tokens):
        return 'existence'
    if bool(ATTRIBUTE_QUERY_HINTS & query_tokens) or 'what color' in normalized_query_text:
        return 'attribute'
    if bool(EGO_DIRECTION_QUERY_HINTS & query_tokens) and bool({'side', 'direction', 'turn', 'turned'} & query_tokens):
        return 'ego_direction'
    if temporal_composition and bool({'when', 'first', 'last', 'leave', 'left', 'enter', 'entered'} & query_tokens):
        return 'temporal'
    if normalized_query_text.startswith('where') or bool({'closest', 'nearest', 'position'} & query_tokens):
        return 'position'
    return 'general'

def _core_terms_for_query(normalized_query_text: str, focus_tokens: tuple[str, ...], *, lexicon: DomainLexicon) -> set[str]:
    terms: set[str] = set(focus_tokens)
    for phrase, aliases in lexicon.core_phrase_aliases.items():
        if phrase in normalized_query_text:
            terms.update(_tokenize(phrase))
            terms.update(aliases)
    for token in focus_tokens:
        terms.update(lexicon.core_term_aliases.get(token, ()))
    terms = {token for token in terms if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    if bool(ATTRIBUTE_QUERY_HINTS & set(focus_tokens)) or 'what color' in normalized_query_text:
        terms = {token for token in terms if token not in ATTRIBUTE_CONTEXT_CORE_TOKENS}
    return terms

def _attribute_binding_terms(query_intent: _QueryIntent) -> tuple[set[str], set[str], set[str]]:
    lexicon = get_domain_lexicon(query_intent.domain_profile or DEFAULT_DOMAIN_PROFILE)
    attribute_value_terms = (set(query_intent.core_terms) | set(query_intent.compiler_attributes)) & COLOR_TOKENS
    attribute_key_terms = _attribute_key_terms(query_intent)
    if not attribute_value_terms and (not attribute_key_terms):
        return (set(), set(), set())
    target_terms: set[str] = set()
    for target in query_intent.compiler_target_entities:
        for token in _tokenize(str(target)):
            if token and token not in ATTRIBUTE_CONTEXT_TARGET_TOKENS:
                target_terms.add(token)
                target_terms.update(lexicon.core_term_aliases.get(token, ()))
    for token in _candidate_focus_tokens(query_intent):
        if token in ATTRIBUTE_TARGET_STOP_TOKENS or token in query_intent.constraint_tokens:
            continue
        target_terms.add(token)
        target_terms.update(lexicon.core_term_aliases.get(token, ()))
    return (attribute_key_terms, attribute_value_terms, {token for token in target_terms if token and token not in ATTRIBUTE_TARGET_STOP_TOKENS})

def _record_matches_attribute_request(record: _GraphRecordView, attribute_key_terms: set[str], attribute_value_terms: set[str]) -> bool:
    tokens = _record_core_tokens(record)
    if attribute_value_terms:
        return bool(tokens & attribute_value_terms)
    if tokens & attribute_key_terms:
        return True
    if 'color' in attribute_key_terms and tokens & COLOR_TOKENS:
        return True
    if 'appearance' in attribute_key_terms and record.source_kind in {'entity_attribute', 'episode', 'state_value', 'entity', 'stable_identity'}:
        return True
    return False

def _record_constraint_overlap(record: _GraphRecordView, query_intent: _QueryIntent) -> set[str]:
    constraint_terms = {token for token in query_intent.constraint_tokens if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    if not constraint_terms:
        return set()
    return _record_core_tokens(record) & constraint_terms

def _record_hard_constraint_group_hits(record: _GraphRecordView, query_intent: _QueryIntent) -> tuple[tuple[str, ...], ...]:
    if not query_intent.hard_constraint_token_groups:
        return ()
    record_tokens = _record_core_tokens(record)
    hits: list[tuple[str, ...]] = []
    for group in query_intent.hard_constraint_token_groups:
        group_terms = {token for token in group if token and token not in NON_SEMANTIC_QUERY_TOKENS}
        overlap = tuple(sorted(record_tokens & group_terms))
        if overlap:
            hits.append(overlap)
    return tuple(hits)

def _hard_constraint_hit_score(hits: tuple[tuple[str, ...], ...], *, record: _GraphRecordView, query_intent: _QueryIntent) -> float:
    if not hits:
        return 0.0
    score = 10.0 * len(hits)
    score += 2.0 * sum((max(0, len(hit) - 1) for hit in hits))
    if len(hits) >= len(query_intent.hard_constraint_token_groups):
        score += 6.0
    if _record_matches_primary_target(record, query_intent):
        score += 5.0
    if record.source_kind in {'episode', 'transition_event'}:
        score += 4.0
    elif record.source_kind == 'entity_attribute':
        score += 2.0
    return score

def _attribute_key_terms(query_intent: _QueryIntent) -> set[str]:
    terms: set[str] = set()
    source_terms = set(query_intent.core_terms) | set(_candidate_focus_tokens(query_intent)) | set(query_intent.compiler_attributes)
    for key, aliases in ATTRIBUTE_KEY_ALIASES.items():
        if key in source_terms or bool(source_terms & set(aliases)):
            terms.add(key)
            terms.update(aliases)
    return {token for token in terms if token}

def _query_wants_base_delta_projection(query_text: str, query_tokens: set[str]) -> bool:
    normalized = ' '.join(query_text.strip().lower().split())
    if not normalized:
        return False
    if any((hint in normalized for hint in BASE_DELTA_QUERY_PHRASE_HINTS)):
        return True
    return bool(BASE_DELTA_QUERY_HINTS & query_tokens)

def _temporal_composition_relevant(record: _GraphRecordView, query_intent: _QueryIntent) -> bool:
    focus_terms = _temporal_composition_focus_terms(query_intent)
    if not focus_terms:
        return True
    primary_overlap = focus_terms & _record_primary_tokens(record)
    if not primary_overlap:
        return False
    transition_overlap = focus_terms & _record_transition_tokens(record)
    if transition_overlap:
        return True
    if query_intent.duration or query_intent.temporal_order:
        return _record_boundary_change_score(record) >= TEMPORAL_COMPOSITION_MIN_BOUNDARY_CHANGE
    return True

def _temporal_composition_focus_terms(query_intent: _QueryIntent) -> set[str]:
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        return {token for token in query_intent.core_terms if token and token not in NON_SEMANTIC_QUERY_TOKENS}
    return {token for token in set(_candidate_focus_tokens(query_intent)) | set(query_intent.core_terms) if token and token not in NON_SEMANTIC_QUERY_TOKENS}

def _record_transition_tokens(record: _GraphRecordView) -> frozenset[str]:
    tokens: set[str] = set()
    for value in record.metadata.get('transition_tokens') or []:
        tokens.update(_tokenize(str(value)))
    for value in record.metadata.get('scene_context_transition_tokens') or []:
        tokens.update(_tokenize(str(value)))
    for value in record.metadata.get('transition_labels') or []:
        prefix, _, label = str(value).partition(':')
        tokens.update(_tokenize(prefix))
        tokens.update(_tokenize(label or str(value)))
    for value in record.metadata.get('scene_context_transition_labels') or []:
        prefix, _, label = str(value).partition(':')
        tokens.update(_tokenize(prefix))
        tokens.update(_tokenize(label or str(value)))
    return frozenset(tokens)

def _record_boundary_change_score(record: _GraphRecordView) -> float:
    try:
        return float(record.metadata.get('boundary_change_score') or 0.0)
    except (TypeError, ValueError):
        return 0.0

def _record_chain_size(record: _GraphRecordView) -> int:
    try:
        return int(record.metadata.get('chain_size') or 0)
    except (TypeError, ValueError):
        return 0

def _ranking_query_tokens(query_tokens: set[str], query_intent: _QueryIntent) -> set[str]:
    if query_intent.query_family not in PRIMARY_INTENT_QUERY_FAMILIES:
        return query_tokens
    ranking_tokens = set(_candidate_focus_tokens(query_intent)) | set(query_intent.core_terms) | set(query_intent.compiler_attributes)
    ranking_tokens.update(_primary_target_terms(query_intent))
    return {token for token in ranking_tokens if token and token not in NON_SEMANTIC_QUERY_TOKENS}

def _query_intent_bonus(record: _GraphRecordView, *, query_intent: _QueryIntent, lexical_overlap: tuple[str, ...], primary_overlap: tuple[str, ...], shared_entities: tuple[str, ...]) -> float:
    if not query_intent.appearance and (not query_intent.disappearance):
        return 0.0
    bonus = 0.0
    event_type = str(record.metadata.get('event_type') or '').upper()
    episode_kind = str(record.metadata.get('episode_kind') or '').upper()
    focus_matched = bool(primary_overlap or shared_entities or (not _candidate_focus_tokens(query_intent)))
    lifecycle_record = record.source_kind == 'entity' or (record.source_kind == 'episode' and episode_kind == 'ENTITY_SEGMENT') or record.source_kind == 'transition_event'
    if query_intent.appearance:
        if record.source_kind == 'transition_event' and event_type == 'APPEAR':
            bonus += 7.0 if focus_matched else -1.0
        elif record.source_kind == 'episode' and episode_kind == 'ENTITY_SEGMENT':
            bonus += 2.4 if focus_matched else -0.8
        elif record.source_kind == 'entity':
            bonus += 0.6 if focus_matched else -0.4
        elif record.source_kind == 'stable_identity':
            bonus -= 0.6
        elif record.source_kind == 'episode' and episode_kind in {'STATE', 'RELATION', RELATION_OBSERVED_BURST_KIND}:
            bonus -= 1.8
    if query_intent.disappearance:
        if record.source_kind == 'transition_event' and event_type == 'DISAPPEAR':
            bonus += 7.0 if focus_matched else -1.0
        elif record.source_kind == 'episode' and episode_kind == 'ENTITY_SEGMENT':
            bonus += 2.0 if focus_matched else -0.8
        elif record.source_kind == 'entity':
            bonus += 0.4 if focus_matched else -0.4
        elif record.source_kind == 'stable_identity':
            bonus -= 0.6
        elif record.source_kind == 'episode' and episode_kind in {'STATE', 'RELATION', RELATION_OBSERVED_BURST_KIND}:
            bonus -= 1.6
    if lifecycle_record and lexical_overlap:
        bonus += 0.8
    if lifecycle_record and shared_entities:
        bonus += 0.8
    return bonus

def _record_evidence_strength(record: _GraphRecordView, *, query_intent: _QueryIntent, core_overlap: tuple[str, ...], lexical_overlap: tuple[str, ...]) -> float:
    if not core_overlap:
        return 0.0
    if record.source_kind in {'entity', 'stable_identity'}:
        return 2.2
    if record.source_kind == 'scene':
        return 3.0 if _direct_semantic_overlap(record, core_overlap) else 1.0
    if record.source_kind == 'interaction_type':
        return -2.0 if _record_is_low_value_relation(record) else -0.5
    if record.source_kind != 'episode':
        return 0.0
    episode_kind = str(record.metadata.get('episode_kind') or '').upper()
    if episode_kind == 'STATE':
        score = 3.0
        if query_intent.query_family == 'attribute':
            score += 2.0
        return score
    if episode_kind == 'ENTITY_SEGMENT':
        return 1.8
    if episode_kind in {'RELATION', RELATION_OBSERVED_BURST_KIND}:
        score = 0.8
        if _record_is_low_value_relation(record):
            score -= 1.6
        if lexical_overlap:
            score += 0.4
        return score
    return 0.0

def _query_family_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent, core_overlap: tuple[str, ...], lexical_overlap: tuple[str, ...]) -> float:
    family = query_intent.query_family
    if family == 'general':
        return 0.0
    score = 0.0
    if query_intent.core_terms and (not core_overlap):
        if family in {'position', 'attribute', 'existence'}:
            if _record_has_semantic_family_fallback(record, query_intent):
                score += 2.5
            score -= 4.5
        elif family == 'ego_direction':
            score -= 1.2
    if family in TEMPORAL_AXIS_QUERY_FAMILIES:
        if record.source_kind == 'temporal_composition':
            score += 5.0
            if _record_boundary_change_score(record) >= TEMPORAL_COMPOSITION_MIN_BOUNDARY_CHANGE:
                score += 2.0
        elif record.source_kind == 'scene':
            score += 2.8
            if core_overlap:
                score += 2.5
        elif record.source_kind in {'transition_event', 'scene_delta'}:
            score += 1.6
        elif record.source_kind == 'interaction_type':
            score -= 5.0
        elif record.source_kind == 'episode' and _record_is_low_value_relation(record) and (not core_overlap):
            score -= 2.0
    elif family == 'position':
        if record.source_kind in {'entity', 'stable_identity', 'scene'} and core_overlap:
            score += 4.0
        if record.source_kind == 'episode' and core_overlap:
            score += 2.5
        if record.source_kind == 'interaction_type':
            score -= 4.0
    elif family == 'attribute':
        if record.source_kind == 'episode' and core_overlap:
            episode_kind = str(record.metadata.get('episode_kind') or '').upper()
            if episode_kind == 'STATE':
                score += 5.0
        if record.source_kind == 'entity' and core_overlap:
            score += 2.0
        if record.source_kind == 'interaction_type':
            score -= 4.0
    elif family == 'existence':
        if _record_is_bound_attribute_existence_answer(record, query_intent):
            score += 5.0
            if record.source_kind == 'entity_attribute':
                score += 7.0
            elif record.source_kind == 'episode':
                score += 8.0
            elif record.source_kind == 'entity':
                score += 2.0
        if core_overlap:
            score += 4.0
        if record.source_kind == 'scene' and core_overlap:
            score += 1.0
        if record.source_kind == 'entity' and core_overlap:
            score += 2.0
        if record.source_kind == 'interaction_type':
            score -= 5.0
    elif family == 'ego_direction':
        if core_overlap or lexical_overlap:
            score += 1.4
        if record.source_kind == 'scene':
            score += 0.8
    return score

def _attribute_binding_score_adjustment(record: _GraphRecordView, *, query_intent: _QueryIntent) -> float:
    attribute_key_terms, attribute_value_terms, target_terms = _attribute_binding_terms(query_intent)
    if not (attribute_key_terms or attribute_value_terms) or not target_terms:
        return 0.0
    record_tokens = _record_core_tokens(record)
    matched_attribute = _record_matches_attribute_request(record, attribute_key_terms, attribute_value_terms)
    matched_targets = bool(record_tokens & target_terms)
    if matched_attribute and matched_targets:
        score = 5.0
        if record.source_kind == 'episode':
            episode_kind = str(record.metadata.get('episode_kind') or '').upper()
            if episode_kind == 'STATE':
                score += 8.0
            elif episode_kind == 'ENTITY_SEGMENT':
                score -= 1.0
        elif record.source_kind == 'state_value':
            score += 3.0
        elif record.source_kind == 'entity_attribute':
            score += 7.0
        elif record.source_kind in {'entity', 'stable_identity'}:
            score -= 1.0
        elif record.source_kind == 'scene':
            score -= 1.5
        return score
    if not matched_attribute or matched_targets:
        return 0.0
    if record.source_kind == 'episode':
        episode_kind = str(record.metadata.get('episode_kind') or '').upper()
        if episode_kind == 'STATE':
            return -8.0
        if episode_kind in {'RELATION', RELATION_OBSERVED_BURST_KIND}:
            return -4.0
    if record.source_kind in {'entity', 'entity_attribute', 'stable_identity', 'scene'}:
        return -2.5
    return -1.0

def _graph_retrieval_profile_from_query(query: MemoryQuery) -> str:
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    return str(metadata.get('graph_retrieval_profile') or '').strip()

def _apply_graph_retrieval_profile_to_evidence_items(items: list[dict[str, Any]], *, query: MemoryQuery) -> list[dict[str, Any]]:
    profile = _graph_retrieval_profile_from_query(query)
    return items
    return [item for item in items]

def _canonical_vector_structural_score_adjustment(record: _GraphRecordView, *, query: MemoryQuery, query_intent: _QueryIntent) -> float:
    dimension = _canonical_vector_attribute_dimension(query, query_intent=query_intent)
    if not dimension:
        return 0.0
    if record.source_kind not in {'entity', 'entity_attribute', 'episode', 'stable_identity', 'state_value'}:
        return 0.0
    record_tokens = _record_core_tokens(record)
    target_terms = _canonical_vector_attribute_owner_terms(query_intent)
    if not target_terms:
        return 0.0
    target_hits = _generic_label_target_overlap(record, target_terms)
    if not target_hits:
        return 0.0
    score = 18.0 + 4.0 * max(0, len(target_hits) - 1)
    if record.source_kind in {'entity_attribute', 'state_value'}:
        score += 10.0
    elif record.source_kind == 'episode' and str(record.metadata.get('episode_kind') or '').upper() == 'STATE':
        score += 8.0
    if dimension == 'appearance':
        return score + (36.0 if record.source_kind == 'entity_attribute' else 10.0)
    dimension_tokens = STRUCTURED_ATTRIBUTE_DIMENSION_TOKENS.get(dimension, set())
    if not dimension_tokens:
        return 0.0
    dimension_hits = record_tokens & dimension_tokens
    if dimension_hits:
        return score + 48.0 + 4.0 * max(0, len(dimension_hits) - 1)
    return score - 46.0

def _canonical_vector_attribute_owner_terms(query_intent: _QueryIntent) -> set[str]:
    if not query_intent.compiler_target_entities:
        return set()
    primary_target = str(query_intent.compiler_target_entities[0] or '').strip().lower()
    if not primary_target:
        return set()
    primary_target = re.split('\\b(?:above|at|behind|below|beside|in|near|on|over|under)\\b', primary_target, maxsplit=1)[0].strip()
    terms = _semantic_tokens(primary_target)
    terms -= set(query_intent.compiler_attributes)
    terms -= COLOR_TOKENS
    terms -= {'attribute', 'color', 'colour', 'material', 'shape', 'state'}
    lexicon = get_domain_lexicon(query_intent.domain_profile or DEFAULT_DOMAIN_PROFILE)
    expanded = _generic_expand_surface_terms(terms)
    for token in tuple(expanded):
        expanded.update(lexicon.core_term_aliases.get(token, ()))
    return _generic_expand_surface_terms(expanded)

def _canonical_vector_attribute_dimension(query: MemoryQuery, *, query_intent: _QueryIntent) -> str:
    ir = _question_ir_from_query(query)
    semantic_ir = ir if isinstance(ir, dict) else {}
    raw_dimension = str(semantic_ir.get('attribute_dimension') or '').strip().lower()
    dimension = STRUCTURED_ATTRIBUTE_DIMENSION_ALIASES.get(raw_dimension, raw_dimension)
    metadata = query.metadata if isinstance(query.metadata, dict) else {}
    raw_question = str(metadata.get('raw_question') or query.query_text or '')
    question_tokens = _tokenize(raw_question)
    if question_tokens & {'color', 'colored', 'colour', 'colours'}:
        dimension = 'color'
    elif 'material' in question_tokens:
        dimension = 'material'
    elif 'shape' in question_tokens:
        dimension = 'shape'
    elif question_tokens & {'appearance', 'look', 'style'}:
        dimension = 'appearance'
    supported_dimensions = set(STRUCTURED_ATTRIBUTE_DIMENSION_TOKENS) | {'appearance'}
    if dimension not in supported_dimensions:
        return ''
    if query_intent.query_family != 'attribute' and raw_dimension not in {'color', 'colour', 'material', 'shape', 'side', 'state'}:
        return ''
    return dimension

def _record_is_bound_attribute_answer(record: _GraphRecordView, query_intent: _QueryIntent) -> bool:
    attribute_key_terms, attribute_value_terms, target_terms = _attribute_binding_terms(query_intent)
    if not (attribute_key_terms or attribute_value_terms) or not target_terms:
        return False
    if record.source_kind not in {'episode', 'state_value', 'entity_attribute'}:
        return False
    tokens = _record_core_tokens(record)
    if not tokens & target_terms:
        return False
    if not _record_matches_attribute_request(record, attribute_key_terms, attribute_value_terms):
        return False
    if record.source_kind == 'episode':
        return str(record.metadata.get('episode_kind') or '').upper() == 'STATE'
    return True

def _record_is_bound_attribute_existence_answer(record: _GraphRecordView, query_intent: _QueryIntent) -> bool:
    if query_intent.query_family != 'existence':
        return False
    attribute_key_terms, attribute_value_terms, target_terms = _attribute_binding_terms(query_intent)
    if not (attribute_key_terms or attribute_value_terms) or not target_terms:
        return False
    if record.source_kind not in {'episode', 'state_value', 'entity_attribute'}:
        return False
    tokens = _record_core_tokens(record)
    if not tokens & target_terms:
        return False
    if not _record_matches_attribute_request(record, attribute_key_terms, attribute_value_terms):
        return False
    if record.source_kind == 'episode':
        return str(record.metadata.get('episode_kind') or '').upper() == 'STATE'
    return True

def _record_matches_primary_target(record: _GraphRecordView, query_intent: _QueryIntent) -> bool:
    target_terms = _primary_target_terms(query_intent)
    if not target_terms:
        return False
    return bool(_record_core_tokens(record) & target_terms)

def _primary_target_terms(query_intent: _QueryIntent) -> set[str]:
    lexicon = get_domain_lexicon(query_intent.domain_profile or DEFAULT_DOMAIN_PROFILE)
    terms: set[str] = set()
    for target in query_intent.compiler_target_entities:
        for token in _tokenize(str(target)):
            if query_intent.query_family == 'attribute' and token in ATTRIBUTE_CONTEXT_TARGET_TOKENS:
                continue
            if token and token not in NON_SEMANTIC_QUERY_TOKENS:
                terms.add(token)
                terms.update(lexicon.core_term_aliases.get(token, ()))
    if query_intent.query_family in PRIMARY_INTENT_QUERY_FAMILIES:
        terms.update((token for token in query_intent.core_terms if token and token not in NON_SEMANTIC_QUERY_TOKENS and (token not in set(query_intent.compiler_attributes)) and (token not in COLOR_TOKENS) and (token not in ATTRIBUTE_QUERY_HINTS)))
    if not terms:
        terms.update((token for token in query_intent.core_terms if token and token not in NON_SEMANTIC_QUERY_TOKENS))
    return terms

def _record_has_semantic_family_fallback(record: _GraphRecordView, query_intent: _QueryIntent) -> bool:
    if query_intent.query_family == 'attribute':
        attribute_key_terms, attribute_value_terms, target_terms = _attribute_binding_terms(query_intent)
        if (attribute_key_terms or attribute_value_terms) and target_terms:
            tokens = _record_core_tokens(record)
            return bool(_record_matches_attribute_request(record, attribute_key_terms, attribute_value_terms) and tokens & target_terms)
        return False
    if query_intent.query_family != 'position':
        return False
    candidate_tokens = set(_candidate_focus_tokens(query_intent))
    if 'nearest' in candidate_tokens or 'closest' in candidate_tokens or 'nearest' in query_intent.constraint_tokens or ('closest' in query_intent.constraint_tokens):
        return record.source_kind in {'scene', 'entity', 'stable_identity', 'episode'}
    return False

def _temporal_composition_is_only_constraint_context(query_intent: _QueryIntent) -> bool:
    if query_intent.query_family in TEMPORAL_AXIS_QUERY_FAMILIES:
        return False
    return bool(query_intent.query_family in {'attribute', 'position', 'existence'} and (query_intent.compiler_events or query_intent.compiler_temporal_constraints or query_intent.compiler_spatial_constraints))

def _record_is_low_value_relation(record: _GraphRecordView) -> bool:
    if record.source_kind == 'interaction_type':
        predicate = str(record.metadata.get('predicate') or '').upper()
        return predicate in LOW_VALUE_PREDICATES
    if record.source_kind != 'episode':
        return False
    predicate = str(record.metadata.get('predicate') or '').upper()
    raw_predicate = str(record.metadata.get('raw_predicate') or '').upper()
    return predicate in LOW_VALUE_PREDICATES or raw_predicate in LOW_VALUE_PREDICATES

def _record_core_tokens(record: _GraphRecordView) -> frozenset[str]:
    tokens = set(_record_primary_tokens(record))
    tokens.update(record.tokens)
    metadata = record.metadata
    for field in ('label', 'primary_entity_label', 'subject_entity_label', 'object_entity_label', 'canonical_label', 'semantic_group', 'attr_value', 'predicate', 'raw_predicate'):
        value = metadata.get(field)
        if value:
            tokens.update(_tokenize(str(value)))
    for value in metadata.get('semantic_alias_terms') or []:
        tokens.update(_tokenize(str(value)))
    for value in metadata.get('salient_labels') or []:
        tokens.update(_tokenize(str(value)))
    return frozenset(tokens)

def _record_primary_tokens(record: _GraphRecordView) -> frozenset[str]:
    metadata = record.metadata
    fields: list[Any] = []
    if record.source_kind == 'entity':
        fields.extend([metadata.get('entity_id'), metadata.get('label'), metadata.get('stable_identity_display_name'), metadata.get('stable_identity_canonical_label'), metadata.get('canonical_label'), metadata.get('semantic_group'), *(metadata.get('semantic_alias_terms') or []), *(metadata.get('raw_visual_attributes') or [])])
    elif record.source_kind == 'entity_attribute':
        fields.extend([metadata.get('entity_id'), metadata.get('label'), metadata.get('canonical_label'), metadata.get('semantic_group'), *(metadata.get('semantic_alias_terms') or []), metadata.get('attr_key'), metadata.get('attr_value'), metadata.get('raw_visual_attribute')])
    elif record.source_kind == 'stable_identity':
        fields.extend([metadata.get('stable_id'), metadata.get('identity_type'), metadata.get('canonical_label'), metadata.get('display_name'), *(metadata.get('member_entity_ids') or []), *(metadata.get('raw_visual_attributes') or [])])
    elif record.source_kind == 'transition_event':
        fields.extend([metadata.get('entity_id'), metadata.get('subject_entity_id'), metadata.get('object_entity_id'), metadata.get('entity_label'), metadata.get('subject_entity_label'), metadata.get('object_entity_label'), metadata.get('event_type'), metadata.get('predicate'), metadata.get('attr_key'), metadata.get('previous_state_value'), metadata.get('current_state_value')])
    elif record.source_kind == 'episode':
        fields.extend([metadata.get('episode_kind'), metadata.get('primary_entity_id'), metadata.get('primary_entity_label'), metadata.get('subject_entity_id'), metadata.get('subject_entity_label'), metadata.get('object_entity_id'), metadata.get('object_entity_label'), metadata.get('predicate'), metadata.get('attr_key'), metadata.get('attr_value')])
    elif record.source_kind == 'scene':
        fields.extend([metadata.get('scene_id'), metadata.get('label'), metadata.get('scene_kind_hint'), *(metadata.get('semantic_alias_terms') or []), *(metadata.get('salient_labels') or [])])
    elif record.source_kind == 'scene_base':
        fields.extend([metadata.get('scene_id'), metadata.get('entity_id'), metadata.get('label'), metadata.get('projection_kind'), *(metadata.get('parent_labels_seen') or []), *(metadata.get('states_seen') or [])])
    elif record.source_kind == 'scene_delta':
        fields.extend([metadata.get('scene_id'), metadata.get('entity_id'), metadata.get('label'), metadata.get('projection_kind'), *(metadata.get('trigger_event_types') or [])])
    elif record.source_kind == 'temporal_composition':
        fields.extend([metadata.get('composition_type'), *(metadata.get('scene_ids') or []), *(metadata.get('transition_labels') or []), *(metadata.get('transition_tokens') or [])])
        for scene in metadata.get('scene_sequence') or []:
            if not isinstance(scene, dict):
                continue
            fields.extend([scene.get('scene_id'), scene.get('label'), scene.get('scene_kind_hint'), *(scene.get('scene_context_hints') or []), *(scene.get('representative_static_labels') or []), *(scene.get('representative_dynamic_labels') or []), *(scene.get('entry_static_labels') or []), *(scene.get('exit_static_labels') or [])])
    else:
        fields.extend(metadata.values())
    tokens: set[str] = set()
    for field in fields:
        normalized = _normalize_index_field(field, video_id=record.video_id)
        if not normalized:
            continue
        tokens.update(_tokenize(normalized))
    return frozenset(tokens)

def _query_window_overlap(record: _GraphRecordView, query: MemoryQuery) -> bool:
    if query.start_ts is None and query.end_ts is None:
        return False
    if not _timed_record(record):
        return False
    query_start = query.start_ts if query.start_ts is not None else -math.inf
    query_end = query.end_ts if query.end_ts is not None else math.inf
    return not (record.end_ts < query_start or record.start_ts > query_end)

def _query_window_gap(record: _GraphRecordView, query: MemoryQuery) -> float | None:
    if query.start_ts is None and query.end_ts is None:
        return None
    if not _timed_record(record):
        return None
    query_start = query.start_ts if query.start_ts is not None else -math.inf
    query_end = query.end_ts if query.end_ts is not None else math.inf
    if _query_window_overlap(record, query):
        return 0.0
    if record.end_ts < query_start:
        return query_start - record.end_ts
    if record.start_ts > query_end:
        return record.start_ts - query_end
    return None

def _record_gap(record_a: _GraphRecordView, record_b: _GraphRecordView) -> float:
    if not _timed_record(record_a) or not _timed_record(record_b):
        return math.inf
    if record_a.end_ts < record_b.start_ts:
        return record_b.start_ts - record_a.end_ts
    if record_b.end_ts < record_a.start_ts:
        return record_a.start_ts - record_b.end_ts
    return 0.0

def _timed_record(record: _GraphRecordView) -> bool:
    return record.start_ts is not None and record.end_ts is not None

def _sort_ts(record: _GraphRecordView) -> float:
    return record.start_ts if record.start_ts is not None else math.inf

def _tokenize(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in TOKEN_PATTERN.findall(text.lower()):
        if not token:
            continue
        tokens.add(token)
        tokens.update((part for part in COMPOSITE_TOKEN_SPLIT_PATTERN.split(token) if part))
    return tokens

def _format_range(start_ts: float | None, end_ts: float | None) -> str:
    if start_ts is None and end_ts is None:
        return 'unknown time'
    if start_ts is None:
        return f'until {end_ts:.2f}s'
    if end_ts is None:
        return f'from {start_ts:.2f}s'
    if start_ts == end_ts:
        return f'{start_ts:.2f}s'
    return f'{start_ts:.2f}s-{end_ts:.2f}s'

def _format_time_range(time_range: Any) -> str:
    start_frame = _range_value(time_range, 'start_frame')
    end_frame = _range_value(time_range, 'end_frame')
    start_ts = _range_value(time_range, 'start_ts')
    end_ts = _range_value(time_range, 'end_ts')
    return _format_time_values(start_frame, end_frame, start_ts, end_ts)

def _format_time_values(start_frame: Any, end_frame: Any, start_ts: Any, end_ts: Any) -> str:
    normalized_start_frame = _coerce_int(start_frame)
    normalized_end_frame = _coerce_int(end_frame)
    normalized_start_ts = _coerce_float(start_ts)
    normalized_end_ts = _coerce_float(end_ts)
    if _uses_frame_axis(normalized_start_frame, normalized_end_frame, normalized_start_ts, normalized_end_ts):
        return _format_frame_range(normalized_start_frame, normalized_end_frame)
    return _format_range(normalized_start_ts, normalized_end_ts)

def _range_value(time_range: Any, key: str) -> Any:
    if isinstance(time_range, dict):
        return time_range.get(key)
    return getattr(time_range, key, None)

def _uses_frame_axis(start_frame: int | None, end_frame: int | None, start_ts: float | None, end_ts: float | None) -> bool:
    if start_frame is None or end_frame is None or start_ts is None or (end_ts is None):
        return False
    return abs(float(start_frame) - start_ts) < 1e-06 and abs(float(end_frame) - end_ts) < 1e-06

def _format_frame_range(start_frame: int | None, end_frame: int | None) -> str:
    if start_frame is None and end_frame is None:
        return 'unknown frame'
    if start_frame is None:
        return f'until frame {end_frame}'
    if end_frame is None:
        return f'from frame {start_frame}'
    if start_frame == end_frame:
        return f'frame {start_frame}'
    return f'frames {start_frame}-{end_frame}'

def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
