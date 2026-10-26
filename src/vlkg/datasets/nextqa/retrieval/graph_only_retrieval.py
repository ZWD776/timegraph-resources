"""Unified graph-only retrieval for NExT-QA canonical temporal graphs."""
from __future__ import annotations
from collections import Counter, deque
from dataclasses import dataclass
import math
from pathlib import Path
from vlkg.workspace import workspace_root
import re
from typing import Any, Iterable, Mapping, Protocol
import yaml
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.nextqa.retrieval.question_ir import QuestionRetrievalInput
from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import CanonicalHybridRetrievalSession
from vlkg.core.graph_retrieval.canonical_record_vector_recall import DEFAULT_VECTOR_MODEL, CanonicalRecordVectorIndex, VectorRecallHit
from vlkg.core.graph_retrieval.canonical_retrieval_types import CanonicalProjection, GraphRecordView
PROFILE_NAMES = {'nextqa_graph_only_unified_v1', 'nextqa_graph_only_unified_v2', 'nextqa_graph_only_unified_v3_evidence24'}
TOKEN_RE = re.compile('[a-z0-9]+')
HUMAN_TERMS = {'baby', 'boy', 'child', 'children', 'girl', 'lady', 'man', 'men', 'people', 'person', 'woman'}
ENTITY_COMPATIBILITY_GROUPS = (HUMAN_TERMS, {'automobile', 'car', 'cars', 'sedan', 'suv', 'truck', 'trucks', 'van', 'vehicle', 'vehicles'}, {'bicycle', 'bicycles', 'bike', 'bikes', 'cycle'})
LOCATION_TERMS = {'balcony', 'building', 'floor', 'garden', 'grassland', 'ground', 'home', 'house', 'outdoor', 'park', 'playground', 'porch', 'restaurant', 'road', 'room', 'stage', 'studio'}
COLOR_TERMS = {'black', 'blue', 'brown', 'gray', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'white', 'yellow'}
STOP_WORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'being', 'by', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'he', 'her', 'his', 'how', 'in', 'is', 'it', 'its', 'near', 'of', 'on', 'she', 'that', 'the', 'their', 'them', 'there', 'they', 'this', 'to', 'was', 'were', 'what', 'where', 'which', 'who', 'why', 'with'}
IRREGULAR_TOKENS = {'children': 'child', 'feet': 'foot', 'men': 'man', 'people': 'person', 'women': 'woman'}
TEMPORAL_EDGE_TYPES = {'ENDS_AT', 'NEXT_EPISODE', 'OVERLAPS_WITH', 'PREVIOUS_EPISODE', 'STARTS_AT', 'TEMPORALLY_NEAR', 'TRIGGERED_BY'}

class VectorIndex(Protocol):
    model_name: str
    cache_status: str

    def search(self, query_text: str, *, top_k: int, min_similarity: float) -> list[VectorRecallHit]:
        ...

@dataclass(frozen=True)
class UnifiedRetrievalConfig:
    profile_name: str
    vector_model: str
    lexical_top_k: int
    vector_top_k: int
    vector_min_similarity: float
    exact_entity_top_k: int
    seed_limit: int
    max_hops: int
    max_neighbors_per_node: int
    expanded_candidate_limit: int
    evidence_limit: int
    max_per_evidence_group: int
    hop_decay: float
    lexical_weight: float
    vector_weight: float
    exact_entity_weight: float
    graph_expansion_weight: float
    structural_weight: float
    temporal_weight: float
    constraint_weight: float
    demand_weight: float
    source_kind_prior: dict[str, float]
    edge_type_weight: dict[str, float]

def load_unified_retrieval_config(path: str | Path) -> UnifiedRetrievalConfig:
    config_path = Path(path).resolve()
    owned_config_dir = workspace_root() / 'configs/datasets/nextqa'
    if not config_path.name.startswith('nextqa_') and config_path.parent != owned_config_dir:
        raise ValueError('NExT-QA retrieval requires a dataset-prefixed config')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('retrieval config must be a mapping')
    serialized_keys = ' '.join(_mapping_keys(payload)).lower()
    forbidden = ('official_type', 'question_type', 'route', 'handler', 'gold')
    if any((term in serialized_keys for term in forbidden)):
        raise ValueError('retrieval config contains a type-, route-, or label-specific key')
    if payload.get('profile_name') not in PROFILE_NAMES:
        raise ValueError(f'profile_name must be one of {sorted(PROFILE_NAMES)}')
    weights = _mapping(payload.get('weights'), 'weights')
    config = UnifiedRetrievalConfig(profile_name=str(payload['profile_name']), vector_model=str(payload.get('vector_model') or DEFAULT_VECTOR_MODEL), lexical_top_k=int(payload['lexical_top_k']), vector_top_k=int(payload['vector_top_k']), vector_min_similarity=float(payload['vector_min_similarity']), exact_entity_top_k=int(payload['exact_entity_top_k']), seed_limit=int(payload['seed_limit']), max_hops=int(payload['max_hops']), max_neighbors_per_node=int(payload['max_neighbors_per_node']), expanded_candidate_limit=int(payload['expanded_candidate_limit']), evidence_limit=int(payload['evidence_limit']), max_per_evidence_group=int(payload['max_per_evidence_group']), hop_decay=float(payload['hop_decay']), lexical_weight=float(weights['lexical']), vector_weight=float(weights['vector']), exact_entity_weight=float(weights['exact_entity']), graph_expansion_weight=float(weights['graph_expansion']), structural_weight=float(weights['structural']), temporal_weight=float(weights['temporal']), constraint_weight=float(weights['constraint']), demand_weight=float(weights.get('demand', 0.0)), source_kind_prior={str(key): float(value) for key, value in _mapping(payload.get('source_kind_prior'), 'source_kind_prior').items()}, edge_type_weight={str(key): float(value) for key, value in _mapping(payload.get('edge_type_weight'), 'edge_type_weight').items()})
    positive_ints = (config.lexical_top_k, config.vector_top_k, config.exact_entity_top_k, config.seed_limit, config.max_neighbors_per_node, config.expanded_candidate_limit, config.evidence_limit, config.max_per_evidence_group)
    if any((value <= 0 for value in positive_ints)) or config.max_hops < 1:
        raise ValueError('retrieval budgets must be positive')
    if not 0.0 < config.hop_decay <= 1.0:
        raise ValueError('hop_decay must be in (0, 1]')
    return config

class UnifiedGraphOnlyRetriever:
    """One fixed retrieval algorithm for every NExT-QA question."""

    def __init__(self, graph: CanonicalTemporalMemoryGraph, *, config: UnifiedRetrievalConfig, vector_cache_path: str | Path | None, vector_index: VectorIndex | None=None) -> None:
        self.graph = graph
        self.config = config
        self.projection: CanonicalProjection = CanonicalHybridRetrievalSession(graph).projection
        self.vector_index: VectorIndex = vector_index or CanonicalRecordVectorIndex.build_or_load(self.projection, cache_path=vector_cache_path, model_name=config.vector_model)
        self.record_tokens = {record_id: _record_tokens(record) for record_id, record in self.projection.records.items()}
        self.document_frequency = Counter((token for tokens in self.record_tokens.values() for token in tokens))
        self.record_count = max(1, len(self.projection.records))
        self.max_degree = max((len(links) for links in self.projection.adjacency.values()), default=1)
        self.video_end_ts = max((record.end_ts for record in self.projection.records.values() if record.end_ts is not None), default=0.0)

    def retrieve(self, question: QuestionRetrievalInput) -> dict[str, Any]:
        expected_video_id = f'nextqa-val/{question.video_id}'
        if self.graph.video_id != expected_video_id:
            raise ValueError(f'question/graph video mismatch: {question.video_id} vs {self.graph.video_id}')
        query_tokens = _query_tokens(question.normalized_question)
        constraint_tokens = _query_tokens(' '.join((*question.target_entities, *question.actors, *question.actions, *question.objects, question.referenced_event)))
        lexical_scores = {record_id: self._lexical_score(query_tokens, tokens) for record_id, tokens in self.record_tokens.items()}
        vector_hits = self.vector_index.search(question.normalized_question, top_k=self.config.vector_top_k, min_similarity=self.config.vector_min_similarity)
        vector_scores = {hit.record_id: max(0.0, float(hit.similarity)) for hit in vector_hits if hit.record_id in self.projection.records}
        exact_entity_ids = self._exact_entity_ids(question)
        exact_scores = {record_id: 1.0 for entity_id in exact_entity_ids for record_id in self.projection.entity_index.get(entity_id, set())}
        lexical_ids = _top_ids(lexical_scores, limit=self.config.lexical_top_k, positive_only=True)
        exact_ids = sorted(exact_scores, key=lambda record_id: (-lexical_scores.get(record_id, 0.0), record_id))[:self.config.exact_entity_top_k]
        candidate_ids = set(lexical_ids) | set(vector_scores) | set(exact_ids)
        base_scores = {record_id: self._base_score(record_id, lexical_scores=lexical_scores, vector_scores=vector_scores, exact_scores=exact_scores) for record_id in candidate_ids}
        seed_ids = _top_ids(base_scores, limit=self.config.seed_limit)
        expansion = self._expand(seed_ids, base_scores, lexical_scores)
        ranked: list[dict[str, Any]] = []
        for record_id, path in expansion.items():
            record = self.projection.records[record_id]
            lexical = lexical_scores.get(record_id, 0.0)
            vector = vector_scores.get(record_id, 0.0)
            exact = exact_scores.get(record_id, 0.0)
            structural = self._structural_score(record_id)
            temporal = self._temporal_score(record, path['edge_types'], question)
            constraint = _overlap_score(constraint_tokens, self.record_tokens.get(record_id, set()))
            demand = self._demand_score(record, exact, question)
            source_prior = self.config.source_kind_prior.get(record.source_kind, 0.0)
            total = self._base_score(record_id, lexical_scores=lexical_scores, vector_scores=vector_scores, exact_scores=exact_scores) + self.config.graph_expansion_weight * float(path['propagated_score']) + self.config.structural_weight * structural + self.config.temporal_weight * temporal + self.config.constraint_weight * constraint + self.config.demand_weight * demand + source_prior
            ranked.append({'record': record, 'source_record_id': record_id, 'source_kind': record.source_kind, 'score': total, 'components': {'lexical': lexical, 'vector': vector, 'exact_entity': exact, 'graph_expansion': float(path['propagated_score']), 'structural': structural, 'temporal': temporal, 'constraint': constraint, 'demand': demand, 'source_kind_prior': source_prior}, 'hop_count': int(path['hop_count']), 'path_record_ids': list(path['record_ids']), 'path_edge_types': list(path['edge_types'])})
        ranked.sort(key=lambda item: (-float(item['score']), int(item['hop_count']), str(item['source_record_id'])))
        selected = _select_diverse_evidence(ranked, limit=self.config.evidence_limit, max_per_group=self.config.max_per_evidence_group)
        evidence = [self._serialize_evidence(item, rank=rank) for rank, item in enumerate(selected, start=1)]
        return {'schema_version': 'nextqa_graph_retrieval_record_v1', 'question_id': question.question_id, 'video_id': question.video_id, 'graph_video_id': self.graph.video_id, 'status': 'completed' if evidence else 'empty', 'graph_retrieval_profile': self.config.profile_name, 'retrieval_algorithm': ['question_only_lexical_recall', 'question_only_canonical_record_vector_recall', 'exact_entity_recall', 'candidate_union', 'fixed_graph_neighborhood_expansion', 'fixed_structural_temporal_rerank'], 'graph_record_ids': [item['source_record_id'] for item in evidence], 'evidence': evidence, 'candidate_counts': {'lexical': len(lexical_ids), 'vector': len(vector_scores), 'exact_entity': len(exact_ids), 'union': len(candidate_ids), 'expanded': len(expansion), 'rendered': len(evidence)}, 'exact_entity_ids': sorted(exact_entity_ids), 'vector_recall': {'model': getattr(self.vector_index, 'model_name', self.config.vector_model), 'cache_status': getattr(self.vector_index, 'cache_status', 'unknown'), 'hit_count': len(vector_scores), 'uses_visual_embeddings': False, 'uses_faiss': False}, 'purity': {'uses_gold_answer': False, 'uses_official_type': False, 'uses_answer_choices': False, 'reads_rgb': False, 'reads_object_crop': False, 'reads_visual_embedding_cache': False, 'uses_visual_index': False, 'shared_hybrid_retrieve_called': False}}

    def _lexical_score(self, query_tokens: set[str], record_tokens: set[str]) -> float:
        overlap = query_tokens & record_tokens
        if not overlap or not query_tokens:
            return 0.0
        denominator = sum((self._idf(token) for token in query_tokens))
        return sum((self._idf(token) for token in overlap)) / max(denominator, 1e-09)

    def _idf(self, token: str) -> float:
        return math.log((self.record_count + 1) / (self.document_frequency[token] + 1)) + 1.0

    def _base_score(self, record_id: str, *, lexical_scores: Mapping[str, float], vector_scores: Mapping[str, float], exact_scores: Mapping[str, float]) -> float:
        return self.config.lexical_weight * lexical_scores.get(record_id, 0.0) + self.config.vector_weight * vector_scores.get(record_id, 0.0) + self.config.exact_entity_weight * exact_scores.get(record_id, 0.0)

    def _exact_entity_ids(self, question: QuestionRetrievalInput) -> set[str]:
        query = question.normalized_question
        query_term_set = set(_tokens(query))
        matched: set[str] = set()
        for record in self.projection.records.values():
            if record.source_kind not in {'entity', 'stable_identity'}:
                continue
            aliases = _record_aliases(record)
            direct_match = any((_phrase_in_text(alias, query) for alias in aliases))
            alias_tokens = set().union(*(_tokens(alias) for alias in aliases))
            compatible_match = any((bool(query_term_set & group) and bool(alias_tokens & group) for group in ENTITY_COMPATIBILITY_GROUPS))
            if direct_match or compatible_match:
                matched.update(record.entity_ids)
        return matched

    def _demand_score(self, record: GraphRecordView, exact_entity_score: float, question: QuestionRetrievalInput) -> float:
        tokens = self.record_tokens.get(record.source_record_id, set())
        if question.count_query:
            if exact_entity_score and record.source_kind == 'stable_identity':
                return 1.0
            if exact_entity_score and record.source_kind == 'entity':
                return 0.8
            if exact_entity_score and record.source_kind == 'episode':
                return 0.35
            return 0.0
        if question.location_query:
            location_signal = 1.0 if tokens & LOCATION_TERMS else 0.0
            scene_signal = 0.8 if record.source_kind == 'scene' else 0.0
            relation_signal = 0.35 if record.source_kind == 'episode' and tokens & LOCATION_TERMS else 0.0
            return min(1.0, max(location_signal, scene_signal, relation_signal))
        if question.attribute_query == 'color':
            color_signal = 1.0 if tokens & COLOR_TERMS else 0.0
            if exact_entity_score and record.source_kind == 'entity_attribute':
                return max(0.9, color_signal)
            if exact_entity_score and record.source_kind in {'entity', 'stable_identity'}:
                return max(0.7, color_signal)
            return 0.25 * color_signal
        return 0.0

    def _expand(self, seed_ids: list[str], base_scores: Mapping[str, float], lexical_scores: Mapping[str, float]) -> dict[str, dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {seed_id: {'seed_id': seed_id, 'record_ids': (seed_id,), 'edge_types': (), 'hop_count': 0, 'propagated_score': 0.0} for seed_id in seed_ids}
        per_seed_limit = max(1, self.config.expanded_candidate_limit // max(1, len(seed_ids)))
        for seed_id in seed_ids:
            queue: deque[tuple[str, tuple[str, ...], tuple[str, ...], int]] = deque([(seed_id, (seed_id,), (), 0)])
            visited = {seed_id}
            visited_result_count = 0
            while queue and visited_result_count < per_seed_limit:
                record_id, record_path, edge_path, hop = queue.popleft()
                visited_result_count += 1
                propagated = 0.0 if hop == 0 else base_scores.get(seed_id, 0.0) * self.config.hop_decay ** hop + _edge_path_mean(edge_path, self.config.edge_type_weight)
                previous = best.get(record_id)
                if previous is None or propagated > float(previous['propagated_score']):
                    best[record_id] = {'seed_id': seed_id, 'record_ids': record_path, 'edge_types': edge_path, 'hop_count': hop, 'propagated_score': propagated}
                if hop >= self.config.max_hops:
                    continue
                links = sorted(self.projection.adjacency.get(record_id, []), key=lambda link: (-self.config.edge_type_weight.get(link.edge_type, 0.0), -lexical_scores.get(link.neighbor_id, 0.0), link.neighbor_id))[:self.config.max_neighbors_per_node]
                for link in links:
                    if link.neighbor_id in visited:
                        continue
                    visited.add(link.neighbor_id)
                    queue.append((link.neighbor_id, (*record_path, link.neighbor_id), (*edge_path, link.edge_type), hop + 1))
        return best

    def _structural_score(self, record_id: str) -> float:
        degree = len(self.projection.adjacency.get(record_id, []))
        return math.log1p(degree) / max(math.log1p(self.max_degree), 1e-09)

    def _temporal_score(self, record: GraphRecordView, edge_types: Iterable[str], question: QuestionRetrievalInput) -> float:
        if question.temporal_direction == 'none' and question.temporal_anchor == 'none':
            return 0.0
        edge_set = set(edge_types)
        temporal_connectivity = 1.0 if edge_set & TEMPORAL_EDGE_TYPES else 0.0
        direction_match = 0.0
        if question.temporal_direction == 'before' and 'PREVIOUS_EPISODE' in edge_set:
            direction_match = 1.0
        elif question.temporal_direction == 'after' and 'NEXT_EPISODE' in edge_set:
            direction_match = 1.0
        elif question.temporal_direction == 'concurrent' and 'OVERLAPS_WITH' in edge_set:
            direction_match = 1.0
        anchor_match = self._anchor_alignment(record, question.temporal_anchor)
        timed = 1.0 if record.start_ts is not None or record.end_ts is not None else 0.0
        return 0.15 * timed + 0.2 * temporal_connectivity + 0.25 * direction_match + 0.4 * anchor_match

    def _anchor_alignment(self, record: GraphRecordView, anchor: str) -> float:
        if anchor == 'none' or self.video_end_ts <= 0.0:
            return 0.0
        start = record.start_ts if record.start_ts is not None else record.end_ts
        end = record.end_ts if record.end_ts is not None else record.start_ts
        if start is None or end is None:
            return 0.0
        position = min(1.0, max(0.0, (start + end) / 2.0 / self.video_end_ts))
        if anchor == 'video_beginning':
            return 1.0 - position
        if anchor == 'video_end':
            return position
        if anchor == 'video_middle':
            return max(0.0, 1.0 - abs(position - 0.5) * 2.0)
        return 0.0

    def _serialize_evidence(self, item: Mapping[str, Any], *, rank: int) -> dict[str, Any]:
        record = item['record']
        return {'rank': rank, 'source_record_id': item['source_record_id'], 'source_kind': item['source_kind'], 'score': round(float(item['score']), 6), 'score_components': {key: round(float(value), 6) for key, value in item['components'].items()}, 'evidence_text': render_record_evidence(record), 'time_range': _time_range(record), 'entity_ids': list(record.entity_ids), 'scene_ids': list(record.scene_ids), 'hop_count': item['hop_count'], 'path_record_ids': item['path_record_ids'], 'path_edge_types': item['path_edge_types']}

def render_record_evidence(record: GraphRecordView) -> str:
    metadata = record.metadata
    time_text = _render_time(record)
    if record.source_kind == 'scene':
        label = _first_text(metadata, 'label', 'scene_kind_hint') or 'scene'
        return f'Scene contains {label}{time_text}.'
    if record.source_kind in {'entity', 'stable_identity'}:
        label = _first_text(metadata, 'canonical_label', 'label') or 'object'
        attributes = _flatten_text(metadata.get('raw_visual_attributes'))
        detail = f"; observed attributes: {', '.join(attributes[:3])}" if attributes else ''
        return f'Observed {label}{time_text}{detail}.'
    if record.source_kind == 'entity_attribute':
        label = _first_text(metadata, 'label', 'canonical_label') or 'entity'
        key = str(metadata.get('attr_key') or 'attribute')
        value = str(metadata.get('attr_value') or 'unknown')
        return f'{label} has {key}={value}{time_text}.'
    if record.source_kind == 'episode':
        kind = str(metadata.get('episode_kind') or '').upper()
        if kind.startswith('RELATION'):
            subject = _first_text(metadata, 'subject_entity_label', 'primary_entity_label') or 'entity'
            predicate = _first_text(metadata, 'raw_predicate', 'predicate') or 'related to'
            obj = _first_text(metadata, 'object_entity_label') or 'object'
            return f'{subject} {predicate} {obj}{time_text}.'
        label = _first_text(metadata, 'primary_entity_label', 'label') or 'entity'
        key = str(metadata.get('attr_key') or 'state')
        value = str(metadata.get('attr_value') or 'observed')
        return f'{label} has {key}={value}{time_text}.'
    if record.source_kind == 'transition_event':
        event = str(metadata.get('event_type') or 'transition').lower().replace('_', ' ')
        label = _first_text(metadata, 'entity_label', 'subject_entity_label') or 'entity'
        obj = _first_text(metadata, 'object_entity_label')
        predicate = _first_text(metadata, 'raw_predicate', 'predicate')
        relation = f': {label} {predicate} {obj}' if predicate and obj else f': {label}'
        return f'{event}{relation}{time_text}.'
    return f"{record.summary.rstrip('.')}{time_text}."

def _record_tokens(record: GraphRecordView) -> set[str]:
    values = [record.index_text, record.summary]
    for key, value in record.metadata.items():
        if key.endswith('_id') or key.endswith('_ids') or key in {'node_id', 'video_id'}:
            continue
        values.extend(_flatten_text(value))
    return _query_tokens(' '.join(values))

def _record_aliases(record: GraphRecordView) -> set[str]:
    values: list[str] = []
    for key in ('label', 'canonical_label', 'labels_seen', 'semantic_alias_terms'):
        values.extend(_flatten_text(record.metadata.get(key)))
    attributes = record.metadata.get('attributes')
    if isinstance(attributes, dict):
        for key in ('canonical_label', 'semantic_alias_terms'):
            values.extend(_flatten_text(attributes.get(key)))
    return {' '.join(TOKEN_RE.findall(value.lower())) for value in values if TOKEN_RE.findall(value.lower())}

def _query_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for token in _tokens(text):
        if token in STOP_WORDS:
            continue
        result.add(token)
        result.add(_stem(token))
    return result

def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(str(text or '').lower()))

def _stem(token: str) -> str:
    if token in IRREGULAR_TOKENS:
        return IRREGULAR_TOKENS[token]
    if len(token) > 5 and token.endswith('ing'):
        root = token[:-3]
        if len(root) > 2 and root[-1] == root[-2]:
            root = root[:-1]
        return root
    if len(token) > 4 and token.endswith('ied'):
        return token[:-3] + 'y'
    if len(token) > 4 and token.endswith('ed'):
        return token[:-2]
    if len(token) > 4 and token.endswith('es'):
        return token[:-2]
    if len(token) > 3 and token.endswith('s'):
        return token[:-1]
    return token

def _phrase_in_text(phrase: str, text: str) -> bool:
    if not phrase:
        return False
    return re.search(f'(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])', text) is not None

def _overlap_score(query_tokens: set[str], record_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens & record_tokens) / len(query_tokens)

def _top_ids(scores: Mapping[str, float], *, limit: int, positive_only: bool=False) -> list[str]:
    items = ((record_id, score) for record_id, score in scores.items() if not positive_only or score > 0.0)
    return [record_id for record_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))[:limit]]

def _edge_path_mean(edge_types: Iterable[str], weights: Mapping[str, float]) -> float:
    edge_list = list(edge_types)
    if not edge_list:
        return 0.0
    return sum((weights.get(edge_type, 0.0) for edge_type in edge_list)) / len(edge_list)

def _select_diverse_evidence(ranked: list[dict[str, Any]], *, limit: int, max_per_group: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    for item in ranked:
        group = _evidence_group(item['record'])
        if group_counts[group] >= max_per_group:
            continue
        selected.append(item)
        selected_ids.add(item['source_record_id'])
        group_counts[group] += 1
        if len(selected) >= limit:
            return selected
    for item in ranked:
        if item['source_record_id'] in selected_ids:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected

def _evidence_group(record: GraphRecordView) -> str:
    metadata = record.metadata
    if record.source_kind == 'episode':
        return ':'.join((record.source_kind, str(metadata.get('subject_entity_id') or metadata.get('primary_entity_id') or ''), str(metadata.get('predicate') or metadata.get('attr_key') or ''), str(metadata.get('object_entity_id') or '')))
    if record.entity_ids:
        return f"{record.source_kind}:{'|'.join(record.entity_ids)}"
    return f'{record.source_kind}:{record.source_record_id}'

def _time_range(record: GraphRecordView) -> dict[str, float | None]:
    return {'start_ts': record.start_ts, 'end_ts': record.end_ts}

def _render_time(record: GraphRecordView) -> str:
    if record.start_ts is None and record.end_ts is None:
        return ''
    if record.start_ts == record.end_ts or record.end_ts is None:
        return f' at {float(record.start_ts or record.end_ts or 0.0):.3f}s'
    return f' from {float(record.start_ts or 0.0):.3f}s to {float(record.end_ts):.3f}s'

def _first_text(metadata: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        values = _flatten_text(metadata.get(key))
        if values:
            return values[0]
    return ''

def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = ' '.join(value.split()).strip()
        return [normalized] if normalized else []
    if isinstance(value, Mapping):
        values: list[str] = []
        for nested in value.values():
            values.extend(_flatten_text(nested))
        return values
    if isinstance(value, Iterable) and (not isinstance(value, (bytes, bytearray))):
        values: list[str] = []
        for nested in value:
            values.extend(_flatten_text(nested))
        return values
    return [str(value)]

def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    return value

def _mapping_keys(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    keys: list[str] = []
    for key, nested in value.items():
        keys.append(str(key))
        keys.extend(_mapping_keys(nested))
    return keys
