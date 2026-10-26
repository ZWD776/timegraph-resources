from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from vlkg.core.models.canonical_graph import CanonicalGraphEdge


CanonicalRetrievalMode = Literal["temporal", "entity"]
CanonicalSourceKind = Literal[
    "scene",
    "entity",
    "entity_attribute",
    "stable_identity",
    "episode",
    "transition_event",
    "state_value",
    "interaction_type",
    "scene_base",
    "scene_delta",
    "temporal_composition",
]


@dataclass(frozen=True)
class CanonicalHybridRetrievalOutput:
    items: list[dict[str, Any]]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class GraphRecordView:
    source_kind: CanonicalSourceKind
    source_record_id: str
    summary: str
    metadata: dict[str, Any]
    index_text: str
    tokens: frozenset[str]
    video_id: str
    scene_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    start_ts: float | None
    end_ts: float | None
    priority: int


@dataclass(frozen=True)
class GraphLink:
    edge_id: str
    edge_type: str
    neighbor_id: str


@dataclass(frozen=True)
class ScoredRecord:
    record: GraphRecordView
    score: float
    lexical_overlap: tuple[str, ...] = ()
    exact_substring: bool = False
    shared_entities: tuple[str, ...] = ()
    shared_scenes: tuple[str, ...] = ()
    query_window_overlap: bool = False
    semantic_similarity: float | None = None


@dataclass(frozen=True)
class GraphPath:
    record_ids: tuple[str, ...]
    edge_types: tuple[str, ...]

    @property
    def hop_count(self) -> int:
        return len(self.edge_types)


@dataclass(frozen=True)
class RelationSignal:
    score: float
    relation_types: tuple[str, ...]
    shared_entities: tuple[str, ...] = ()
    shared_scenes: tuple[str, ...] = ()
    path_record_ids: tuple[str, ...] = ()
    path_edge_types: tuple[str, ...] = ()
    hop_count: int = 0


@dataclass(frozen=True)
class GraphFragment:
    seed_record_id: str
    record_ids: tuple[str, ...]
    score: float
    relation_types: tuple[str, ...]
    shared_entities: tuple[str, ...] = ()
    shared_scenes: tuple[str, ...] = ()
    path_record_ids: tuple[str, ...] = ()
    path_edge_types: tuple[str, ...] = ()
    hop_count: int = 0


@dataclass(frozen=True)
class QueryIntent:
    domain_profile: str = ""
    appearance: bool = False
    disappearance: bool = False
    temporal_composition: bool = False
    duration: bool = False
    temporal_order: bool = False
    earliest: bool = False
    latest: bool = False
    query_family: str = "general"
    core_terms: tuple[str, ...] = ()
    focus_tokens: tuple[str, ...] = ()
    constraint_tokens: tuple[str, ...] = ()
    hard_constraint_token_groups: tuple[tuple[str, ...], ...] = ()
    compiler_answer_type: str = ""
    compiler_retrieval_strategy: str = ""
    compiler_current_window_required: bool = False
    compiler_target_entities: tuple[str, ...] = ()
    compiler_attributes: tuple[str, ...] = ()
    compiler_events: tuple[str, ...] = ()
    compiler_temporal_constraints: tuple[str, ...] = ()
    compiler_spatial_constraints: tuple[str, ...] = ()
    compiler_ranking_policy: tuple[str, ...] = ()
    compiler_preferred_source_kinds: tuple[str, ...] = ()
    compiler_risk_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryPlan:
    global_scope_score: int
    retrieval_budget: int
    seed_limit: int
    max_hops: int
    domain_profile: str = ""
    candidate_record_ids: frozenset[str] | None = None
    candidate_strategy: str = "global_scan"
    global_fallback: bool = False
    scene_seed_count: int = 0
    query_family: str = "general"
    core_terms: tuple[str, ...] = ()
    constraint_tokens: tuple[str, ...] = ()
    hard_constraint_token_groups: tuple[tuple[str, ...], ...] = ()
    compiler_answer_type: str = ""
    compiler_retrieval_strategy: str = ""
    compiler_current_window_required: bool = False
    compiler_target_entities: tuple[str, ...] = ()
    compiler_attributes: tuple[str, ...] = ()
    compiler_events: tuple[str, ...] = ()
    compiler_temporal_constraints: tuple[str, ...] = ()
    compiler_spatial_constraints: tuple[str, ...] = ()
    compiler_ranking_policy: tuple[str, ...] = ()
    compiler_risk_notes: tuple[str, ...] = ()


@dataclass
class CanonicalProjection:
    records: dict[str, GraphRecordView] = field(default_factory=dict)
    adjacency: dict[str, list[GraphLink]] = field(default_factory=dict)
    entity_index: dict[str, set[str]] = field(default_factory=dict)
    scene_index: dict[str, set[str]] = field(default_factory=dict)
    token_index: dict[str, set[str]] = field(default_factory=dict)
    source_kind_index: dict[CanonicalSourceKind, set[str]] = field(default_factory=dict)

    def add_record(self, record: GraphRecordView) -> None:
        self.records[record.source_record_id] = record
        self.source_kind_index.setdefault(record.source_kind, set()).add(record.source_record_id)
        for token in record.tokens:
            self.token_index.setdefault(token, set()).add(record.source_record_id)
        for entity_id in record.entity_ids:
            self.entity_index.setdefault(entity_id, set()).add(record.source_record_id)
        for scene_id in record.scene_ids:
            self.scene_index.setdefault(scene_id, set()).add(record.source_record_id)

    def add_edge(self, edge: CanonicalGraphEdge) -> None:
        self.add_link(
            source_record_id=edge.source_node_id,
            target_record_id=edge.target_node_id,
            edge_type=edge.edge_type,
            edge_id=edge.edge_id,
        )

    def add_link(
        self,
        *,
        source_record_id: str,
        target_record_id: str,
        edge_type: str,
        edge_id: str | None = None,
    ) -> None:
        if source_record_id not in self.records or target_record_id not in self.records:
            return
        link_id = edge_id or f"projection:{edge_type}:{source_record_id}:{target_record_id}"
        self.adjacency.setdefault(source_record_id, []).append(
            GraphLink(
                edge_id=link_id,
                edge_type=edge_type,
                neighbor_id=target_record_id,
            )
        )
        self.adjacency.setdefault(target_record_id, []).append(
            GraphLink(
                edge_id=link_id,
                edge_type=edge_type,
                neighbor_id=source_record_id,
            )
        )
