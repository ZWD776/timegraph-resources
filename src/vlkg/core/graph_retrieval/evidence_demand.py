from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Iterable


EVIDENCE_DEMAND_CONTRACT = "evidence_demand.v1"

EVIDENCE_UNIT_GRAPH_FACT = "graph_fact"
EVIDENCE_UNIT_OBJECT_CROP = "object_crop"
EVIDENCE_UNIT_FULL_FRAME = "full_frame"
EVIDENCE_UNIT_FRAME_WINDOW = "frame_window"

AGGREGATION_SINGLE = "single"
AGGREGATION_SET = "set"
AGGREGATION_COUNT = "count"
AGGREGATION_COVERAGE = "coverage"
AGGREGATION_INTERVAL = "interval"

_DIRECTIONAL_RELATIONS = {
    "above",
    "below",
    "behind",
    "in front of",
    "inside",
    "left of",
    "on",
    "on top of",
    "right of",
    "under",
}
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class VisualQueryIntent:
    text: str
    role: str
    evidence_unit: str
    owner_terms: tuple[str, ...] = ()
    relation: str = ""
    source: str = "evidence_demand"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceDemand:
    primary_targets: tuple[str, ...] = ()
    target_descriptors: tuple[str, ...] = ()
    requested_facets: tuple[str, ...] = ()
    reference_entities: tuple[str, ...] = ()
    spatial_relations: tuple[str, ...] = ()
    event_constraints: tuple[str, ...] = ()
    evidence_units: tuple[str, ...] = (EVIDENCE_UNIT_GRAPH_FACT,)
    temporal_scope: dict[str, Any] = field(default_factory=dict)
    aggregation: str = AGGREGATION_SINGLE
    owner_binding_required: bool = False
    reference_binding_required: bool = False
    direction_binding_required: bool = False
    negative_evidence_coverage_required: bool = False
    route_id: str = ""
    taxonomy_path: str = ""
    query_intents: tuple[VisualQueryIntent, ...] = ()
    source_fields: tuple[str, ...] = ()
    uses_gold_answer: bool = False
    contract: str = EVIDENCE_DEMAND_CONTRACT

    def to_dict(self) -> dict[str, Any]:
        payload = _json_compatible(asdict(self))
        payload["query_intents"] = [intent.to_dict() for intent in self.query_intents]
        return payload


def compile_evidence_demand(
    question: str,
    *,
    question_ir: dict[str, Any] | None = None,
    question_route: dict[str, Any] | None = None,
) -> EvidenceDemand:
    """Compile retrieval needs without reading answers, graph data, or images."""

    ir = question_ir if isinstance(question_ir, dict) else {}
    route = _question_route(ir, question_route)
    route_id = _clean_text(route.get("route_id") or ir.get("question_route_id"))
    taxonomy_path = _clean_text(
        ir.get("evidence_question_type")
        or route.get("taxonomy_path")
        or ir.get("question_type")
    )

    references, relations = _spatial_bindings(ir.get("spatial_constraints"))
    references = _ordered_unique(
        [
            *references,
            *_constraint_targets(ir.get("hard_constraints"), spatial_only=True),
        ]
    )
    raw_targets = _ordered_unique(
        [
            *_text_values(ir.get("target_entities")),
            *_text_values(route.get("primary_target_terms")),
        ]
    )
    reference_norms = {_normalize(value) for value in references}
    primary_targets = tuple(
        value for value in raw_targets if _normalize(value) not in reference_norms
    )
    if not primary_targets:
        primary_targets = tuple(raw_targets)

    descriptors = _ordered_unique(
        [
            *_text_values(ir.get("target_descriptors")),
            *_attribute_constraint_texts(ir.get("hard_constraints")),
            *_target_state_descriptors(ir.get("target_state"), primary_targets),
        ]
    )
    requested_facets = _ordered_unique(
        [
            *_text_values(ir.get("attribute_dimension")),
            *_text_values(ir.get("attribute_dimension_aliases")),
            *_text_values(route.get("attribute_terms")),
        ]
    )
    event_constraints = _ordered_unique(
        [
            *_constraint_texts(ir.get("event_constraints")),
            *_constraint_texts(route.get("event_constraints")),
        ]
    )
    relations = _ordered_unique(
        [*relations, *_text_values(route.get("spatial_constraints"))]
    )
    aggregation = _aggregation(ir, route_id=route_id, taxonomy_path=taxonomy_path)
    evidence_units = _evidence_units(
        route_id=route_id,
        taxonomy_path=taxonomy_path,
        aggregation=aggregation,
    )
    owner_binding_required = bool(primary_targets and requested_facets)
    reference_binding_required = bool(references)
    direction_binding_required = bool(
        {_normalize(value) for value in relations} & _DIRECTIONAL_RELATIONS
    )
    negative_coverage = _requires_negative_coverage(ir, route_id=route_id)
    query_intents = _build_query_intents(
        primary_targets=primary_targets,
        descriptors=descriptors,
        references=references,
        relations=relations,
        evidence_units=evidence_units,
    )
    source_fields = _present_source_fields(ir, route)

    return EvidenceDemand(
        primary_targets=primary_targets,
        target_descriptors=descriptors,
        requested_facets=requested_facets,
        reference_entities=references,
        spatial_relations=relations,
        event_constraints=event_constraints,
        evidence_units=evidence_units,
        temporal_scope=_temporal_scope(ir, route),
        aggregation=aggregation,
        owner_binding_required=owner_binding_required,
        reference_binding_required=reference_binding_required,
        direction_binding_required=direction_binding_required,
        negative_evidence_coverage_required=negative_coverage,
        route_id=route_id,
        taxonomy_path=taxonomy_path,
        query_intents=query_intents,
        source_fields=source_fields,
    )


def attach_evidence_demand_to_ir(
    question: str,
    question_ir: dict[str, Any] | None,
    *,
    question_route: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(question_ir or {})
    updated["evidence_demand"] = compile_evidence_demand(
        question,
        question_ir=updated,
        question_route=question_route,
    ).to_dict()
    return updated


def visual_query_intents(
    evidence_demand: EvidenceDemand | dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if isinstance(evidence_demand, EvidenceDemand):
        return [intent.to_dict() for intent in evidence_demand.query_intents]
    if not isinstance(evidence_demand, dict):
        return []
    rows = evidence_demand.get("query_intents")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict) and _clean_text(row.get("text"))]


def _question_route(
    ir: dict[str, Any],
    explicit_route: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(explicit_route, dict):
        return explicit_route
    route = ir.get("question_route")
    if isinstance(route, dict):
        return route
    routing = ir.get("routing")
    return routing if isinstance(routing, dict) else {}


def _evidence_units(
    *,
    route_id: str,
    taxonomy_path: str,
    aggregation: str,
) -> tuple[str, ...]:
    semantic = f"{route_id} {taxonomy_path}".lower()
    units = [EVIDENCE_UNIT_GRAPH_FACT]
    if any(token in semantic for token in ("temporal", "ego_motion", "ego.direction")):
        units.extend((EVIDENCE_UNIT_FRAME_WINDOW, EVIDENCE_UNIT_FULL_FRAME))
    elif any(
        token in semantic
        for token in (
            "spatial",
            "state.scene",
            "functional",
            "world_knowledge",
            "open_descriptive",
        )
    ):
        units.extend((EVIDENCE_UNIT_OBJECT_CROP, EVIDENCE_UNIT_FULL_FRAME))
    else:
        units.append(EVIDENCE_UNIT_OBJECT_CROP)
    if aggregation in {AGGREGATION_COUNT, AGGREGATION_COVERAGE} or route_id == "binary_existence":
        units.append(EVIDENCE_UNIT_FULL_FRAME)
    return _ordered_unique(units)


def _aggregation(
    ir: dict[str, Any],
    *,
    route_id: str,
    taxonomy_path: str,
) -> str:
    raw = _normalize(ir.get("aggregation"))
    if route_id == "count_set" or raw in {"count", "count distinct", "count_distinct"}:
        return AGGREGATION_COUNT
    if route_id == "temporal_interval" or raw == "interval":
        return AGGREGATION_INTERVAL
    if raw in {"all", "coverage", "multiple", "set", "set valued", "set_valued"}:
        return AGGREGATION_SET
    if ".count" in taxonomy_path:
        return AGGREGATION_COUNT
    return AGGREGATION_SINGLE


def _requires_negative_coverage(ir: dict[str, Any], *, route_id: str) -> bool:
    operation = _normalize(ir.get("operation"))
    answer_type = _normalize(ir.get("answer_type"))
    return route_id == "binary_existence" or (
        answer_type == "binary"
        and operation in {"current state", "current_state", "existence query", "existence_query"}
    )


def _build_query_intents(
    *,
    primary_targets: tuple[str, ...],
    descriptors: tuple[str, ...],
    references: tuple[str, ...],
    relations: tuple[str, ...],
    evidence_units: tuple[str, ...],
) -> tuple[VisualQueryIntent, ...]:
    visual_units = [unit for unit in evidence_units if unit != EVIDENCE_UNIT_GRAPH_FACT]
    if not visual_units:
        return ()
    primary_unit = (
        EVIDENCE_UNIT_OBJECT_CROP
        if EVIDENCE_UNIT_OBJECT_CROP in visual_units
        else visual_units[0]
    )
    descriptor_text = " ".join(descriptors[:3]).strip()
    intents: list[VisualQueryIntent] = []
    for target in primary_targets[:4]:
        target_text = target
        if descriptor_text and not _descriptor_already_present(descriptor_text, target):
            target_text = f"{descriptor_text} {target}".strip()
        intents.append(
            VisualQueryIntent(
                text=target_text,
                role="primary_target",
                evidence_unit=primary_unit,
                owner_terms=(target,),
            )
        )
    if not intents and descriptor_text:
        intents.append(
            VisualQueryIntent(
                text=descriptor_text,
                role="primary_target",
                evidence_unit=primary_unit,
            )
        )
    relation = relations[0] if relations else ""
    for reference in references[:4]:
        intents.append(
            VisualQueryIntent(
                text=reference,
                role="reference_entity",
                evidence_unit=primary_unit,
                relation=relation,
            )
        )
    if EVIDENCE_UNIT_FULL_FRAME in visual_units:
        context_terms = _ordered_unique([*primary_targets[:2], *references[:2]])
        if context_terms:
            intents.append(
                VisualQueryIntent(
                    text=" near ".join(context_terms),
                    role="scene_context",
                    evidence_unit=EVIDENCE_UNIT_FULL_FRAME,
                    relation=relation,
                )
            )
    return _dedupe_intents(intents)


def _spatial_bindings(value: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    references: list[str] = []
    relations: list[str] = []
    for item in _items(value):
        if isinstance(item, dict):
            relations.extend(_text_values(item.get("relation")))
            references.extend(
                [
                    *_text_values(item.get("target")),
                    *_text_values(item.get("reference")),
                    *_text_values(item.get("reference_entity")),
                ]
            )
            continue
        text = _clean_text(item)
        if text:
            relations.append(text)
    return _ordered_unique(references), _ordered_unique(relations)


def _constraint_targets(value: Any, *, spatial_only: bool) -> tuple[str, ...]:
    targets: list[str] = []
    for item in _items(value):
        if not isinstance(item, dict):
            continue
        if spatial_only and _normalize(item.get("kind")) != "spatial":
            continue
        targets.extend(_text_values(item.get("reference_entities")))
        targets.extend(_text_values(item.get("reference")))
    return _ordered_unique(targets)


def _attribute_constraint_texts(value: Any) -> tuple[str, ...]:
    results: list[str] = []
    for item in _items(value):
        if not isinstance(item, dict):
            continue
        kind = _normalize(item.get("kind"))
        role = _normalize(item.get("role"))
        if kind in {"attribute", "visual descriptor", "visual_descriptor"} or role == "target modifier":
            results.extend(_text_values(item.get("text")))
    return _ordered_unique(results)


def _target_state_descriptors(value: Any, targets: Iterable[str]) -> tuple[str, ...]:
    state = _clean_text(value)
    if not state:
        return ()
    state_norm = _normalize(state)
    if any(_normalize(target) == state_norm for target in targets):
        return ()
    return (state,)


def _constraint_texts(value: Any) -> tuple[str, ...]:
    texts: list[str] = []
    for item in _items(value):
        if isinstance(item, dict):
            texts.extend(_text_values(item.get("text")))
            texts.extend(_text_values(item.get("relation")))
        else:
            texts.extend(_text_values(item))
    return _ordered_unique(texts)


def _temporal_scope(ir: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    scope: dict[str, Any] = {}
    for key in (
        "temporal_constraints",
        "time_constraints",
        "time_window",
        "reference_time",
    ):
        if ir.get(key) not in (None, "", [], {}):
            scope[key] = ir[key]
    route_time = route.get("time_requirements")
    if isinstance(route_time, dict) and route_time:
        scope["route_time_requirements"] = dict(route_time)
    return scope


def _present_source_fields(
    ir: dict[str, Any],
    route: dict[str, Any],
) -> tuple[str, ...]:
    keys = (
        "target_entities",
        "target_state",
        "target_descriptors",
        "attribute_dimension",
        "spatial_constraints",
        "event_constraints",
        "hard_constraints",
        "aggregation",
        "evidence_question_type",
        "question_route",
    )
    result = [key for key in keys if ir.get(key) not in (None, "", [], {})]
    if route and "question_route" not in result:
        result.append("question_route")
    return tuple(result)


def _descriptor_already_present(descriptor: str, target: str) -> bool:
    descriptor_tokens = set(_TOKEN_PATTERN.findall(_normalize(descriptor)))
    target_tokens = set(_TOKEN_PATTERN.findall(_normalize(target)))
    return bool(descriptor_tokens) and descriptor_tokens.issubset(target_tokens)


def _dedupe_intents(intents: Iterable[VisualQueryIntent]) -> tuple[VisualQueryIntent, ...]:
    result: list[VisualQueryIntent] = []
    seen: set[tuple[str, str, str]] = set()
    for intent in intents:
        key = (_normalize(intent.text), intent.role, intent.evidence_unit)
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(intent)
    return tuple(result)


def _items(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _text_values(item)]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ("text", "target", "name", "value", "relation"):
            values.extend(_text_values(value.get(key)))
        return values
    text = _clean_text(value)
    return [text] if text else []


def _ordered_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        normalized = _normalize(text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    return tuple(result)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip(" .,:;")


def _normalize(value: Any) -> str:
    return _clean_text(value).lower().replace("_", " ")


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value
