from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from vlkg.core.models.canonical_graph import (
    CanonicalEpisodeNode,
    CanonicalTemporalMemoryGraph,
    CanonicalTimeRange,
    CanonicalTransitionEventNode,
)


TOKEN_SPACE = re.compile(r"\s+")


def build_compressed_fact_layer(
    graph: CanonicalTemporalMemoryGraph,
    *,
    source_graph_path: str | Path | None = None,
    sample_limit: int = 16,
    include_direct_entity_attribute_facts: bool = False,
) -> dict[str, Any]:
    """Build a read-only fact projection from a canonical graph.

    The projection reduces retrieval-facing volume by removing topology edges
    and grouping repeated fact episodes. It does not merge entity identities:
    each entity keeps its own entity-profile fact, and episode grouping requires
    the same bound entity ids.
    """

    resolved_sample_limit = max(1, int(sample_limit))
    entity_labels = {node.entity_id: node.label for node in graph.entity_nodes}
    scene_ids_by_entity = _scene_ids_by_entity(graph)

    facts: list[dict[str, Any]] = []
    facts.extend(_scene_profile_facts(graph, sample_limit=resolved_sample_limit))
    facts.extend(
        _entity_profile_facts(
            graph,
            scene_ids_by_entity=scene_ids_by_entity,
            sample_limit=resolved_sample_limit,
        )
    )
    if include_direct_entity_attribute_facts:
        attribute_facts, attribute_report = _direct_entity_attribute_facts(
            graph,
            scene_ids_by_entity=scene_ids_by_entity,
        )
    else:
        attribute_facts = []
        attribute_report = {
            "enabled": False,
            "entity_count_with_attributes": 0,
            "direct_entity_attribute_fact_count": 0,
            "attribute_count_per_entity_histogram": {},
        }
    facts.extend(attribute_facts)
    episode_facts, episode_report = _episode_fact_span_facts(
        graph.episode_nodes,
        entity_labels=entity_labels,
        sample_limit=resolved_sample_limit,
    )
    facts.extend(episode_facts)
    transition_facts, transition_report = _transition_event_span_facts(
        graph.transition_event_nodes,
        entity_labels=entity_labels,
        sample_limit=resolved_sample_limit,
    )
    facts.extend(transition_facts)

    facts.sort(
        key=lambda item: (
            _sort_optional_number(item.get("time_range", {}).get("start_frame")),
            _sort_optional_number(item.get("time_range", {}).get("start_ts")),
            str(item.get("kind") or ""),
            str(item.get("fact_id") or ""),
        )
    )

    fact_counts = Counter(str(item.get("kind") or "") for item in facts)
    source_size = _source_graph_size(graph)
    layer_size = {
        "fact_count": len(facts),
        "fact_count_by_kind": dict(sorted(fact_counts.items())),
    }
    source_record_count = source_size["node_count"] + source_size["edge_count"]
    reduction_ratio = (
        round(len(facts) / source_record_count, 6) if source_record_count else 0.0
    )
    return {
        "contract": "canonical_compressed_fact_layer_v1",
        "source_graph": str(source_graph_path) if source_graph_path is not None else None,
        "source_video_id": graph.video_id,
        "policy": {
            "purpose": "read_only_retrieval_projection",
            "does_not_modify_source_graph": True,
            "does_not_merge_entities": True,
            "does_not_inspect_questions_or_gold_answers": True,
            "does_not_inspect_retrieval_or_qa_outputs": True,
            "does_not_use_seq_specific_rules": True,
            "does_not_use_answer_specific_rules": True,
            "grouping_rule": (
                "Scene/entity profiles are kept per source node. STATE facts are "
                "grouped only when scene, primary entity, attr key, and attr value "
                "are identical. RELATION facts are grouped only when scene, "
                "subject entity, object entity, predicate, and raw predicate are "
                "identical. Transition events are grouped only when event type "
                "and all bound entity/predicate/state fields are identical."
            ),
            "sample_limit": resolved_sample_limit,
        },
        "source_size": source_size,
        "layer_size": layer_size,
        "compression_estimate": {
            "source_node_plus_edge_records": source_record_count,
            "compressed_fact_records": len(facts),
            "compressed_to_source_record_ratio": reduction_ratio,
            "estimated_record_reduction_percent": (
                round((1.0 - reduction_ratio) * 100.0, 2)
                if source_record_count
                else 0.0
            ),
        },
        "episode_fact_span_report": episode_report,
        "direct_entity_attribute_report": attribute_report,
        "transition_event_span_report": transition_report,
        "facts": facts,
    }


def _scene_profile_facts(
    graph: CanonicalTemporalMemoryGraph,
    *,
    sample_limit: int,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for node in graph.scene_nodes:
        labels = _unique_strings(
            [
                node.label,
                *node.representative_static_labels,
                *node.representative_dynamic_labels,
                *node.entry_static_labels,
                *node.exit_static_labels,
                *node.salient_labels,
            ]
        )
        facts.append(
            {
                "fact_id": f"scene_profile:{_hash_parts([node.node_id])}",
                "kind": "scene_profile",
                "source_node_id": node.node_id,
                "scene_id": node.scene_id,
                "entity_ids": list(node.representative_static_entity_ids),
                "time_range": _time_range_payload(node.time_range),
                "label": node.label,
                "labels": labels,
                "text": _clean_text(
                    " ".join(
                        [
                            f"Scene {node.scene_id} ({node.label}) spans "
                            f"{_time_text(node.time_range)}.",
                            _maybe_join("labels", labels, sample_limit=sample_limit),
                        ]
                    )
                ),
            }
        )
    return facts


def _entity_profile_facts(
    graph: CanonicalTemporalMemoryGraph,
    *,
    scene_ids_by_entity: dict[str, set[str]],
    sample_limit: int,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for node in graph.entity_nodes:
        raw_attrs = _unique_strings(node.attributes.get("raw_visual_attributes") or [])
        labels = _unique_strings(
            [
                node.label,
                *node.labels_seen,
                *node.states_seen,
                *node.parent_labels_seen,
                str(node.attributes.get("canonical_label") or ""),
                str(node.attributes.get("semantic_group") or ""),
                *(node.attributes.get("semantic_alias_terms") or []),
            ]
        )
        scene_ids = sorted(scene_ids_by_entity.get(node.entity_id, set()))
        facts.append(
            {
                "fact_id": f"entity_profile:{_hash_parts([node.node_id])}",
                "kind": "entity_profile",
                "source_node_id": node.node_id,
                "entity_ids": [node.entity_id],
                "stable_identity_id": node.stable_identity_id,
                "stable_identity_display_name": node.stable_identity_display_name,
                "stable_identity_canonical_label": node.stable_identity_canonical_label,
                "scene_ids": scene_ids,
                "time_range": _time_range_payload(node.time_range),
                "label": node.label,
                "node_role": node.node_role,
                "is_static": node.is_static,
                "is_noise": node.is_noise,
                "is_fragment": node.is_fragment,
                "observation_count": node.observation_count,
                "labels": labels,
                "raw_visual_attributes": raw_attrs,
                "canonical_image_path": node.canonical_image_path,
                "text": _clean_text(
                    " ".join(
                        [
                            f"Entity {node.entity_id} ({node.label}) appears "
                            f"during {_time_text(node.time_range)}.",
                            f"role={node.node_role}.",
                            (
                                f"stable_identity={node.stable_identity_display_name} "
                                f"({node.stable_identity_canonical_label})."
                                if node.stable_identity_id
                                else ""
                            ),
                            _maybe_join("labels", labels, sample_limit=sample_limit),
                            _maybe_join(
                                "visual_attributes",
                                raw_attrs,
                                sample_limit=sample_limit,
                            ),
                        ]
                    )
                ),
            }
        )
    return facts


def _episode_fact_span_facts(
    nodes: list[CanonicalEpisodeNode],
    *,
    entity_labels: dict[str, str],
    sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, ...], list[CanonicalEpisodeNode]] = defaultdict(list)
    passthrough_count = 0
    for node in nodes:
        signature = _episode_fact_signature(node)
        if signature is None:
            passthrough_count += 1
            continue
        groups[signature].append(node)

    facts: list[dict[str, Any]] = []
    grouped_source_count = 0
    source_count_by_kind: Counter[str] = Counter()
    group_count_by_kind: Counter[str] = Counter()
    group_size_histogram: Counter[str] = Counter()

    for signature, source_nodes in sorted(groups.items(), key=lambda item: item[0]):
        ordered = sorted(source_nodes, key=_episode_sort_key)
        first = ordered[0]
        grouped_source_count += len(ordered)
        source_count_by_kind[first.episode_kind] += len(ordered)
        group_count_by_kind[first.episode_kind] += 1
        group_size_histogram[_histogram_bucket(len(ordered))] += 1
        time_range = _span_time_ranges([node.time_range for node in ordered])
        source_node_ids = [node.node_id for node in ordered]
        source_ref_ids = [node.source_ref_id for node in ordered]
        fact = {
            "fact_id": f"episode_fact_span:{_hash_parts(signature)}",
            "kind": "episode_fact_span",
            "episode_kind": first.episode_kind,
            "scene_id": first.scene_id,
            "entity_ids": _unique_strings(
                [
                    *first.participant_entity_ids,
                    first.primary_entity_id or "",
                    first.subject_entity_id or "",
                    first.object_entity_id or "",
                ]
            ),
            "primary_entity_id": first.primary_entity_id,
            "subject_entity_id": first.subject_entity_id,
            "object_entity_id": first.object_entity_id,
            "primary_entity_label": entity_labels.get(first.primary_entity_id or ""),
            "subject_entity_label": entity_labels.get(first.subject_entity_id or ""),
            "object_entity_label": entity_labels.get(first.object_entity_id or ""),
            "attr_key": first.attr_key,
            "attr_value": first.attr_value,
            "predicate": first.predicate,
            "raw_predicate": first.raw_predicate,
            "time_range": _time_range_payload(time_range),
            "source_episode_count": len(ordered),
            "source_node_id_samples": _sample_values(source_node_ids, sample_limit),
            "source_ref_id_samples": _sample_values(source_ref_ids, sample_limit),
        }
        fact["text"] = _episode_fact_text(fact)
        facts.append(fact)

    return (
        facts,
        {
            "input_episode_nodes": len(nodes),
            "non_fact_passthrough_episode_count": passthrough_count,
            "source_episode_count_in_fact_spans": grouped_source_count,
            "fact_span_count": len(facts),
            "fact_span_count_by_kind": dict(sorted(group_count_by_kind.items())),
            "source_episode_count_by_kind": dict(sorted(source_count_by_kind.items())),
            "fact_span_size_histogram": dict(sorted(group_size_histogram.items())),
        },
    )


def _direct_entity_attribute_facts(
    graph: CanonicalTemporalMemoryGraph,
    *,
    scene_ids_by_entity: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    attribute_count_by_entity = Counter()
    for node in graph.entity_nodes:
        raw_attrs = _unique_strings(node.attributes.get("raw_visual_attributes") or [])
        if not raw_attrs:
            continue
        labels = _unique_strings(
            [
                node.label,
                *node.labels_seen,
                str(node.attributes.get("canonical_label") or ""),
                str(node.attributes.get("semantic_group") or ""),
                *(node.attributes.get("semantic_alias_terms") or []),
            ]
        )
        scene_ids = sorted(scene_ids_by_entity.get(node.entity_id, set()))
        for index, attribute_value in enumerate(raw_attrs, start=1):
            facts.append(
                {
                    "fact_id": (
                        "direct_entity_attribute:"
                        f"{_hash_parts([node.node_id, index, attribute_value])}"
                    ),
                    "kind": "direct_entity_attribute",
                    "source_node_id": node.node_id,
                    "entity_ids": [node.entity_id],
                    "entity_id": node.entity_id,
                    "stable_identity_id": node.stable_identity_id,
                    "stable_identity_display_name": node.stable_identity_display_name,
                    "stable_identity_canonical_label": (
                        node.stable_identity_canonical_label
                    ),
                    "scene_ids": scene_ids,
                    "time_range": _time_range_payload(node.time_range),
                    "label": node.label,
                    "labels": labels,
                    "attr_key": "visual_attribute",
                    "attr_value": attribute_value,
                    "raw_visual_attribute": attribute_value,
                    "canonical_image_path": node.canonical_image_path,
                    "text": _clean_text(
                        " ".join(
                            [
                                "Direct entity attribute:",
                                f"entity {node.entity_id} ({node.label})",
                                f"visual_attribute={attribute_value}",
                                f"during {_time_text(node.time_range)}.",
                            ]
                        )
                    ),
                }
            )
        attribute_count_by_entity[len(raw_attrs)] += 1
    return (
        facts,
        {
            "enabled": True,
            "entity_count_with_attributes": sum(attribute_count_by_entity.values()),
            "direct_entity_attribute_fact_count": len(facts),
            "attribute_count_per_entity_histogram": dict(
                sorted(
                    (str(key), value)
                    for key, value in attribute_count_by_entity.items()
                )
            ),
        },
    )


def _transition_event_span_facts(
    nodes: list[CanonicalTransitionEventNode],
    *,
    entity_labels: dict[str, str],
    sample_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, ...], list[CanonicalTransitionEventNode]] = defaultdict(list)
    for node in nodes:
        groups[_transition_signature(node)].append(node)

    facts: list[dict[str, Any]] = []
    event_count_by_type: Counter[str] = Counter()
    group_size_histogram: Counter[str] = Counter()
    for signature, source_nodes in sorted(groups.items(), key=lambda item: item[0]):
        ordered = sorted(source_nodes, key=_transition_sort_key)
        first = ordered[0]
        event_count_by_type[first.event_type] += len(ordered)
        group_size_histogram[_histogram_bucket(len(ordered))] += 1
        time_range = _span_time_ranges([node.time_range for node in ordered])
        source_node_ids = [node.node_id for node in ordered]
        source_ref_ids = [node.source_ref_id for node in ordered]
        fact = {
            "fact_id": f"transition_event_span:{_hash_parts(signature)}",
            "kind": "transition_event_span",
            "event_type": first.event_type,
            "scene_id": first.scene_id,
            "entity_ids": _unique_strings(
                [
                    first.entity_id or "",
                    first.subject_entity_id or "",
                    first.object_entity_id or "",
                ]
            ),
            "entity_id": first.entity_id,
            "subject_entity_id": first.subject_entity_id,
            "object_entity_id": first.object_entity_id,
            "entity_label": entity_labels.get(first.entity_id or ""),
            "subject_entity_label": entity_labels.get(first.subject_entity_id or ""),
            "object_entity_label": entity_labels.get(first.object_entity_id or ""),
            "predicate": first.predicate,
            "raw_predicate": first.raw_predicate,
            "attr_key": first.attr_key,
            "previous_state_value": first.previous_state_value,
            "current_state_value": first.current_state_value,
            "time_range": _time_range_payload(time_range),
            "source_event_count": len(ordered),
            "source_node_id_samples": _sample_values(source_node_ids, sample_limit),
            "source_ref_id_samples": _sample_values(source_ref_ids, sample_limit),
        }
        fact["text"] = _transition_fact_text(fact)
        facts.append(fact)

    return (
        facts,
        {
            "input_transition_event_nodes": len(nodes),
            "transition_event_span_count": len(facts),
            "source_event_count_by_type": dict(sorted(event_count_by_type.items())),
            "transition_span_size_histogram": dict(sorted(group_size_histogram.items())),
        },
    )


def _episode_fact_signature(node: CanonicalEpisodeNode) -> tuple[str, ...] | None:
    if node.episode_kind == "STATE":
        if not node.scene_id or not node.primary_entity_id:
            return None
        if not node.attr_key or not node.attr_value:
            return None
        return (
            node.episode_kind,
            node.scene_id,
            node.primary_entity_id,
            node.attr_key,
            node.attr_value,
        )
    if node.episode_kind == "RELATION":
        if not node.scene_id or not node.subject_entity_id or not node.object_entity_id:
            return None
        if not node.predicate and not node.raw_predicate:
            return None
        return (
            node.episode_kind,
            node.scene_id,
            node.subject_entity_id,
            node.object_entity_id,
            node.predicate or "",
            node.raw_predicate or "",
        )
    return None


def _transition_signature(node: CanonicalTransitionEventNode) -> tuple[str, ...]:
    return (
        node.event_type,
        node.scene_id or "",
        node.entity_id or "",
        node.subject_entity_id or "",
        node.object_entity_id or "",
        node.predicate or "",
        node.raw_predicate or "",
        node.attr_key or "",
        node.previous_state_value or "",
        node.current_state_value or "",
    )


def _scene_ids_by_entity(graph: CanonicalTemporalMemoryGraph) -> dict[str, set[str]]:
    by_entity: dict[str, set[str]] = defaultdict(set)
    scene_by_node_id = {node.node_id: node.scene_id for node in graph.scene_nodes}
    entity_by_node_id = {node.node_id: node.entity_id for node in graph.entity_nodes}
    for edge in graph.edges:
        if edge.edge_type != "IN_SCENE":
            continue
        entity_id = entity_by_node_id.get(edge.source_node_id)
        scene_id = scene_by_node_id.get(edge.target_node_id)
        if entity_id and scene_id:
            by_entity[entity_id].add(scene_id)
    for node in graph.episode_nodes:
        if not node.scene_id:
            continue
        for entity_id in [
            *node.participant_entity_ids,
            node.primary_entity_id,
            node.subject_entity_id,
            node.object_entity_id,
        ]:
            if entity_id:
                by_entity[entity_id].add(node.scene_id)
    for node in graph.transition_event_nodes:
        if not node.scene_id:
            continue
        for entity_id in (node.entity_id, node.subject_entity_id, node.object_entity_id):
            if entity_id:
                by_entity[entity_id].add(node.scene_id)
    return by_entity


def _episode_fact_text(fact: dict[str, Any]) -> str:
    kind = str(fact.get("episode_kind") or "")
    time_text = _time_text_from_payload(fact.get("time_range") or {})
    if kind == "STATE":
        return _clean_text(
            " ".join(
                [
                    "State fact:",
                    f"entity {fact.get('primary_entity_id')} "
                    f"({fact.get('primary_entity_label')})",
                    f"{fact.get('attr_key')}={fact.get('attr_value')}",
                    f"in {fact.get('scene_id')} during {time_text}.",
                    f"source_count={fact.get('source_episode_count')}.",
                ]
            )
        )
    return _clean_text(
        " ".join(
            [
                "Relation fact:",
                f"subject {fact.get('subject_entity_id')} "
                f"({fact.get('subject_entity_label')})",
                str(fact.get("predicate") or fact.get("raw_predicate") or "RELATION"),
                f"object {fact.get('object_entity_id')} "
                f"({fact.get('object_entity_label')})",
                f"in {fact.get('scene_id')} during {time_text}.",
                f"source_count={fact.get('source_episode_count')}.",
            ]
        )
    )


def _transition_fact_text(fact: dict[str, Any]) -> str:
    time_text = _time_text_from_payload(fact.get("time_range") or {})
    relation = ""
    if fact.get("subject_entity_id") or fact.get("object_entity_id"):
        relation = (
            f"subject {fact.get('subject_entity_id')} "
            f"({fact.get('subject_entity_label')}) "
            f"{fact.get('predicate') or fact.get('raw_predicate') or ''} "
            f"object {fact.get('object_entity_id')} "
            f"({fact.get('object_entity_label')})"
        )
    entity = ""
    if fact.get("entity_id"):
        entity = f"entity {fact.get('entity_id')} ({fact.get('entity_label')})"
    state = ""
    if fact.get("attr_key"):
        state = (
            f"{fact.get('attr_key')} changed from "
            f"{fact.get('previous_state_value')} to {fact.get('current_state_value')}"
        )
    return _clean_text(
        " ".join(
            [
                "Transition event:",
                str(fact.get("event_type") or ""),
                entity,
                relation,
                state,
                f"in {fact.get('scene_id')} during {time_text}.",
                f"source_count={fact.get('source_event_count')}.",
            ]
        )
    )


def _source_graph_size(graph: CanonicalTemporalMemoryGraph) -> dict[str, int]:
    return {
        "node_count": graph.node_count,
        "edge_count": graph.edge_count,
        "scene_node_count": len(graph.scene_nodes),
        "entity_node_count": len(graph.entity_nodes),
        "stable_identity_node_count": len(graph.stable_identity_nodes),
        "state_value_node_count": len(graph.state_value_nodes),
        "interaction_type_node_count": len(graph.interaction_type_nodes),
        "episode_node_count": len(graph.episode_nodes),
        "transition_event_node_count": len(graph.transition_event_nodes),
        "relation_observed_sample_count": len(graph.relation_observed_samples),
    }


def _time_range_payload(time_range: CanonicalTimeRange) -> dict[str, Any]:
    return {
        "start_frame": time_range.start_frame,
        "end_frame": time_range.end_frame,
        "start_ts": time_range.start_ts,
        "end_ts": time_range.end_ts,
        "support_frames": time_range.support_frames,
    }


def _span_time_ranges(time_ranges: Iterable[CanonicalTimeRange]) -> CanonicalTimeRange:
    items = list(time_ranges)
    if not items:
        return CanonicalTimeRange()
    return CanonicalTimeRange(
        start_frame=_min_optional(item.start_frame for item in items),
        end_frame=_max_optional(item.end_frame for item in items),
        start_ts=_min_optional(item.start_ts for item in items),
        end_ts=_max_optional(item.end_ts for item in items),
        support_frames=sum(
            int(item.support_frames)
            for item in items
            if item.support_frames is not None
        )
        or None,
    )


def _time_text(time_range: CanonicalTimeRange) -> str:
    return _time_text_from_payload(_time_range_payload(time_range))


def _time_text_from_payload(payload: dict[str, Any]) -> str:
    start_frame = payload.get("start_frame")
    end_frame = payload.get("end_frame")
    start_ts = payload.get("start_ts")
    end_ts = payload.get("end_ts")
    frame_text = (
        f"frames {start_frame}-{end_frame}"
        if start_frame != end_frame
        else f"frame {start_frame}"
    )
    if start_ts is None and end_ts is None:
        return frame_text
    ts_text = (
        f"{start_ts:.3f}s-{end_ts:.3f}s"
        if isinstance(start_ts, (int, float)) and isinstance(end_ts, (int, float))
        else f"{start_ts}-{end_ts}s"
    )
    return f"{frame_text}, {ts_text}"


def _maybe_join(label: str, values: list[str], *, sample_limit: int) -> str:
    if not values:
        return ""
    shown = _sample_values(values, sample_limit)
    suffix = "" if len(shown) == len(values) else f" (+{len(values) - len(shown)} more)"
    return f"{label}: {' | '.join(shown)}{suffix}."


def _sample_values(values: list[Any], sample_limit: int) -> list[Any]:
    if len(values) <= sample_limit:
        return list(values)
    if sample_limit <= 2:
        return values[:sample_limit]
    head_count = sample_limit // 2
    tail_count = sample_limit - head_count
    return [*values[:head_count], *values[-tail_count:]]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(str(value or ""))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _clean_text(value: str) -> str:
    return TOKEN_SPACE.sub(" ", value).strip()


def _hash_parts(parts: Iterable[Any]) -> str:
    text = "||".join(str(part or "") for part in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _episode_sort_key(node: CanonicalEpisodeNode) -> tuple[float, float, str]:
    return (
        _sort_optional_number(node.time_range.start_frame),
        _sort_optional_number(node.time_range.start_ts),
        node.node_id,
    )


def _transition_sort_key(node: CanonicalTransitionEventNode) -> tuple[float, float, str]:
    return (
        _sort_optional_number(node.time_range.start_frame),
        _sort_optional_number(node.time_range.start_ts),
        node.node_id,
    )


def _sort_optional_number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else float("inf")


def _min_optional(values: Iterable[Any]) -> Any:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _max_optional(values: Iterable[Any]) -> Any:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _histogram_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 50:
        return "26-50"
    return "51+"
