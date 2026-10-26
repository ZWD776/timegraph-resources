from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any

from vlkg.core.graph_retrieval.canonical_hybrid_retrieval import (
    PRIORITY_BY_KIND,
    _normalize_index_field,
    _tokenize,
)
from vlkg.core.graph_retrieval.canonical_retrieval_types import (
    CanonicalProjection,
    GraphRecordView,
)


FACT_KIND_TO_SOURCE_KIND = {
    "scene_profile": "scene",
    "entity_profile": "entity",
    "direct_entity_attribute": "entity_attribute",
    "episode_fact_span": "episode",
    "transition_event_span": "transition_event",
}


def build_projection_from_compressed_fact_layer(
    layer: dict[str, Any],
    *,
    include_structural_links: bool = True,
) -> tuple[CanonicalProjection, dict[str, Any]]:
    """Adapt a compressed fact layer into the existing retrieval projection format.

    This is a retrieval-only projection. It does not merge entity ids and does not
    inspect questions, gold answers, retrieval output, or QA results.
    """

    projection = CanonicalProjection()
    video_id = str(layer.get("source_video_id") or "")
    facts = [fact for fact in layer.get("facts", []) if isinstance(fact, dict)]
    record_id_by_fact_id: dict[str, str] = {}
    record_id_by_entity_id: dict[str, str] = {}
    record_id_by_scene_id: dict[str, str] = {}
    used_record_ids: set[str] = set()
    skipped_fact_kinds: Counter[str] = Counter()

    for fact in facts:
        source_kind = FACT_KIND_TO_SOURCE_KIND.get(str(fact.get("kind") or ""))
        if source_kind is None:
            skipped_fact_kinds[str(fact.get("kind") or "")] += 1
            continue
        record_id = _unique_record_id(_preferred_record_id(fact), used_record_ids)
        used_record_ids.add(record_id)
        record = _record_from_fact(fact, source_kind=source_kind, record_id=record_id, video_id=video_id)
        projection.add_record(record)

        fact_id = str(fact.get("fact_id") or "").strip()
        if fact_id:
            record_id_by_fact_id[fact_id] = record_id
        if source_kind == "entity":
            for entity_id in _string_list(fact.get("entity_ids")):
                record_id_by_entity_id.setdefault(entity_id, record_id)
        elif source_kind == "scene":
            scene_id = str(fact.get("scene_id") or "").strip()
            if scene_id:
                record_id_by_scene_id.setdefault(scene_id, record_id)

    if include_structural_links:
        _add_structural_links(
            projection,
            facts=facts,
            record_id_by_fact_id=record_id_by_fact_id,
            record_id_by_entity_id=record_id_by_entity_id,
            record_id_by_scene_id=record_id_by_scene_id,
        )

    fact_counts = Counter(str(fact.get("kind") or "") for fact in facts)
    record_counts = Counter(record.source_kind for record in projection.records.values())
    link_count = sum(len(links) for links in projection.adjacency.values()) // 2
    report = {
        "contract": "canonical_compressed_fact_projection_report_v1",
        "source_contract": layer.get("contract"),
        "source_video_id": video_id,
        "source_fact_count": len(facts),
        "source_fact_count_by_kind": dict(sorted(fact_counts.items())),
        "projection_record_count": len(projection.records),
        "projection_record_count_by_kind": dict(sorted(record_counts.items())),
        "projection_link_count": link_count,
        "skipped_fact_count_by_kind": dict(sorted(skipped_fact_kinds.items())),
        "policy": {
            "read_only_projection": True,
            "does_not_merge_entities": True,
            "does_not_inspect_questions_or_gold_answers": True,
            "does_not_inspect_retrieval_or_qa_outputs": True,
            "source_kind_mapping": dict(FACT_KIND_TO_SOURCE_KIND),
        },
    }
    return projection, report


def _record_from_fact(
    fact: dict[str, Any],
    *,
    source_kind: str,
    record_id: str,
    video_id: str,
) -> GraphRecordView:
    metadata = _metadata_from_fact(fact, source_kind=source_kind, record_id=record_id)
    fields = _index_fields_from_fact(fact, metadata=metadata)
    index_text = " ".join(
        normalized
        for field in fields
        for normalized in (_normalize_index_field(field, video_id=video_id),)
        if normalized
    )
    time_range = _time_range(fact)
    return GraphRecordView(
        source_kind=source_kind,  # type: ignore[arg-type]
        source_record_id=record_id,
        summary=str(fact.get("text") or _fallback_summary(fact)),
        metadata=metadata,
        index_text=index_text,
        tokens=frozenset(_tokenize(index_text)),
        video_id=video_id,
        scene_ids=tuple(_scene_ids(fact)),
        entity_ids=tuple(_string_list(fact.get("entity_ids"))),
        start_ts=_optional_float(time_range.get("start_ts")),
        end_ts=_optional_float(time_range.get("end_ts")),
        priority=PRIORITY_BY_KIND[source_kind],  # type: ignore[index]
    )


def _metadata_from_fact(
    fact: dict[str, Any],
    *,
    source_kind: str,
    record_id: str,
) -> dict[str, Any]:
    time_range = _time_range(fact)
    base = {
        "node_id": record_id,
        "compressed_fact_id": fact.get("fact_id"),
        "compressed_fact_kind": fact.get("kind"),
        "source_node_id": fact.get("source_node_id"),
        "source_node_id_samples": _string_list(fact.get("source_node_id_samples")),
        "source_ref_id_samples": _string_list(fact.get("source_ref_id_samples")),
        "source_ref_id": _first_string(fact.get("source_ref_id_samples")),
        "source_episode_count": fact.get("source_episode_count"),
        "source_event_count": fact.get("source_event_count"),
        "label": fact.get("label") or fact.get("text"),
        "time_range": time_range,
        "projection_source": "compressed_fact_layer_v1",
    }
    if source_kind == "scene":
        base.update(
            {
                "scene_id": fact.get("scene_id"),
                "salient_labels": _string_list(fact.get("labels")),
                "semantic_alias_terms": [],
            }
        )
    elif source_kind == "entity":
        labels = _string_list(fact.get("labels"))
        label = str(fact.get("label") or "").strip()
        base.update(
            {
                "entity_id": _first_string(fact.get("entity_ids")),
                "label": label,
                "node_role": fact.get("node_role"),
                "is_static": fact.get("is_static"),
                "is_noise": fact.get("is_noise"),
                "is_fragment": fact.get("is_fragment"),
                "stable_identity_id": fact.get("stable_identity_id"),
                "stable_identity_display_name": fact.get("stable_identity_display_name"),
                "stable_identity_canonical_label": fact.get("stable_identity_canonical_label"),
                "canonical_label": fact.get("stable_identity_canonical_label") or label,
                "semantic_group": label,
                "semantic_alias_terms": [],
                "labels_seen": labels,
                "states_seen": [],
                "parent_labels_seen": [],
                "raw_visual_attributes": _string_list(fact.get("raw_visual_attributes")),
                "attributes": {
                    "compressed_fact_id": fact.get("fact_id"),
                    "canonical_image_path": fact.get("canonical_image_path"),
                },
            }
        )
    elif source_kind == "entity_attribute":
        label = str(fact.get("label") or "").strip()
        base.update(
            {
                "source_entity_node_id": fact.get("source_node_id"),
                "entity_id": fact.get("entity_id")
                or _first_string(fact.get("entity_ids")),
                "label": label,
                "canonical_label": fact.get("stable_identity_canonical_label") or label,
                "semantic_group": label,
                "semantic_alias_terms": [],
                "stable_identity_id": fact.get("stable_identity_id"),
                "stable_identity_display_name": fact.get(
                    "stable_identity_display_name"
                ),
                "stable_identity_canonical_label": fact.get(
                    "stable_identity_canonical_label"
                ),
                "attr_key": fact.get("attr_key") or "visual_attribute",
                "attr_value": fact.get("attr_value"),
                "raw_visual_attribute": fact.get("raw_visual_attribute")
                or fact.get("attr_value"),
                "attributes": {
                    "compressed_fact_id": fact.get("fact_id"),
                    "canonical_image_path": fact.get("canonical_image_path"),
                },
            }
        )
    elif source_kind == "episode":
        base.update(
            {
                "episode_id": fact.get("fact_id"),
                "episode_kind": fact.get("episode_kind"),
                "scene_id": fact.get("scene_id"),
                "primary_entity_id": fact.get("primary_entity_id"),
                "primary_entity_label": fact.get("primary_entity_label"),
                "subject_entity_id": fact.get("subject_entity_id"),
                "subject_entity_label": fact.get("subject_entity_label"),
                "object_entity_id": fact.get("object_entity_id"),
                "object_entity_label": fact.get("object_entity_label"),
                "participant_entity_ids": _string_list(fact.get("entity_ids")),
                "attr_key": fact.get("attr_key"),
                "attr_value": fact.get("attr_value"),
                "predicate": fact.get("predicate"),
                "raw_predicate": fact.get("raw_predicate"),
                "attributes": {"compressed_source_count": fact.get("source_episode_count")},
            }
        )
    elif source_kind == "transition_event":
        base.update(
            {
                "event_id": fact.get("fact_id"),
                "event_type": fact.get("event_type"),
                "scene_id": fact.get("scene_id"),
                "entity_id": fact.get("entity_id"),
                "entity_label": fact.get("entity_label"),
                "subject_entity_id": fact.get("subject_entity_id"),
                "subject_entity_label": fact.get("subject_entity_label"),
                "object_entity_id": fact.get("object_entity_id"),
                "object_entity_label": fact.get("object_entity_label"),
                "predicate": fact.get("predicate"),
                "raw_predicate": fact.get("raw_predicate"),
                "attr_key": fact.get("attr_key"),
                "previous_state_value": fact.get("previous_state_value"),
                "current_state_value": fact.get("current_state_value"),
            }
        )
    return base


def _index_fields_from_fact(fact: dict[str, Any], *, metadata: dict[str, Any]) -> list[Any]:
    fields: list[Any] = [
        fact.get("fact_id"),
        fact.get("source_node_id"),
        fact.get("text"),
        metadata.get("label"),
        metadata.get("scene_id"),
        metadata.get("entity_id"),
        metadata.get("primary_entity_id"),
        metadata.get("primary_entity_label"),
        metadata.get("subject_entity_id"),
        metadata.get("subject_entity_label"),
        metadata.get("object_entity_id"),
        metadata.get("object_entity_label"),
        metadata.get("event_type"),
        metadata.get("episode_kind"),
        metadata.get("attr_key"),
        metadata.get("attr_value"),
        metadata.get("predicate"),
        metadata.get("raw_predicate"),
        metadata.get("previous_state_value"),
        metadata.get("current_state_value"),
        *(_string_list(fact.get("labels"))),
        *(_string_list(fact.get("raw_visual_attributes"))),
        *(_string_list(fact.get("entity_ids"))),
        *(_string_list(fact.get("source_ref_id_samples"))),
    ]
    return [field for field in fields if str(field or "").strip()]


def _add_structural_links(
    projection: CanonicalProjection,
    *,
    facts: list[dict[str, Any]],
    record_id_by_fact_id: dict[str, str],
    record_id_by_entity_id: dict[str, str],
    record_id_by_scene_id: dict[str, str],
) -> None:
    for fact in facts:
        fact_id = str(fact.get("fact_id") or "").strip()
        record_id = record_id_by_fact_id.get(fact_id)
        if not record_id:
            continue
        for scene_id in _scene_ids(fact):
            scene_record_id = record_id_by_scene_id.get(scene_id)
            if scene_record_id and scene_record_id != record_id:
                projection.add_link(
                    source_record_id=record_id,
                    target_record_id=scene_record_id,
                    edge_type="IN_SCENE",
                    edge_id=f"compressed:IN_SCENE:{record_id}:{scene_record_id}",
                )
        fact_kind = str(fact.get("kind") or "")
        if fact_kind == "direct_entity_attribute":
            _link_entity_to_fact(
                projection,
                record_id=record_id,
                entity_id=str(
                    fact.get("entity_id") or _first_string(fact.get("entity_ids")) or ""
                ),
                record_id_by_entity_id=record_id_by_entity_id,
                edge_type="DESCRIBES_ENTITY",
            )
        elif fact_kind == "episode_fact_span":
            episode_kind = str(fact.get("episode_kind") or "").upper()
            if episode_kind == "STATE":
                _link_entity_to_fact(
                    projection,
                    record_id=record_id,
                    entity_id=str(fact.get("primary_entity_id") or ""),
                    record_id_by_entity_id=record_id_by_entity_id,
                    edge_type="HAS_STATE",
                )
            else:
                for entity_id in _string_list(fact.get("entity_ids")):
                    _link_entity_to_fact(
                        projection,
                        record_id=record_id,
                        entity_id=entity_id,
                        record_id_by_entity_id=record_id_by_entity_id,
                        edge_type="HAS_EPISODE",
                    )
                subject_record_id = record_id_by_entity_id.get(str(fact.get("subject_entity_id") or ""))
                object_record_id = record_id_by_entity_id.get(str(fact.get("object_entity_id") or ""))
                if subject_record_id and object_record_id and subject_record_id != object_record_id:
                    projection.add_link(
                        source_record_id=subject_record_id,
                        target_record_id=object_record_id,
                        edge_type="INTERACTS_WITH",
                        edge_id=f"compressed:INTERACTS_WITH:{record_id}:{subject_record_id}:{object_record_id}",
                    )
        elif fact_kind == "transition_event_span":
            for entity_id in _string_list(fact.get("entity_ids")):
                _link_entity_to_fact(
                    projection,
                    record_id=record_id,
                    entity_id=entity_id,
                    record_id_by_entity_id=record_id_by_entity_id,
                    edge_type="TRIGGERED_BY",
                )


def _link_entity_to_fact(
    projection: CanonicalProjection,
    *,
    record_id: str,
    entity_id: str,
    record_id_by_entity_id: dict[str, str],
    edge_type: str,
) -> None:
    entity_record_id = record_id_by_entity_id.get(str(entity_id or "").strip())
    if not entity_record_id or entity_record_id == record_id:
        return
    projection.add_link(
        source_record_id=entity_record_id,
        target_record_id=record_id,
        edge_type=edge_type,
        edge_id=f"compressed:{edge_type}:{entity_record_id}:{record_id}",
    )


def _preferred_record_id(fact: dict[str, Any]) -> str:
    kind = str(fact.get("kind") or "")
    source_node_id = str(fact.get("source_node_id") or "").strip()
    if kind in {"scene_profile", "entity_profile"} and source_node_id:
        return source_node_id
    source_node_samples = _string_list(fact.get("source_node_id_samples"))
    source_count = int(fact.get("source_episode_count") or fact.get("source_event_count") or 0)
    if source_count == 1 and len(source_node_samples) == 1:
        return source_node_samples[0]
    return str(fact.get("fact_id") or source_node_id or _hash_payload(fact)).strip()


def _unique_record_id(record_id: str, used: set[str]) -> str:
    candidate = record_id or "compressed_fact"
    if candidate not in used:
        return candidate
    digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:10]
    index = 2
    while f"{candidate}:compressed:{digest}:{index}" in used:
        index += 1
    return f"{candidate}:compressed:{digest}:{index}"


def _fallback_summary(fact: dict[str, Any]) -> str:
    return f"{fact.get('kind') or 'fact'} {fact.get('fact_id') or ''}".strip()


def _scene_ids(fact: dict[str, Any]) -> list[str]:
    values = _string_list(fact.get("scene_ids"))
    scene_id = str(fact.get("scene_id") or "").strip()
    if scene_id:
        values.append(scene_id)
    return list(dict.fromkeys(values))


def _time_range(fact: dict[str, Any]) -> dict[str, Any]:
    value = fact.get("time_range")
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item).strip() for item in value if str(item or "").strip()]
    text = str(value).strip()
    return [text] if text else []


def _first_string(value: Any) -> str | None:
    values = _string_list(value)
    return values[0] if values else None


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _hash_payload(value: Any) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:16]
