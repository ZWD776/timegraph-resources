"""OpenEQA adapter for its NAVQA-equivalent visual runtime v1.

The dataset adapter translates OpenEQA evidence demand and episode-local
visual hits into the contracts consumed by the OpenEQA-owned parity fork.  It
does not import the NAVQA strategy runtime, change graph retrieval, or mutate
the canonical graph.
"""
from __future__ import annotations
from copy import deepcopy
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.evidence_router import canonical_route_id_for_need, NEED_COUNT_SET, NEED_ENTITY_ATTRIBUTE, NEED_EXISTENCE_OR_STATE, NEED_FUNCTIONAL_WORLD, NEED_OBJECT_IDENTITY, NEED_OCR_LOGO_TEXT, NEED_OPEN_DESCRIPTIVE, NEED_SPATIAL_LOCALIZATION
from vlkg.datasets.openeqa.runtime.visual_runtime_v1 import OPENEQA_VISUAL_RUNTIME_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.question_route_hybrid_backbone import fuse_question_route_hybrid_backbone, question_route_backbone_policy
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.route_contracts import ROUTE_BINARY_EXISTENCE, ROUTE_COUNT_SET, ROUTE_ENTITY_ATTRIBUTE, ROUTE_ENTITY_RELATION_SPATIAL, ROUTE_OCR_LOGO_TEXT, ROUTE_OPEN_DESCRIPTIVE
from vlkg.datasets.openeqa.runtime.visual_runtime_v1.visual_faiss_grounding import VisualFaissConfig
OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT = 'openeqa_navqa_equivalent_visual_backbone.v1'

def navqa_route_id_for_evidence_demand(evidence_demand: dict[str, Any]) -> str:
    """Map an OpenEQA evidence need to the equivalent NAVQA evidence route."""
    need = str(evidence_demand.get('primary_need') or '').strip()
    return canonical_route_id_for_need(need)

def navqa_equivalent_question_ir(question_ir: dict[str, Any] | None, *, evidence_demand: dict[str, Any]) -> dict[str, Any]:
    """Expose OpenEQA demand fields through the NAVQA-equivalent IR contract."""
    updated = deepcopy(question_ir or {})
    route_id = navqa_route_id_for_evidence_demand(evidence_demand)
    targets = _clean_texts(evidence_demand.get('target_terms'))
    anchors = _clean_texts(evidence_demand.get('anchor_terms'))
    attributes = _clean_texts(evidence_demand.get('attribute_dimensions'))
    states = _clean_texts(evidence_demand.get('state_terms'))
    constraints = _clean_texts(evidence_demand.get('constraint_terms'))
    spatial_constraints = _spatial_constraints(evidence_demand)
    spatial_relations = _clean_texts(evidence_demand.get('spatial_relations'))
    updated['answer_type'] = str(evidence_demand.get('answer_shape') or updated.get('answer_type') or 'text')
    updated['target_entities'] = targets
    if attributes:
        updated['attribute_dimension'] = attributes
    if states:
        updated['target_state'] = states
    if constraints:
        updated['hard_constraints'] = [{'kind': 'visual', 'text': value} for value in constraints]
    if spatial_constraints:
        updated['spatial_constraints'] = spatial_constraints
    if route_id == ROUTE_COUNT_SET:
        updated['aggregation'] = 'count'
    route = deepcopy(evidence_demand)
    route['route_id'] = route_id
    route['answer_shape'] = updated['answer_type']
    route['primary_target_terms'] = targets
    route['attribute_terms'] = attributes
    route['event_constraints'] = list(route.get('event_constraints') or [])
    route['spatial_constraints'] = spatial_relations
    route['time_requirements'] = {}
    route['source'] = str(route.get('source') or 'openeqa_dataset_adapter')
    updated['question_route'] = route
    updated['openeqa_visual_adapter'] = {'contract': OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT, 'navqa_route_id': route_id, 'temporal_route_enabled': False, 'uses_gold_answer': False}
    return updated

def adapt_visual_payload_for_navqa_backbone(visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any], max_lines: int=8) -> dict[str, Any]:
    """Translate OpenEQA object hits into the NAVQA visual payload schema."""
    entity_by_id = {str(node.entity_id): node for node in graph.entity_nodes}
    entity_labels = {str(node.entity_id): str(node.label or '') for node in graph.entity_nodes}
    direct_facts = _direct_facts_by_entity(graph, entity_labels=entity_labels)
    frame_rows = _visual_rows_by_frame(visual_payload)
    groups: list[dict[str, Any]] = []
    phrase_records: list[dict[str, Any]] = []
    rendered_lines: list[str] = []
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        intent = group.get('query_intent')
        intent = intent if isinstance(intent, dict) else {}
        phrase = _clean_text(intent.get('text') or group.get('rendered_query'))
        if not phrase:
            continue
        phrase_role = _navqa_phrase_role(intent)
        results: list[dict[str, Any]] = []
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            row = _adapt_hit(hit, episode_history=str(visual_payload.get('episode_history') or ''), entity_by_id=entity_by_id, direct_facts=direct_facts, frame_rows=frame_rows)
            results.append(row)
            if len(rendered_lines) < max(1, int(max_lines)):
                rendered_lines.append(_visual_hit_line(phrase=phrase, row=row))
        if not results:
            continue
        groups.append({'query_phrase': phrase, 'query_phrase_text': phrase, 'query_phrase_role': phrase_role, 'query_phrase_source': str(intent.get('source') or 'openeqa_demand'), 'results': results})
        phrase_records.append({'text': phrase, 'role': phrase_role, 'source': str(intent.get('source') or 'openeqa_demand')})
    return {'contract': 'openeqa_to_navqa_visual_payload.v1', 'question': question, 'question_ir': question_ir, 'seq_id': str(visual_payload.get('episode_history') or ''), 'episode_history': str(visual_payload.get('episode_history') or ''), 'episode_scoped_search': bool(visual_payload.get('episode_scoped_search')), 'uses_gold_answers': False, 'uses_question_type_branches': True, 'uses_one_persisted_question_route': True, 'query_phrase_records': phrase_records, 'groups': groups, 'lines': _ordered_unique(rendered_lines), 'source_search_contract': visual_payload.get('contract'), 'source_stats': _source_visual_stats(visual_payload)}

def fuse_navqa_equivalent_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Run the isolated OpenEQA v1 evidence builders through the adapter."""
    adapted_ir = navqa_equivalent_question_ir(question_ir, evidence_demand=evidence_demand)
    adapted_payload = adapt_visual_payload_for_navqa_backbone(visual_payload, graph=graph, question=question, question_ir=adapted_ir, max_lines=max_visual_lines)
    route = dict(adapted_ir['question_route'])
    policy = question_route_backbone_policy(route, specialized=True)
    config = VisualFaissConfig(max_lines=max(1, int(max_visual_lines)), ger_object_limit=max(1, int(max_candidate_packs)), ger_context_limit=max(1, int(max_candidate_packs)), local_subgraph_neighbor_limit=6)
    fused = fuse_question_route_hybrid_backbone(deepcopy(graph_result), adapted_payload, graph=graph, config=config, question=question, question_ir=adapted_ir, semantic_text_encoder=semantic_text_encoder, specialized=True, retrieval_window=None)
    return _attach_openeqa_visual_contract(fused, source_payload=visual_payload, adapted_payload=adapted_payload, evidence_demand=evidence_demand, policy=policy, max_visual_lines=max_visual_lines)

def _attach_openeqa_visual_contract(graph_result: dict[str, Any], *, source_payload: dict[str, Any], adapted_payload: dict[str, Any], evidence_demand: dict[str, Any], policy: dict[str, Any], max_visual_lines: int) -> dict[str, Any]:
    bundle = graph_result.get('llm_ready_bundle')
    if not isinstance(bundle, dict):
        return graph_result
    updated = dict(bundle)
    structured = dict(updated.get('structured_graph_context') or {})
    entity_pack_metadata = updated.get('visual_entity_candidate_packs')
    entity_pack_metadata = entity_pack_metadata if isinstance(entity_pack_metadata, dict) else {}
    count_set = updated.get('question_route_count_set')
    count_set = count_set if isinstance(count_set, dict) else {}
    ger = updated.get('visual_faiss_ger_fusion')
    ger = ger if isinstance(ger, dict) else {}
    object_local = structured.get('visual_faiss_object_local_facts')
    object_local = object_local if isinstance(object_local, list) else []
    bound_facts = structured.get('visual_faiss_ger_bound_facts')
    bound_facts = bound_facts if isinstance(bound_facts, list) else []
    candidate_packs = list(entity_pack_metadata.get('packs', []) or [])
    if not candidate_packs:
        candidate_packs = list(count_set.get('candidate_items', []) or [])
    if not candidate_packs:
        candidate_packs = list(ger.get('contexts', []) or [])
    visual_lines = _route_visual_lines(updated)
    visual_lines = visual_lines[:max(1, int(max_visual_lines))]
    visual_evidence = {'contract': OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT, 'visual_runtime_contract': OPENEQA_VISUAL_RUNTIME_CONTRACT, 'available': bool(visual_lines or candidate_packs), 'episode_history': source_payload.get('episode_history'), 'evidence_demand': deepcopy(evidence_demand), 'navqa_route_id': str((updated.get('question_route_hybrid_backbone') or {}).get('question_route_id') or ''), 'route_policy': dict(policy), 'candidate_packs': candidate_packs, 'object_local_facts': object_local, 'bound_facts': bound_facts, 'spatial_pairs': [], 'evidence_lines': visual_lines, 'stats': {**_source_visual_stats(source_payload), 'candidate_pack_count': len(candidate_packs), 'object_local_fact_count': len(object_local), 'bound_fact_count': len(bound_facts)}, 'episode_scoped_search': bool(source_payload.get('episode_scoped_search')), 'uses_gold_answer': False, 'graph_retrieval_is_route_conditioned': False, 'unbound_observations_are_graph_facts': False, 'adapter_contract': adapted_payload.get('contract')}
    updated['visual_evidence_contract'] = OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT
    updated['visual_evidence'] = visual_evidence
    updated['evidence_demand'] = deepcopy(evidence_demand)
    structured['visual_evidence_contract'] = OPENEQA_NAVQA_EQUIVALENT_BACKBONE_CONTRACT
    structured['visual_evidence'] = visual_evidence
    structured['evidence_demand'] = deepcopy(evidence_demand)
    updated['structured_graph_context'] = structured
    graph_result['llm_ready_bundle'] = updated
    graph_result['visual_evidence'] = visual_evidence
    return graph_result

def _adapt_hit(hit: dict[str, Any], *, episode_history: str, entity_by_id: dict[str, Any], direct_facts: dict[str, list[dict[str, Any]]], frame_rows: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    row = deepcopy(hit)
    entity_id = _clean_text(hit.get('canonical_entity_id'))
    frame = _optional_int(hit.get('graph_frame_index'))
    object_index = _optional_int(hit.get('object_index'))
    row['seq_id'] = episode_history
    row['entity_id'] = entity_id
    row['anchor_entity_id'] = entity_id
    row['score'] = float(hit.get('similarity') or 0.0)
    row['is_graph_bound'] = bool(entity_id)
    row['unbound_visual_observation_is_graph_fact'] = False
    row['hsgg_visual_facts'] = _facts_at_frame(direct_facts.get(entity_id, []), frame=frame)[:12]
    row['hsgg_frame_context'] = _same_frame_context(frame_rows.get(frame, []) if frame is not None else [], object_index=object_index, entity_id=entity_id)
    row['hsgg_neighbor_attributes'] = _neighbor_attributes(frame_rows.get(frame, []) if frame is not None else [], object_index=object_index, entity_id=entity_id, frame=frame, entity_by_id=entity_by_id, direct_facts=direct_facts)
    return row

def _direct_facts_by_entity(graph: CanonicalTemporalMemoryGraph, *, entity_labels: dict[str, str]) -> dict[str, list[dict[str, Any]]]:
    facts: dict[str, list[dict[str, Any]]] = {}

    def add(entity_id: str, value: str, *, start_frame: int | None=None, end_frame: int | None=None) -> None:
        if entity_id and value:
            facts.setdefault(entity_id, []).append({'text': value, 'start_frame': start_frame, 'end_frame': end_frame})
    for entity in graph.entity_nodes:
        entity_id = str(entity.entity_id)
        label = str(entity.label or 'object')
        for value in _entity_attribute_values(entity):
            add(entity_id, f'direct_object_attribute owner=entity={entity_id} label={label!r} attr={value}')
    for episode in graph.episode_nodes:
        kind = str(episode.episode_kind or '').upper()
        if kind == 'STATE' and episode.primary_entity_id:
            entity_id = str(episode.primary_entity_id)
            label = entity_labels.get(entity_id, 'object')
            value = '='.join((item for item in (str(episode.attr_key or ''), str(episode.attr_value or '')) if item))
            add(entity_id, f'direct_object_attribute owner=entity={entity_id} label={label!r} attr={value}', start_frame=_optional_int(episode.time_range.start_frame), end_frame=_optional_int(episode.time_range.end_frame))
        elif kind == 'RELATION':
            subject_id = str(episode.subject_entity_id or '')
            object_id = str(episode.object_entity_id or '')
            predicate = str(episode.raw_predicate or episode.predicate or 'related_to')
            subject_label = entity_labels.get(subject_id, subject_id)
            object_label = entity_labels.get(object_id, object_id)
            relation = f'{subject_label} {predicate} {object_label}'.strip()
            for entity_id in (subject_id, object_id):
                add(entity_id, f'object entity={entity_id} relation={relation}', start_frame=_optional_int(episode.time_range.start_frame), end_frame=_optional_int(episode.time_range.end_frame))
    return {key: _unique_fact_rows(values) for key, values in facts.items()}

def _entity_attribute_values(entity: Any) -> list[str]:
    values: list[str] = []
    values.extend(_flatten_text(getattr(entity, 'states_seen', None)))
    attributes = getattr(entity, 'attributes', None)
    if isinstance(attributes, dict):
        for key, value in attributes.items():
            if not _is_visual_attribute_key(str(key)):
                continue
            for item in _flatten_text(value):
                values.append(f'{key}={item}')
    return _ordered_unique(values)[:12]

def _visual_rows_by_frame(visual_payload: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    rows: dict[int, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        for hit in group.get('hits', []) or []:
            if not isinstance(hit, dict):
                continue
            frame = _optional_int(hit.get('graph_frame_index'))
            if frame is None:
                continue
            key = (_optional_int(hit.get('object_index')), _clean_text(hit.get('observation_id')), _optional_int(hit.get('faiss_id')))
            rows.setdefault(frame, {})[key] = hit
    return {frame: list(bucket.values()) for frame, bucket in rows.items()}

def _same_frame_context(rows: list[dict[str, Any]], *, object_index: int | None, entity_id: str) -> list[str]:
    context: list[str] = []
    for row in rows:
        other_index = _optional_int(row.get('object_index'))
        if object_index is not None and other_index == object_index:
            continue
        other_entity_id = _clean_text(row.get('canonical_entity_id'))
        if entity_id and other_entity_id == entity_id:
            continue
        context.append(f"same_frame_anchor obj={other_index} label={_clean_text(row.get('label'))!r} bbox={row.get('bbox')}")
    return _ordered_unique(context)[:8]

def _neighbor_attributes(rows: list[dict[str, Any]], *, object_index: int | None, entity_id: str, frame: int | None, entity_by_id: dict[str, Any], direct_facts: dict[str, list[dict[str, Any]]]) -> list[str]:
    attributes: list[str] = []
    for row in rows:
        other_index = _optional_int(row.get('object_index'))
        if object_index is not None and other_index == object_index:
            continue
        other_entity_id = _clean_text(row.get('canonical_entity_id'))
        if not other_entity_id or other_entity_id == entity_id or other_entity_id not in entity_by_id:
            continue
        for fact in _facts_at_frame(direct_facts.get(other_entity_id, []), frame=frame):
            if ' attr=' not in fact:
                continue
            attributes.append(f"neighbor_object_attribute owner=frame={row.get('graph_frame_index')} obj={other_index} label={_clean_text(row.get('label'))!r}: {fact.split(' attr=', 1)[1]}")
    return _ordered_unique(attributes)[:8]

def _facts_at_frame(rows: list[dict[str, Any]], *, frame: int | None) -> list[str]:
    values: list[str] = []
    for row in rows:
        start = _optional_int(row.get('start_frame'))
        end = _optional_int(row.get('end_frame'))
        if frame is not None and start is not None and (frame < start):
            continue
        if frame is not None and end is not None and (frame > end):
            continue
        text = _clean_text(row.get('text'))
        if text:
            values.append(text)
    return _ordered_unique(values)

def _unique_fact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, int | None]] = set()
    for row in rows:
        key = (_clean_text(row.get('text')).lower(), _optional_int(row.get('start_frame')), _optional_int(row.get('end_frame')))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result

def _is_visual_attribute_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in {'raw_visual_attributes', 'raw_state_values', 'states_seen', 'visual_attributes'}:
        return True
    return any((token in normalized for token in ('appearance', 'brand', 'color', 'colour', 'logo', 'material', 'pattern', 'shape', 'size', 'state', 'text', 'texture')))

def _spatial_constraints(evidence_demand: dict[str, Any]) -> list[dict[str, str]]:
    rows = evidence_demand.get('spatial_constraints')
    result = [dict(row) for row in rows or [] if isinstance(row, dict)]
    if result:
        return result
    relations = _clean_texts(evidence_demand.get('spatial_relations'))
    anchors = _clean_texts(evidence_demand.get('anchor_terms'))
    for relation in relations or ['']:
        for anchor in anchors:
            result.append({'relation': relation, 'target': anchor, 'reference': anchor})
    return result

def _navqa_phrase_role(intent: dict[str, Any]) -> str:
    role = _clean_text(intent.get('role'))
    source = _clean_text(intent.get('source'))
    if source in {'target_attribute_or_state', 'ocr_logo'}:
        return 'target_attribute'
    if role in {'target', 'spatial_target'}:
        return 'primary_target'
    if role == 'owner':
        return 'target_attribute'
    if role == 'candidate_set':
        return 'primary_target'
    if role == 'anchor':
        return 'spatial_constraint'
    if role == 'context':
        return 'support'
    return 'fallback'

def _route_visual_lines(bundle: dict[str, Any]) -> list[str]:
    keys = ('visual_entity_candidate_pack_lines', 'question_route_count_set_lines', 'visual_faiss_object_local_fact_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_ger_evidence_lines', 'visual_faiss_evidence_lines')
    return _ordered_unique((str(line).strip() for key in keys for line in bundle.get(key, []) or [] if str(line).strip()))

def _source_visual_stats(payload: dict[str, Any]) -> dict[str, int]:
    groups = [row for row in payload.get('groups', []) or [] if isinstance(row, dict)]
    hits = [hit for group in groups for hit in group.get('hits', []) or [] if isinstance(hit, dict)]
    stats = {'query_intent_count': int(payload.get('query_intent_count') or len(groups)), 'episode_candidate_count': int(payload.get('episode_candidate_count') or 0), 'hit_count': len(hits), 'graph_bound_candidate_count': sum((1 for hit in hits if _clean_text(hit.get('canonical_entity_id')))), 'unbound_hit_count': sum((1 for hit in hits if not _clean_text(hit.get('canonical_entity_id')))), 'coframe_expansion_hit_count': int(payload.get('coframe_expansion_hit_count') or 0)}
    for unit, count in (payload.get('retrieved_unit_group_counts') or {}).items():
        stats[f'retrieved_unit_{unit}_group_count'] = int(count or 0)
    return stats

def _visual_hit_line(*, phrase: str, row: dict[str, Any]) -> str:
    label = row.get('aggregate_labels') or row.get('label')
    retrieval_kind = row.get('retrieval_kind') or 'object_crop'
    return f"kind=episode_scoped_visual_candidate unit={retrieval_kind} query={phrase!r} label={label!r} frame={row.get('graph_frame_index')} obj={row.get('object_index')} similarity={float(row.get('similarity') or 0.0):.4f} entity={row.get('entity_id') or 'unbound_visual_observation'}"

def _clean_text(value: Any) -> str:
    return ' '.join(str(value or '').split())

def _clean_texts(value: Any) -> list[str]:
    return _ordered_unique(_flatten_text(value))

def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in _flatten_text(f'{key}={item}' if isinstance(item, (str, int, float)) else item)]
    if isinstance(value, Iterable):
        return [text for item in value for text in _flatten_text(item)]
    text = _clean_text(value)
    return [text] if text else []

def _ordered_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result

def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
