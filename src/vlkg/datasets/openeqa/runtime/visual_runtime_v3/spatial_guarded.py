"""Conservative OpenEQA spatial evidence augmentation.

The spatial verifier is useful for rejecting impossible directions, but a
verified pair is not necessarily the unique answer. This runtime therefore
preserves the broad v1 evidence and inserts a small number of geometry
candidates after its strongest evidence instead of replacing it.
"""
from __future__ import annotations
from copy import deepcopy
import re
from typing import Any, Callable, Iterable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.openeqa.retrieval.navqa_equivalent_visual_backbone import fuse_navqa_equivalent_visual_evidence
from vlkg.datasets.openeqa.runtime.visual_runtime_v2 import requested_geometry_relations, verify_spatial_pairs
SPATIAL_GUARDED_CONTRACT = 'openeqa_spatial_guarded_augmentation.v1'
SPATIAL_GUARDED_BACKBONE_CONTRACT = 'openeqa_navqa_equivalent_visual_backbone.spatial_guarded_v3'
_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v3'
_PRESERVED_BASELINE_PREFIX = 4
_MAX_VISIBLE_PAIR_LINES = 1
_MIN_UNAMBIGUOUS_SCORE_MARGIN = 0.05
_MIN_OWNER_SCORE_MARGIN = 0.12
_MAX_OWNER_RELATION_FRAME_DISTANCE = 12
_LABEL_PATTERN = re.compile("\\blabel='([^']+)'", re.IGNORECASE)
_TOKEN_PATTERN = re.compile('[a-z0-9]+')
_COMPONENT_LABEL_TOKENS = {'armrest', 'backrest', 'button', 'casing', 'cushion', 'fabric', 'handle', 'hinge', 'knob', 'lampshade', 'lid', 'panel', 'pocket', 'rung', 'seat', 'slat', 'spacebar', 'strap', 'surface', 'zipper'}
_GENERIC_OWNER_LABELS = {'area', 'ceiling', 'floor', 'room', 'scene', 'surface', 'wall'}
_FORWARD_OWNER_RELATIONS = {'attached_to': 0.92, 'belongs_to': 0.98, 'component_of': 1.0, 'on': 0.72, 'part_of': 1.0}
_INVERSE_OWNER_RELATIONS = {'contains': 0.82, 'has': 1.0, 'has_part': 1.0, 'holding': 0.72, 'supporting': 0.82, 'supports': 0.82}
_NON_OBJECT_ANSWER_PATTERN = re.compile('\\b(written|read|reads|says|text|word|words|logo|sign)\\b', re.IGNORECASE)
_YES_NO_PREFIX_PATTERN = re.compile('^(?:is|are|am|was|were|would|could|can|does|do|did|has|have|had)\\b', re.IGNORECASE)

def fuse_spatial_guarded_visual_evidence(graph_result: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, question: str, question_ir: dict[str, Any] | None, evidence_demand: dict[str, Any], max_candidate_packs: int=8, max_visual_lines: int=8, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Preserve v1 evidence and add bounded, non-exclusive spatial support."""
    fused = fuse_navqa_equivalent_visual_evidence(graph_result, visual_payload, graph=graph, question=question, question_ir=question_ir, evidence_demand=evidence_demand, max_candidate_packs=max_candidate_packs, max_visual_lines=max_visual_lines, semantic_text_encoder=semantic_text_encoder)
    requested = requested_geometry_relations(evidence_demand)
    applicable = bool(requested and len(requested) == 1 and evidence_demand.get('anchor_terms') and _is_unknown_object_relation_query(question))
    verifier_demand = deepcopy(evidence_demand)
    if applicable:
        verifier_demand['open_target_slot'] = True
        verifier_demand['target_terms'] = []
    raw_pairs = verify_spatial_pairs(visual_payload, graph=graph, evidence_demand=verifier_demand, limit=max(1, int(max_candidate_packs))) if applicable else []
    pairs, owner_resolution = _resolve_component_pair_owners(raw_pairs, graph=graph)
    return _augment_baseline(fused, pairs=pairs, raw_pairs=raw_pairs, owner_resolution=owner_resolution, requested_relations=requested, applied=applicable, max_visual_lines=max_visual_lines)

def _augment_baseline(graph_result: dict[str, Any], *, pairs: list[dict[str, Any]], raw_pairs: list[dict[str, Any]], owner_resolution: dict[str, Any], requested_relations: set[str], applied: bool, max_visual_lines: int) -> dict[str, Any]:
    bundle = graph_result.get('llm_ready_bundle')
    if not isinstance(bundle, dict):
        return graph_result
    updated = deepcopy(bundle)
    old_visual = deepcopy(graph_result.get('visual_evidence') or {})
    visible_pairs = _select_answer_visible_pairs(pairs, ambiguity_pairs=raw_pairs, baseline_lines=updated.get('evidence_lines', []), limit=min(_MAX_VISIBLE_PAIR_LINES, max(1, int(max_visual_lines))))
    pair_lines = [_spatial_candidate_line(index, pair) for index, pair in enumerate(visible_pairs, start=1)]
    if pair_lines:
        updated['evidence_lines'] = _insert_after_prefix(updated.get('evidence_lines', []), pair_lines)
        updated['evidence_aggregation_lines'] = _insert_after_prefix(updated.get('evidence_aggregation_lines', []), pair_lines)
        aggregation = deepcopy(updated.get('evidence_aggregation') or {})
        aggregation['candidate_facts'] = _insert_after_prefix(aggregation.get('candidate_facts', []), [_spatial_candidate_fact(pair) for pair in visible_pairs])
        updated['evidence_aggregation'] = aggregation
    verifier = {'contract': SPATIAL_GUARDED_CONTRACT, 'applied': applied, 'requested_relations': sorted(requested_relations), 'verified_pair_count': len(raw_pairs), 'owner_resolved_pair_count': len(pairs), 'answer_visible_pair_count': len(visible_pairs), 'fallback_to_v1_visual_evidence': applied and (not bool(visible_pairs)), 'ambiguous_pairs_suppressed': max(0, len(pairs) - len(visible_pairs)), 'baseline_visual_evidence_preserved': True, 'geometry_candidates_prove_unique_answer': False, 'baseline_semantic_support_required': True, 'unverified_visual_candidates_are_relation_proof': False, 'component_owner_resolution': owner_resolution, 'uses_gold_answer': False}
    baseline_packs = list(old_visual.get('candidate_packs', []) or [])
    visual_evidence = {**old_visual, 'contract': SPATIAL_GUARDED_BACKBONE_CONTRACT, 'visual_runtime_contract': _RUNTIME_CONTRACT, 'spatial_pairs': pairs, 'raw_spatial_pairs': raw_pairs, 'spatial_candidate_lines': pair_lines, 'spatial_verification': verifier, 'stats': {**dict(old_visual.get('stats') or {}), 'v1_candidate_pack_count_preserved': len(baseline_packs), 'verified_spatial_pair_count': len(raw_pairs), 'owner_resolved_spatial_pair_count': len(pairs), 'owner_promoted_spatial_pair_count': int(owner_resolution.get('owner_promoted_pair_count') or 0), 'owner_anchor_collision_suppressed_count': int(owner_resolution.get('owner_anchor_collision_suppressed_count') or 0), 'answer_visible_spatial_pair_count': len(visible_pairs), 'answer_visible_unverified_spatial_candidate_count': len(baseline_packs)}}
    graph_result['llm_ready_bundle'] = updated
    graph_result['visual_evidence'] = visual_evidence
    return graph_result

def _spatial_candidate_line(index: int, pair: dict[str, Any]) -> str:
    owner_resolution = pair.get('component_owner_resolution') or {}
    owner_suffix = f" component_target='{pair.get('original_target_label') or 'object'}' owner_relation={owner_resolution.get('relation') or ''} owner_relation_graph_verified=true" if owner_resolution else ''
    return f"spatial_candidate={index} kind=openeqa_spatial_geometry_candidate frame={pair.get('frame_index')} target='{pair.get('target_label') or 'object'}' relation={pair.get('relation') or ''} anchor='{pair.get('anchor_label') or 'object'}' pair_score={float(pair.get('pair_score') or 0.0):.3f} support_frames={pair.get('support_frames') or [pair.get('frame_index')]} relation_geometry_verified=true unique_answer_not_proven=true uses_gold_answer=false{owner_suffix}"

def _spatial_candidate_fact(pair: dict[str, Any]) -> dict[str, Any]:
    owner_resolution = pair.get('component_owner_resolution') or {}
    return {'fact_type': 'spatial_geometry_candidate', 'source': 'openeqa_visual_runtime_v3', 'source_kind': 'visual_bbox_geometry', 'subject': str(pair.get('target_label') or 'object'), 'predicate': str(pair.get('relation') or ''), 'object': str(pair.get('anchor_label') or 'object'), 'score': float(pair.get('pair_score') or 0.0), 'relation_geometry_verified': True, 'component_owner_graph_verified': bool(owner_resolution), 'component_target': pair.get('original_target_label'), 'component_owner_relation': owner_resolution.get('relation'), 'unique_answer_proven': False, 'uses_gold_answer': False}

def _select_answer_visible_pairs(pairs: list[dict[str, Any]], *, ambiguity_pairs: list[dict[str, Any]] | None=None, baseline_lines: Any, limit: int) -> list[dict[str, Any]]:
    if not pairs or limit < 1:
        return []
    top = pairs[0]
    top_target_keys = {_pair_target_key(top), str(top.get('original_target_entity_id') or ''), str(top.get('original_target_label') or '').lower()}
    top_target_keys.discard('')
    other_target_scores = [float(pair.get('pair_score') or 0.0) for pair in (ambiguity_pairs if ambiguity_pairs is not None else pairs[1:]) if _pair_target_key(pair) not in top_target_keys]
    margin = float(top.get('pair_score') or 0.0) - max(other_target_scores) if other_target_scores else float('inf')
    baseline_prefix = list(baseline_lines or [])[:_PRESERVED_BASELINE_PREFIX]
    supported = _target_supported_by_baseline(top, baseline_prefix)
    strong = not other_target_scores or margin >= _MIN_UNAMBIGUOUS_SCORE_MARGIN
    return [top] if strong and supported else []

def _pair_target_key(pair: dict[str, Any]) -> str:
    return str(pair.get('target_entity_id') or '') or str(pair.get('target_label') or '').lower()

def _target_supported_by_baseline(pair: dict[str, Any], baseline_lines: list[Any]) -> bool:
    target_tokens = _label_tokens(pair.get('target_label')) | _label_tokens(pair.get('original_target_label'))
    if not target_tokens:
        return False
    for line in baseline_lines:
        for label in _LABEL_PATTERN.findall(str(line)):
            if target_tokens & _label_tokens(label):
                return True
    return False

def _label_tokens(value: Any) -> set[str]:
    tokens = set(_TOKEN_PATTERN.findall(str(value or '').lower()))
    result = set(tokens)
    for token in tokens:
        if len(token) > 4 and token.endswith('ies'):
            result.add(token[:-3] + 'y')
        elif len(token) > 3 and token.endswith('s'):
            result.add(token[:-1])
    return result

def _resolve_component_pair_owners(pairs: list[dict[str, Any]], *, graph: CanonicalTemporalMemoryGraph) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve component detections to graph-backed owners before QA exposure."""
    entity_by_id = _entity_lookup(graph)
    resolved: list[dict[str, Any]] = []
    stats = {'contract': 'openeqa_component_owner_resolution.v1', 'component_pair_count': 0, 'owner_promoted_pair_count': 0, 'owner_ambiguous_pair_count': 0, 'owner_unresolved_pair_count': 0, 'owner_anchor_collision_suppressed_count': 0, 'deduplicated_pair_count': 0, 'uses_gold_answer': False}
    for source in pairs:
        if not _is_component_label(source.get('target_label')):
            resolved.append(source)
            continue
        stats['component_pair_count'] += 1
        promoted, status = _resolve_pair_target_owner(source, graph=graph, entity_by_id=entity_by_id)
        if status == 'ambiguous':
            stats['owner_ambiguous_pair_count'] += 1
        elif status == 'unresolved':
            stats['owner_unresolved_pair_count'] += 1
        elif status == 'promoted':
            stats['owner_promoted_pair_count'] += 1
        if _pair_target_is_anchor(promoted):
            stats['owner_anchor_collision_suppressed_count'] += 1
            continue
        resolved.append(promoted)
    deduplicated = _deduplicate_owner_resolved_pairs(resolved)
    stats['deduplicated_pair_count'] = len(resolved) - len(deduplicated)
    return (deduplicated, stats)

def _resolve_pair_target_owner(pair: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, entity_by_id: dict[str, Any]) -> tuple[dict[str, Any], str]:
    target_id = str(pair.get('target_entity_id') or '')
    target = entity_by_id.get(target_id)
    if not target_id or target is None:
        return (pair, 'unresolved')
    frame = _optional_int(pair.get('frame_index'))
    scene_id = str(pair.get('scene_id') or '')
    candidates: dict[str, dict[str, Any]] = {}
    for episode in graph.episode_nodes:
        if episode.episode_kind != 'RELATION':
            continue
        if scene_id and episode.scene_id and (episode.scene_id != scene_id):
            continue
        subject = entity_by_id.get(str(episode.subject_entity_id or ''))
        object_ = entity_by_id.get(str(episode.object_entity_id or ''))
        if subject is None or object_ is None:
            continue
        relation = _normalized_relation(episode.raw_predicate or episode.predicate)
        owner = None
        base_score = 0.0
        if subject.entity_id == target.entity_id and relation in _FORWARD_OWNER_RELATIONS:
            owner = object_
            base_score = _FORWARD_OWNER_RELATIONS[relation]
        elif object_.entity_id == target.entity_id and relation in _INVERSE_OWNER_RELATIONS:
            owner = subject
            base_score = _INVERSE_OWNER_RELATIONS[relation]
        if owner is None or owner.entity_id == target.entity_id:
            continue
        if _is_component_label(owner.label) or _is_generic_owner_label(owner.label):
            continue
        distance = _frame_distance(frame, episode.time_range)
        if distance is not None and distance > _MAX_OWNER_RELATION_FRAME_DISTANCE:
            continue
        temporal_bonus = 0.0 if distance is None else max(0.0, 0.24 - 0.02 * distance)
        score = base_score + temporal_bonus
        candidate = {'owner': owner, 'relation': relation, 'relation_episode_id': episode.episode_id, 'relation_frame_distance': distance, 'score': score}
        previous = candidates.get(owner.entity_id)
        if previous is None or score > float(previous['score']):
            candidates[owner.entity_id] = candidate
    ranked = sorted(candidates.values(), key=lambda item: (-float(item['score']), int(item['relation_frame_distance'] or 0), str(item['owner'].entity_id)))
    if not ranked:
        return (pair, 'unresolved')
    if len(ranked) > 1 and float(ranked[0]['score']) - float(ranked[1]['score']) < _MIN_OWNER_SCORE_MARGIN:
        return (pair, 'ambiguous')
    selected = ranked[0]
    owner = selected['owner']
    updated = deepcopy(pair)
    updated['original_target_label'] = str(pair.get('target_label') or 'object')
    updated['original_target_entity_id'] = target.entity_id
    updated['target_label'] = owner.label
    updated['target_entity_id'] = owner.entity_id
    updated['graph_bound'] = bool(owner.entity_id and pair.get('anchor_entity_id'))
    updated['verification_sources'] = _ordered_unique([*list(pair.get('verification_sources') or []), 'canonical_component_owner_relation'])
    updated['component_owner_resolution'] = {'contract': 'openeqa_component_owner_resolution.v1', 'component_entity_id': target.entity_id, 'component_label': target.label, 'owner_entity_id': owner.entity_id, 'owner_label': owner.label, 'relation': selected['relation'], 'relation_episode_id': selected['relation_episode_id'], 'relation_frame_distance': selected['relation_frame_distance'], 'candidate_owner_count': len(ranked), 'uses_gold_answer': False}
    return (updated, 'promoted')

def _entity_lookup(graph: CanonicalTemporalMemoryGraph) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for entity in graph.entity_nodes:
        lookup[entity.entity_id] = entity
        member_ids = entity.attributes.get('member_entity_ids', [])
        if isinstance(member_ids, list):
            for member_id in member_ids:
                lookup.setdefault(str(member_id), entity)
    return lookup

def _pair_target_is_anchor(pair: dict[str, Any]) -> bool:
    target_id = str(pair.get('target_entity_id') or '')
    if not target_id:
        return False
    anchor_ids = {value.strip() for value in str(pair.get('anchor_entity_id') or '').split(' + ') if value.strip()}
    for observation in pair.get('anchor_observations', []) or []:
        if not isinstance(observation, dict):
            continue
        anchor_id = str(observation.get('canonical_entity_id') or '').strip()
        if anchor_id:
            anchor_ids.add(anchor_id)
    return target_id in anchor_ids

def _deduplicate_owner_resolved_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for pair in pairs:
        key = (_pair_target_key(pair), str(pair.get('relation') or ''), str(pair.get('anchor_entity_id') or pair.get('anchor_label') or ''))
        if key in seen:
            continue
        seen.add(key)
        result.append(pair)
    return result

def _is_component_label(value: Any) -> bool:
    return bool(_label_tokens(value) & _COMPONENT_LABEL_TOKENS)

def _is_generic_owner_label(value: Any) -> bool:
    tokens = _label_tokens(value)
    return bool(tokens and tokens <= _GENERIC_OWNER_LABELS)

def _normalized_relation(value: Any) -> str:
    normalized = '_'.join(_TOKEN_PATTERN.findall(str(value or '').lower()))
    return normalized.removeprefix('raw_relation_')

def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None

def _frame_distance(frame: int | None, time_range: Any) -> int | None:
    if frame is None:
        return None
    start = _optional_int(getattr(time_range, 'start_frame', None))
    end = _optional_int(getattr(time_range, 'end_frame', None))
    if start is None and end is None:
        return None
    start = end if start is None else start
    end = start if end is None else end
    assert start is not None and end is not None
    if start <= frame <= end:
        return 0
    return min(abs(frame - start), abs(frame - end))

def _is_unknown_object_relation_query(question: str) -> bool:
    normalized = ' '.join(str(question or '').split())
    if not normalized or _YES_NO_PREFIX_PATTERN.search(normalized):
        return False
    if _NON_OBJECT_ANSWER_PATTERN.search(normalized):
        return False
    if re.match('^which\\b', normalized, re.IGNORECASE):
        return False
    return bool(re.search('\\b(?:what|which|name|identify)\\b', normalized, re.I))

def _insert_after_prefix(values: Any, additions: list[Any]) -> list[Any]:
    baseline = list(values or [])
    prefix_size = min(_PRESERVED_BASELINE_PREFIX, len(baseline))
    return _ordered_unique([*baseline[:prefix_size], *additions, *baseline[prefix_size:]])

def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for value in values:
        marker = repr(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)
    return result
