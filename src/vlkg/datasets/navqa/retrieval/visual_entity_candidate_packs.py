from __future__ import annotations
import re
from typing import Any, Callable
from vlkg.core.models.canonical_graph import CanonicalTemporalMemoryGraph
from vlkg.datasets.navqa.retrieval.evidence_score_contract import SCORE_CONTRACT_ID, apply_candidate_pack_score_contract, score_contract_aggregation_line, score_contract_pack_line
from vlkg.datasets.navqa.retrieval.synonym_lexicon import synonym_expansion_enabled, synonym_expansion_metadata
VISUAL_ENTITY_PACK_CONTRACT = 'experimental_visual_faiss_entity_candidate_pack_v1'
_STOPWORDS = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'how', 'i', 'in', 'is', 'it', 'me', 'near', 'of', 'on', 'or', 'that', 'the', 'there', 'this', 'to', 'was', 'were', 'what', 'when', 'where', 'which', 'who', 'with', 'you'}
_ROLE_ORDER = {'target_direct': 0, 'target_query': 1, 'provisional_target_query': 1, 'attribute_support': 2, 'visual_recall_candidate': 3, 'constraint_context': 4, 'fallback': 5}
_SEMANTIC_ROLE_GATE_THRESHOLD = 0.78
_SEMANTIC_ROLE_GATE_MARGIN = 0.05

def visual_entity_pack_query_phrase_records(*, question: str, question_ir: dict[str, Any], limit: int) -> list[dict[str, str]]:
    """Build query phrases whose roles keep answer targets separate from constraints."""
    raw_records: list[dict[str, str]] = []

    def add(value: Any, *, role: str, source: str) -> None:
        text = _clean_phrase(value)
        if not text:
            return
        raw_records.append({'text': text, 'rendered': f'a photo of {text}', 'role': role, 'source': source})
    target_values = _iter_text_values(question_ir.get('target_entities'))
    for value in target_values:
        add(value, role='primary_target', source='target_entities')
    target_aliases = _iter_text_values(question_ir.get('target_entity_aliases'))
    for value in target_aliases[:4]:
        add(value, role='primary_target', source='target_entity_aliases')
    if not target_values and (not target_aliases):
        add(question_ir.get('target_state'), role='primary_target', source='target_state')
    attribute_dimension = _clean_phrase(question_ir.get('attribute_dimension'))
    attribute_aliases = _iter_text_values(question_ir.get('attribute_dimension_aliases'))
    if attribute_dimension:
        for value in [*target_values, *target_aliases][:3]:
            add(f'{value} {attribute_dimension}', role='target_attribute', source='target_attribute')
        add(attribute_dimension, role='target_attribute', source='attribute_dimension')
    for alias in attribute_aliases[:3]:
        for value in [*target_values, *target_aliases][:3]:
            add(f'{value} {alias}', role='target_attribute', source='target_attribute_alias')
        add(alias, role='target_attribute', source='attribute_dimension_aliases')
    for value in _iter_text_values(question_ir.get('event_constraints')):
        add(value, role='event_constraint', source='event_constraints')
    for value in _iter_text_values(question_ir.get('event_constraint_aliases'))[:3]:
        add(value, role='event_constraint', source='event_constraint_aliases')
    for value in _iter_text_values(question_ir.get('spatial_constraints')):
        add(value, role='spatial_constraint', source='spatial_constraints')
    for value in _iter_text_values(question_ir.get('spatial_constraint_aliases'))[:3]:
        add(value, role='spatial_constraint', source='spatial_constraint_aliases')
    for value in _iter_text_values(question_ir.get('hard_constraints')):
        add(value, role='hard_constraint', source='hard_constraints')
    for value in _iter_text_values(question_ir.get('hard_constraint_aliases'))[:3]:
        add(value, role='hard_constraint', source='hard_constraint_aliases')
    for value in _iter_text_values(question_ir.get('evidence_requirements')):
        add(value, role='support', source='evidence_requirements')
    for value in _iter_text_values(question_ir.get('support_aliases'))[:2]:
        add(value, role='support', source='support_aliases')
    question_route = _question_route(question_ir)
    if question_route:
        for value in _iter_text_values(question_route.get('primary_target_terms'))[:4]:
            add(value, role='primary_target', source='question_route_primary_target')
        for value in _iter_text_values(question_route.get('attribute_terms'))[:4]:
            add(value, role='target_attribute', source='question_route_attribute')
        for value in _iter_text_values(question_route.get('event_constraints'))[:4]:
            add(value, role='event_constraint', source='question_route_event')
        for value in _iter_text_values(question_route.get('spatial_constraints'))[:4]:
            add(value, role='spatial_constraint', source='question_route_spatial')
        route_id = str(question_route.get('route_id') or '').strip()
        if route_id == 'ocr_logo_text':
            for value in ('visible text', 'logo', 'writing'):
                add(value, role='support', source='question_route_ocr_logo_text')
        elif route_id == 'temporal_interval':
            add('event start end interval', role='support', source='question_route_time')
        elif route_id == 'count_set':
            add('object instances', role='support', source='question_route_count')
    add(_compact_question_visual_phrase(question), role='fallback', source='question')
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for record in raw_records:
        normalized = ' '.join(record['text'].lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        records.append(record)
        if len(records) >= max(1, limit):
            break
    return records

def fuse_visual_entity_candidate_packs(graph_answer: dict[str, Any], visual_payload: dict[str, Any], *, graph: CanonicalTemporalMemoryGraph, object_limit: int=8, neighbor_limit: int=6, question: str='', question_ir: dict[str, Any] | None=None, semantic_align_enabled: bool=False, semantic_gated_role_enabled: bool=False, score_contract_enabled: bool=False, retrieval_window: dict[str, Any] | None=None, semantic_text_encoder: Callable[[list[str]], list[Any]] | None=None) -> dict[str, Any]:
    """Promote object FAISS hits into entity-centered candidate packs.

    FAISS supplies visual anchors; the canonical graph supplies entity identity,
    time span, direct attributes/relations, and bounded context. The pack, not an
    isolated evidence line, becomes the main rerank/QA candidate.
    """
    bundle = graph_answer.get('llm_ready_bundle', {})
    if not isinstance(bundle, dict):
        return graph_answer
    question_ir = question_ir or {}
    anchor_index = _load_anchor_index(graph)
    rows = _entity_pack_object_rows(visual_payload, object_limit=max(1, int(object_limit)), question=question, question_ir=question_ir, semantic_gated_role_enabled=semantic_gated_role_enabled)
    packs = _entity_candidate_packs(rows, graph=graph, anchor_index=anchor_index, question=question, question_ir=question_ir, limit=max(1, int(object_limit)), neighbor_limit=max(1, int(neighbor_limit)), semantic_align_enabled=semantic_align_enabled or semantic_gated_role_enabled, semantic_gated_role_enabled=semantic_gated_role_enabled, semantic_text_encoder=semantic_text_encoder)
    if not packs:
        return graph_answer
    if score_contract_enabled:
        packs = apply_candidate_pack_score_contract(packs, question=question, question_ir=question_ir, question_route=_question_route(question_ir), retrieval_window=retrieval_window, text_encoder=semantic_text_encoder)
    if score_contract_enabled:
        pack_lines = [score_contract_pack_line(index=index, pack=pack) for index, pack in enumerate(packs, start=1)]
        aggregation_lines = [score_contract_aggregation_line(index=index, pack=pack) for index, pack in enumerate(packs, start=1)]
    else:
        pack_lines = [_entity_candidate_pack_line(index=index, pack=pack) for index, pack in enumerate(packs, start=1)]
        aggregation_lines = [_entity_candidate_pack_aggregation_line(index=index, pack=pack) for index, pack in enumerate(packs, start=1)]
    updated = dict(bundle)
    existing_evidence = [str(line).strip() for line in updated.get('evidence_lines', []) or [] if str(line).strip()]
    fallback_evidence = existing_evidence[:4]
    updated['visual_entity_candidate_pack_lines'] = pack_lines
    uses_synonym_lexicon = synonym_expansion_enabled(question_ir)
    pack_metadata = {'contract': VISUAL_ENTITY_PACK_CONTRACT, 'pack_count': len(packs), 'object_row_count': len(rows), 'graph_mutation': False, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon, 'candidate_unit': 'entity_or_tracklet_pack', 'query_phrase_records': list(visual_payload.get('query_phrase_records', []) or []), 'packs': packs}
    if uses_synonym_lexicon:
        pack_metadata['synonym_expansion'] = synonym_expansion_metadata(question_ir)
    if score_contract_enabled:
        pack_metadata.update({'score_contract_enabled': True, 'score_contract': SCORE_CONTRACT_ID, 'candidate_unit': 'candidate_pack', 'fallback_evidence_count': len(fallback_evidence)})
    if semantic_align_enabled or semantic_gated_role_enabled:
        pack_metadata.update({'semantic_alignment_enabled': True, 'semantic_alignment_provider': 'siglip_text', 'llm_semantic_alignment': False})
    question_route = _question_route(question_ir)
    if question_route:
        pack_metadata['question_route'] = question_route
        pack_metadata['uses_question_route'] = True
    if semantic_gated_role_enabled:
        pack_metadata.update({'semantic_role_gate_enabled': True, 'semantic_role_gate_threshold': _SEMANTIC_ROLE_GATE_THRESHOLD, 'semantic_role_gate_margin': _SEMANTIC_ROLE_GATE_MARGIN})
    updated['visual_entity_candidate_packs'] = pack_metadata
    updated['visual_faiss_grounding'] = visual_payload
    if score_contract_enabled:
        updated['fallback_evidence_lines'] = fallback_evidence
        updated['score_contract'] = {'contract': SCORE_CONTRACT_ID, 'enabled': True, 'candidate_unit': 'candidate_pack', 'fallback_unit': 'fallback_evidence', 'pack_count': len(packs), 'fallback_evidence_count': len(fallback_evidence), 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': uses_synonym_lexicon, 'uses_synonym_lexicon': uses_synonym_lexicon}
        updated['evidence_lines'] = _ordered_unique_texts([*pack_lines, *fallback_evidence])
        for key in ('visual_faiss_ger_evidence_lines', 'visual_faiss_ger_bound_fact_lines', 'visual_faiss_object_local_fact_lines', 'visual_faiss_local_subgraph_fact_lines', 'visual_faiss_role_aware_lines', 'visual_faiss_role_aware_aux_lines', 'visual_faiss_unified_candidate_lines', 'visual_faiss_evidence_layer_lines', 'frame_faiss_evidence_lines', 'merged_frame_faiss_evidence_lines'):
            updated[key] = []
    else:
        updated['evidence_lines'] = _ordered_unique_texts([*pack_lines, *existing_evidence])
    evidence_aggregation = dict(updated.get('evidence_aggregation', {}) or {})
    existing_facts = list(evidence_aggregation.get('candidate_facts', []) or [])
    evidence_aggregation['candidate_facts'] = list(packs) if score_contract_enabled else [*packs, *existing_facts]
    updated['evidence_aggregation'] = evidence_aggregation
    existing_aggregation_lines = [str(line).strip() for line in updated.get('evidence_aggregation_lines', []) or [] if str(line).strip()]
    updated['evidence_aggregation_lines'] = aggregation_lines if score_contract_enabled else _ordered_unique_texts([*aggregation_lines, *existing_aggregation_lines])
    context_lines = list(str(updated.get('llm_context_text', '')).splitlines())
    context_lines.extend(['Visual FAISS entity candidate packs:'])
    context_lines.extend((f'- {line}' for line in pack_lines))
    updated['llm_context_text'] = '\n'.join((line for line in context_lines if line))
    structured = dict(updated.get('structured_graph_context', {}) or {})
    structured['visual_entity_candidate_pack_lines'] = pack_lines
    structured['visual_entity_candidate_packs'] = packs
    if score_contract_enabled:
        structured['score_contract'] = updated['score_contract']
        structured['fallback_evidence_lines'] = fallback_evidence
    updated['structured_graph_context'] = structured
    graph_answer['llm_ready_bundle'] = updated
    retrieval_bundle = dict(graph_answer.get('retrieval_bundle', {}) or {})
    digest = list(retrieval_bundle.get('evidence_digest', []) or [])
    for index, pack in enumerate(packs, start=1):
        digest.append({'rank': index, 'source_kind': 'visual_entity_candidate_pack', 'source_record_id': pack.get('source_record_id'), 'summary': pack.get('fact_text'), 'score': pack.get('answerability_score') if score_contract_enabled else pack.get('score'), 'entity_id': pack.get('anchor_entity_id'), 'metadata': {'contract': VISUAL_ENTITY_PACK_CONTRACT, 'ranking_score_breakdown': pack.get('ranking_score_breakdown', {}), 'score_contract': pack.get('score_contract'), 'final_rank_key': pack.get('final_rank_key')}})
    retrieval_bundle['evidence_digest'] = digest
    graph_answer['retrieval_bundle'] = retrieval_bundle
    return graph_answer

def _entity_pack_object_rows(visual_payload: dict[str, Any], *, object_limit: int, question: str, question_ir: dict[str, Any], semantic_gated_role_enabled: bool=False) -> list[dict[str, Any]]:
    target_terms = _role_terms(_iter_text_values(question_ir.get('target_entities')))
    attribute_terms = _role_terms([question_ir.get('attribute_dimension')])
    constraint_terms = _constraint_terms(question_ir)
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for group in visual_payload.get('groups', []) or []:
        if not isinstance(group, dict):
            continue
        phrase = str(group.get('query_phrase') or '').strip()
        phrase_text = str(group.get('query_phrase_text') or phrase).strip()
        phrase_role = _normalize_phrase_role(group.get('query_phrase_role'))
        phrase_role = phrase_role or _classify_phrase_role(phrase_text or phrase, target_terms=target_terms, attribute_terms=attribute_terms, constraint_terms=constraint_terms)
        for result in group.get('results', []) or []:
            if not isinstance(result, dict):
                continue
            row = dict(result)
            row['query_phrase'] = phrase
            row['query_phrase_text'] = phrase_text
            row['query_phrase_role'] = phrase_role
            row['entity_pack_row_role'] = _row_entity_pack_role(row, phrase_role=phrase_role, target_terms=target_terms, attribute_terms=attribute_terms, constraint_terms=constraint_terms, semantic_gated_role_enabled=semantic_gated_role_enabled)
            row['entity_pack_row_score'] = _row_score(row, target_terms=target_terms, attribute_terms=attribute_terms, constraint_terms=constraint_terms)
            entity_id = str(row.get('entity_id') or row.get('anchor_entity_id') or '').strip()
            key = (str(row.get('seq_id') or ''), 'entity_tracklet', entity_id) if row.get('tracklet_hit') and entity_id else (str(row.get('seq_id') or ''), 'object_observation', _optional_int(row.get('graph_frame_index')), _optional_int(row.get('object_index')))
            existing = rows_by_key.get(key)
            if existing is None or _row_sort_key(row) < _row_sort_key(existing):
                rows_by_key[key] = row
    rows = list(rows_by_key.values())
    rows.sort(key=_row_sort_key)
    return rows[:max(1, object_limit) * 4]

def _entity_candidate_packs(rows: list[dict[str, Any]], *, graph: CanonicalTemporalMemoryGraph, anchor_index: dict[tuple[int, int], dict[str, Any]], question: str, question_ir: dict[str, Any], limit: int, neighbor_limit: int, semantic_align_enabled: bool, semantic_gated_role_enabled: bool, semantic_text_encoder: Callable[[list[str]], list[Any]] | None) -> list[dict[str, Any]]:
    entity_index = {str(entity.entity_id): entity for entity in graph.entity_nodes}
    target_terms = _role_terms(_iter_text_values(question_ir.get('target_entities')))
    attribute_terms = _role_terms([question_ir.get('attribute_dimension')])
    constraint_terms = _constraint_terms(question_ir)
    question_terms = _role_terms([question])
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        entity_id = _entity_id_for_row(row, anchor_index=anchor_index, graph=graph)
        if not entity_id:
            frame = _optional_int(row.get('graph_frame_index'))
            object_index = _optional_int(row.get('object_index'))
            entity_id = f"observation:{row.get('seq_id')}:{frame}:{object_index}"
        groups.setdefault(entity_id, []).append(row)
    packs: list[dict[str, Any]] = []
    for entity_id, entity_rows in groups.items():
        entity = entity_index.get(entity_id)
        pack = _build_entity_pack(entity_id=entity_id, entity=entity, rows=entity_rows, graph=graph, target_terms=target_terms, attribute_terms=attribute_terms, constraint_terms=constraint_terms, question_terms=question_terms, neighbor_limit=neighbor_limit)
        if pack:
            packs.append(pack)
    if semantic_align_enabled:
        _apply_semantic_alignment(packs, question_ir=question_ir, text_encoder=semantic_text_encoder, semantic_gated_role_enabled=semantic_gated_role_enabled)
    packs.sort(key=_pack_sort_key)
    return packs[:max(1, limit)]

def _build_entity_pack(*, entity_id: str, entity: Any | None, rows: list[dict[str, Any]], graph: CanonicalTemporalMemoryGraph, target_terms: set[str], attribute_terms: set[str], constraint_terms: set[str], question_terms: set[str], neighbor_limit: int) -> dict[str, Any] | None:
    rows = sorted(rows, key=_row_sort_key)
    entity_label = str(getattr(entity, 'label', '') or rows[0].get('label') or '').strip()
    time_range = _entity_time_range(entity, rows=rows)
    visual_anchors = [_visual_anchor(row) for row in rows[:4]]
    direct_attributes = _ordered_unique_texts([*_entity_attribute_texts(entity), *[text for row in rows for text in _row_direct_attribute_texts(row)]])[:8]
    direct_relations = _direct_relation_texts(graph, entity_id=entity_id, rows=rows, limit=8)
    neighbor_context = _neighbor_context_texts(graph, entity_id=entity_id, rows=rows, limit=neighbor_limit)
    all_direct_text = ' '.join([entity_id, entity_label, *direct_attributes, *direct_relations, *[str(anchor.get('summary') or '') for anchor in visual_anchors]])
    all_context_text = ' '.join([all_direct_text, *neighbor_context])
    target_hits = sorted(target_terms & _text_tokens(all_direct_text))
    attribute_hits = sorted(attribute_terms & _text_tokens(all_direct_text))
    constraint_hits = sorted(constraint_terms & _text_tokens(all_context_text))
    question_hits = sorted(question_terms & _text_tokens(all_context_text))
    roles = [str(row.get('entity_pack_row_role') or 'fallback') for row in rows]
    best_role = sorted(roles, key=lambda role: _ROLE_ORDER.get(role, 99))[0]
    direct_target_bound = bool(target_hits)
    score_breakdown = {'visual_similarity': round(max((_optional_float(row.get('similarity')) or 0.0 for row in rows), default=0.0) * 8.0, 6), 'target_direct': 4.0 if direct_target_bound else 0.0, 'target_query': 2.0 if best_role == 'target_query' else 0.0, 'direct_attribute': min(3.0, len(direct_attributes) * 0.45), 'direct_relation': min(2.5, len(direct_relations) * 0.4), 'constraint': min(3.0, len(constraint_hits) * 0.55), 'question_overlap': min(2.0, len(question_hits) * 0.15)}
    score = round(sum((float(value) for value in score_breakdown.values())), 6)
    if score <= 0.0 and (not direct_attributes) and (not direct_relations):
        return None
    fact_text = _entity_candidate_pack_text(entity_id=entity_id, entity_label=entity_label, role=best_role, time_range=time_range, visual_anchors=visual_anchors, direct_attributes=direct_attributes, direct_relations=direct_relations, constraint_hits=constraint_hits, neighbor_context=neighbor_context, score_breakdown=score_breakdown, semantic_alignment=None)
    return {'fact_type': 'visual_entity_candidate_pack', 'source': 'visual_entity_candidate_packs', 'source_kind': 'visual_entity_candidate_pack', 'source_record_id': f'visual_entity_candidate_pack:{entity_id}', 'fact_text': fact_text, 'anchor_entity_id': entity_id, 'entity_label': entity_label, 'candidate_role': best_role, 'binding_strength': _pack_binding_strength(direct_target_bound=direct_target_bound, direct_attributes=direct_attributes, direct_relations=direct_relations, constraint_hits=constraint_hits), 'time_span': time_range, 'visual_anchors': visual_anchors, 'direct_attributes': direct_attributes, 'direct_relations': direct_relations, 'event_constraint': constraint_hits, 'spatial_constraint': constraint_hits, 'neighbor_context': neighbor_context, 'target_hits': target_hits, 'attribute_hits': attribute_hits, 'constraint_hits': constraint_hits, 'question_hits': question_hits, 'ranking_score_breakdown': score_breakdown, 'score': score, 'uses_gold_answers': False, 'uses_question_type_branches': False, 'uses_dataset_or_domain_wordlist': False}

def _entity_candidate_pack_text(*, entity_id: str, entity_label: str, role: str, time_range: dict[str, Any], visual_anchors: list[dict[str, Any]], direct_attributes: list[str], direct_relations: list[str], constraint_hits: list[str], neighbor_context: list[str], score_breakdown: dict[str, float], semantic_alignment: dict[str, Any] | None) -> str:
    pieces = ['visual_entity_candidate_pack', f'entity={entity_id}', f'label={entity_label!r}' if entity_label else '', f'role={role}', f'time={_time_range_text(time_range)}' if time_range else '']
    if visual_anchors:
        pieces.append('visual_anchor=' + ' | '.join((_short_text(anchor.get('summary'), limit=110) for anchor in visual_anchors[:2])))
    if direct_attributes:
        pieces.append('direct_attributes=' + ' | '.join((_short_text(value, limit=100) for value in direct_attributes[:4])))
    if direct_relations:
        pieces.append('direct_relations=' + ' | '.join((_short_text(value, limit=100) for value in direct_relations[:4])))
    if constraint_hits:
        pieces.append('constraints_satisfied=' + ','.join(constraint_hits[:10]))
    if neighbor_context:
        pieces.append('neighbor_context=' + ' | '.join((_short_text(value, limit=90) for value in neighbor_context[:3])))
    if semantic_alignment:
        similarity = _optional_float(semantic_alignment.get('target_semantic_similarity'))
        query_text = str(semantic_alignment.get('target_semantic_query_text') or '').strip()
        match_text = str(semantic_alignment.get('target_semantic_match_text') or '').strip()
        if similarity is not None and query_text and match_text:
            pieces.append(f'semantic_target={similarity:.3f}: {_short_text(query_text, limit=50)} -> {_short_text(match_text, limit=70)}')
        if 'semantic_role_gate_passed' in semantic_alignment:
            constraint_similarity = _optional_float(semantic_alignment.get('constraint_semantic_similarity'))
            before = str(semantic_alignment.get('candidate_role_before_semantic_gate') or '').strip()
            after = str(semantic_alignment.get('candidate_role_after_semantic_gate') or role).strip()
            gate_state = 'pass' if semantic_alignment.get('semantic_role_gate_passed') else 'fail'
            gate_piece = f'semantic_gate={gate_state}:{before}->{after}'
            if similarity is not None:
                gate_piece += f' target={similarity:.3f}'
            if constraint_similarity is not None:
                gate_piece += f' constraint={constraint_similarity:.3f}'
            pieces.append(gate_piece)
    pieces.append('ranking_score_breakdown=' + ','.join((f'{key}:{value:g}' for key, value in score_breakdown.items() if value)))
    return '; '.join((piece for piece in pieces if piece))

def _entity_candidate_pack_line(*, index: int, pack: dict[str, Any]) -> str:
    return f"rank={index} kind=visual_entity_candidate_pack score={pack.get('score')} entity={pack.get('anchor_entity_id')} label={pack.get('entity_label')!r} role={pack.get('candidate_role')} binding={pack.get('binding_strength')} :: {pack.get('fact_text')}"

def _entity_candidate_pack_aggregation_line(*, index: int, pack: dict[str, Any]) -> str:
    return f"group=visual_entity_candidate_pack:{index} binding={pack.get('binding_strength')} entity={pack.get('anchor_entity_id')} label={pack.get('entity_label')!r} direct_attributes={len(pack.get('direct_attributes', []) or [])} direct_relations={len(pack.get('direct_relations', []) or [])} constraints={','.join(pack.get('constraint_hits', []) or [])} {pack.get('fact_text')}"

def _load_anchor_index(graph: CanonicalTemporalMemoryGraph) -> dict[tuple[int, int], dict[str, Any]]:
    from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import _visual_faiss_observation_anchor_index
    return _visual_faiss_observation_anchor_index(graph)

def _entity_id_for_row(row: dict[str, Any], *, anchor_index: dict[tuple[int, int], dict[str, Any]], graph: CanonicalTemporalMemoryGraph) -> str:
    direct_entity_id = str(row.get('entity_id') or row.get('anchor_entity_id') or '').strip()
    if direct_entity_id:
        return direct_entity_id
    frame = _optional_int(row.get('graph_frame_index'))
    object_index = _optional_int(row.get('object_index'))
    if frame is not None and object_index is not None:
        anchor = anchor_index.get((frame, object_index))
        if isinstance(anchor, dict):
            entity_id = str(anchor.get('entity_id') or '').strip()
            if entity_id:
                return entity_id
    label_terms = _text_tokens(row.get('label'))
    for entity in graph.entity_nodes:
        time_range = _entity_time_range(entity, rows=[])
        if frame is not None and (not _frame_in_time_range(frame, time_range)):
            continue
        entity_terms = _text_tokens(getattr(entity, 'label', ''))
        if label_terms and entity_terms and label_terms & entity_terms:
            return str(entity.entity_id)
    return ''

def _row_entity_pack_role(row: dict[str, Any], *, phrase_role: str, target_terms: set[str], attribute_terms: set[str], constraint_terms: set[str], semantic_gated_role_enabled: bool=False) -> str:
    label_hits = _text_tokens(row.get('label')) & target_terms
    if label_hits:
        return 'target_direct'
    if phrase_role == 'primary_target':
        if semantic_gated_role_enabled:
            return 'provisional_target_query'
        return 'target_query'
    if phrase_role == 'target_attribute' or attribute_terms & _direct_row_terms(row):
        return 'attribute_support'
    if phrase_role in {'event_constraint', 'spatial_constraint', 'hard_constraint', 'support'}:
        return 'constraint_context'
    if constraint_terms & _candidate_row_terms(row):
        return 'constraint_context'
    return 'fallback'

def _row_score(row: dict[str, Any], *, target_terms: set[str], attribute_terms: set[str], constraint_terms: set[str]) -> float:
    score = (_optional_float(row.get('similarity')) or 0.0) * 8.0
    direct_terms = _direct_row_terms(row)
    all_terms = _candidate_row_terms(row)
    score += len(direct_terms & target_terms) * 4.0
    score += len(direct_terms & attribute_terms) * 1.4
    score += len(all_terms & constraint_terms) * 0.7
    if row.get('visual_attribute'):
        score += 0.8
    if row.get('visual_relations'):
        score += 0.5
    return round(score, 6)

def _row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    role = str(row.get('entity_pack_row_role') or 'fallback')
    return (_ROLE_ORDER.get(role, 99), -float(row.get('entity_pack_row_score') or 0.0), -(_optional_float(row.get('similarity')) or 0.0), _optional_int(row.get('graph_frame_index')) or 0, _optional_int(row.get('object_index')) or 0)

def _pack_sort_key(pack: dict[str, Any]) -> tuple[Any, ...]:
    role = str(pack.get('candidate_role') or 'fallback')
    if pack.get('semantic_alignment_provider') == 'siglip_text':
        return (_ROLE_ORDER.get(role, 99), -len(pack.get('target_hits', []) or []), -float(pack.get('score') or 0.0), -len(pack.get('constraint_hits', []) or []), str(pack.get('anchor_entity_id') or ''))
    return (_ROLE_ORDER.get(role, 99), -len(pack.get('target_hits', []) or []), -len(pack.get('constraint_hits', []) or []), -float(pack.get('score') or 0.0), str(pack.get('anchor_entity_id') or ''))

def _pack_binding_strength(*, direct_target_bound: bool, direct_attributes: list[str], direct_relations: list[str], constraint_hits: list[str]) -> str:
    if direct_target_bound and direct_attributes and constraint_hits:
        return 'entity_pack_target_attribute_constraint_bound'
    if direct_target_bound and direct_attributes:
        return 'entity_pack_target_attribute_bound'
    if direct_target_bound and direct_relations:
        return 'entity_pack_target_relation_bound'
    if direct_target_bound:
        return 'entity_pack_target_bound'
    if constraint_hits:
        return 'entity_pack_constraint_context'
    return 'entity_pack_visual_anchor'

def _apply_semantic_alignment(packs: list[dict[str, Any]], *, question_ir: dict[str, Any], text_encoder: Callable[[list[str]], list[Any]] | None, semantic_gated_role_enabled: bool=False) -> None:
    target_texts = _semantic_target_texts(question_ir)
    constraint_texts = _semantic_constraint_texts(question_ir, target_texts=target_texts)
    if not target_texts or not packs:
        for pack in packs:
            _mark_semantic_alignment_disabled(pack)
        return
    candidate_texts_by_pack: list[list[str]] = [_semantic_candidate_texts(pack) for pack in packs]
    all_texts = _ordered_unique_texts([*target_texts, *constraint_texts, *[text for texts in candidate_texts_by_pack for text in texts]])
    embeddings = _semantic_text_embeddings(all_texts, text_encoder=text_encoder)
    for pack, candidate_texts in zip(packs, candidate_texts_by_pack, strict=True):
        best_similarity = 0.0
        best_query = ''
        best_match = ''
        best_constraint_similarity = 0.0
        best_constraint_query = ''
        best_constraint_match = ''
        for target_text in target_texts:
            target_vector = embeddings.get(target_text)
            if target_vector is None:
                continue
            for candidate_text in candidate_texts:
                candidate_vector = embeddings.get(candidate_text)
                if candidate_vector is None:
                    continue
                similarity = _cosine_similarity(target_vector, candidate_vector)
                if similarity > best_similarity or not best_query:
                    best_similarity = similarity
                    best_query = target_text
                    best_match = candidate_text
        for constraint_text in constraint_texts:
            constraint_vector = embeddings.get(constraint_text)
            if constraint_vector is None:
                continue
            for candidate_text in candidate_texts:
                candidate_vector = embeddings.get(candidate_text)
                if candidate_vector is None:
                    continue
                similarity = _cosine_similarity(constraint_vector, candidate_vector)
                if similarity > best_constraint_similarity or not best_constraint_query:
                    best_constraint_similarity = similarity
                    best_constraint_query = constraint_text
                    best_constraint_match = candidate_text
        role_before = str(pack.get('candidate_role') or 'fallback')
        role_after = role_before
        gate_passed = False
        gate_bypass = ''
        if semantic_gated_role_enabled:
            role_after, gate_passed, gate_bypass = _semantic_gated_role(role_before=role_before, target_similarity=best_similarity, constraint_similarity=best_constraint_similarity, target_hits=list(pack.get('target_hits', []) or []))
            pack['candidate_role'] = role_after
        bonus = _semantic_alignment_bonus(best_similarity, candidate_role=role_after, has_direct_text=bool(candidate_texts), semantic_gated_role_enabled=semantic_gated_role_enabled, semantic_role_gate_passed=gate_passed)
        score_breakdown = dict(pack.get('ranking_score_breakdown', {}) or {})
        if semantic_gated_role_enabled:
            score_breakdown.pop('target_query', None)
            score_breakdown.pop('semantic_alignment', None)
            if role_after == 'target_query':
                score_breakdown['target_query'] = 2.0
        if bonus:
            score_breakdown['semantic_alignment'] = bonus
        score = round(sum((float(value) for value in score_breakdown.values())), 6)
        update_values = {'target_semantic_similarity': round(best_similarity, 6), 'target_semantic_query_text': best_query, 'target_semantic_match_text': best_match, 'constraint_semantic_similarity': round(best_constraint_similarity, 6), 'constraint_semantic_query_text': best_constraint_query, 'constraint_semantic_match_text': best_constraint_match, 'semantic_alignment_bonus': bonus, 'semantic_alignment_provider': 'siglip_text', 'llm_semantic_alignment': False, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False, 'ranking_score_breakdown': score_breakdown, 'score': score}
        if semantic_gated_role_enabled:
            update_values.update({'candidate_role_before_semantic_gate': role_before, 'candidate_role_after_semantic_gate': role_after, 'semantic_role_gate_passed': gate_passed, 'semantic_role_gate_threshold': _SEMANTIC_ROLE_GATE_THRESHOLD, 'semantic_role_gate_margin': _SEMANTIC_ROLE_GATE_MARGIN, 'semantic_role_gate_bypass': gate_bypass})
        pack.update(update_values)
        pack['fact_text'] = _entity_candidate_pack_text(entity_id=str(pack.get('anchor_entity_id') or ''), entity_label=str(pack.get('entity_label') or ''), role=role_after, time_range=dict(pack.get('time_span', {}) or {}), visual_anchors=list(pack.get('visual_anchors', []) or []), direct_attributes=list(pack.get('direct_attributes', []) or []), direct_relations=list(pack.get('direct_relations', []) or []), constraint_hits=list(pack.get('constraint_hits', []) or []), neighbor_context=list(pack.get('neighbor_context', []) or []), score_breakdown=score_breakdown, semantic_alignment=pack)

def _mark_semantic_alignment_disabled(pack: dict[str, Any]) -> None:
    pack.update({'target_semantic_similarity': 0.0, 'target_semantic_query_text': '', 'target_semantic_match_text': '', 'semantic_alignment_bonus': 0.0, 'semantic_alignment_provider': 'siglip_text', 'llm_semantic_alignment': False, 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': False})

def _semantic_gated_role(*, role_before: str, target_similarity: float, constraint_similarity: float, target_hits: list[str]) -> tuple[str, bool, str]:
    if role_before == 'target_direct' or target_hits:
        return ('target_direct', True, 'target_direct')
    if role_before not in {'target_query', 'provisional_target_query'}:
        return (role_before, False, '')
    passes_threshold = target_similarity >= _SEMANTIC_ROLE_GATE_THRESHOLD
    passes_margin = target_similarity >= constraint_similarity + _SEMANTIC_ROLE_GATE_MARGIN
    if passes_threshold and passes_margin:
        return ('target_query', True, '')
    return ('visual_recall_candidate', False, '')

def _semantic_target_texts(question_ir: dict[str, Any]) -> list[str]:
    target_entities = _ordered_unique_texts((_clean_phrase(value) for value in _iter_text_values(question_ir.get('target_entities'))))
    if target_entities:
        return target_entities
    return _ordered_unique_texts((_clean_phrase(value) for value in _iter_text_values(question_ir.get('target_state'))))

def _semantic_constraint_texts(question_ir: dict[str, Any], *, target_texts: list[str]) -> list[str]:
    target_normalized = {' '.join(text.lower().split()) for text in target_texts if text}
    values: list[str] = []
    for key in ('event_constraints', 'spatial_constraints', 'hard_constraints', 'evidence_requirements'):
        values.extend(_iter_constraint_semantic_values(question_ir.get(key)))
    cleaned = []
    for value in values:
        text = _clean_phrase(value)
        normalized = ' '.join(text.lower().split())
        if not normalized or normalized in target_normalized:
            continue
        cleaned.append(text)
    return _ordered_unique_texts(cleaned)

def _iter_constraint_semantic_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target_state', 'relation', 'operation'):
            if value.get(key):
                values.append(str(value[key]))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_iter_constraint_semantic_values(item))
        return values
    return [str(value)]

def _semantic_candidate_texts(pack: dict[str, Any]) -> list[str]:
    values: list[Any] = [pack.get('entity_label'), *(pack.get('direct_attributes', []) or []), *(pack.get('direct_relations', []) or [])]
    return _ordered_unique_texts((_clean_phrase(value) for value in values))

def _question_route(question_ir: dict[str, Any]) -> dict[str, Any]:
    route = question_ir.get('question_route') if isinstance(question_ir, dict) else None
    return route if isinstance(route, dict) else {}

def _semantic_text_embeddings(texts: list[str], *, text_encoder: Callable[[list[str]], list[Any]] | None) -> dict[str, Any]:
    cleaned = _ordered_unique_texts((_clean_phrase(text) for text in texts))
    if not cleaned:
        return {}
    encoder = text_encoder or _default_semantic_text_encoder
    rendered = [f'a photo of {text}' for text in cleaned]
    vectors = encoder(rendered)
    return {text: vector for text, vector in zip(cleaned, vectors, strict=False) if vector is not None}

def _default_semantic_text_encoder(texts: list[str]) -> list[Any]:
    from vlkg.core.indexing.vision import VisionFeatureExtractor
    extractor = VisionFeatureExtractor(local_files_only=True)
    return extractor.encode_text_batch(texts)

def _semantic_alignment_bonus(similarity: float, *, candidate_role: str, has_direct_text: bool, semantic_gated_role_enabled: bool=False, semantic_role_gate_passed: bool=False) -> float:
    if not has_direct_text or candidate_role == 'constraint_context':
        return 0.0
    if semantic_gated_role_enabled and (not semantic_role_gate_passed or candidate_role not in {'target_direct', 'target_query'}):
        return 0.0
    return round(min(1.2, max(0.0, float(similarity) - 0.2) * 1.5), 6)

def _cosine_similarity(left: Any, right: Any) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0
    dot = sum((a * b for a, b in zip(left_values, right_values, strict=True)))
    left_norm = sum((value * value for value in left_values)) ** 0.5
    right_norm = sum((value * value for value in right_values)) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)

def _direct_row_terms(row: dict[str, Any]) -> set[str]:
    return _text_tokens(' '.join([str(row.get('label') or '')]))

def _candidate_row_terms(row: dict[str, Any]) -> set[str]:
    values: list[Any] = [row.get('label'), row.get('visual_attribute'), row.get('hsgg_visual_facts'), row.get('visual_relations')]
    return _text_tokens(' '.join((str(value) for value in values if value)))

def _visual_anchor(row: dict[str, Any]) -> dict[str, Any]:
    frame = _optional_int(row.get('graph_frame_index'))
    object_index = _optional_int(row.get('object_index'))
    recall_level = str(row.get('entity_pack_recall_level') or '').strip()
    tracklet_count = _optional_int(row.get('tracklet_observation_count'))
    tracklet_text = f', tracklet_observations={tracklet_count}' if row.get('tracklet_hit') and tracklet_count is not None else ''
    return {'seq_id': str(row.get('seq_id') or ''), 'frame': frame, 'object_index': object_index, 'entity_id': row.get('entity_id') or row.get('anchor_entity_id'), 'recall_level': recall_level or ('tracklet' if row.get('tracklet_hit') else 'observation'), 'tracklet_hit': bool(row.get('tracklet_hit')), 'tracklet_observation_count': tracklet_count, 'tracklet_time_span': row.get('tracklet_time_span'), 'representative_objects': row.get('representative_objects', []), 'label': str(row.get('label') or ''), 'bbox': row.get('bbox'), 'similarity': _optional_float(row.get('similarity')), 'query_phrase': row.get('query_phrase'), 'query_phrase_role': row.get('query_phrase_role'), 'summary': f"object=(seq{row.get('seq_id')}, frame={frame}, obj={object_index}, label={row.get('label')!r}, sim={_optional_float(row.get('similarity'))}{tracklet_text})"}

def _row_direct_attribute_texts(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    attr = str(row.get('visual_attribute') or '').strip()
    if attr:
        values.append(attr)
    for value in row.get('hsgg_visual_facts', []) or []:
        text = str(value).strip()
        if not text or 'neighbor obj=' in text:
            continue
        if ' attr=' in text or ' attribute=' in text:
            values.append(text)
    for representative in row.get('representative_objects', []) or []:
        if not isinstance(representative, dict):
            continue
        for key in ('state_values', 'raw_state_values'):
            for value in representative.get(key, []) or []:
                text = str(value).strip()
                if text:
                    values.append(text)
    return values

def _entity_attribute_texts(entity: Any | None) -> list[str]:
    if entity is None:
        return []
    attrs = getattr(entity, 'attributes', {}) or {}
    values: list[str] = []
    values.extend((str(value) for value in getattr(entity, 'states_seen', []) or []))
    if isinstance(attrs, dict):
        raw_attrs = attrs.get('raw_visual_attributes')
        if isinstance(raw_attrs, list):
            values.extend((str(value) for value in raw_attrs))
        for key in ('canonical_label', 'semantic_group'):
            if attrs.get(key):
                values.append(str(attrs[key]))
    return _ordered_unique_texts(values)

def _direct_relation_texts(graph: CanonicalTemporalMemoryGraph, *, entity_id: str, rows: list[dict[str, Any]], limit: int) -> list[str]:
    del rows
    texts: list[str] = []
    for episode in graph.episode_nodes:
        ids = _episode_entity_ids(episode)
        if entity_id not in ids:
            continue
        label = str(getattr(episode, 'label', '') or '').strip()
        if label:
            texts.append(label)
    for event in graph.transition_event_nodes:
        ids = _transition_event_entity_ids(event)
        if entity_id not in ids:
            continue
        label = str(getattr(event, 'label', '') or '').strip()
        if label:
            texts.append(label)
    for sample in graph.relation_observed_samples:
        ids = [str(getattr(sample, 'subject_entity_id', '') or ''), str(getattr(sample, 'object_entity_id', '') or '')]
        if entity_id not in ids:
            continue
        label = str(getattr(sample, 'label', '') or '').strip()
        if label:
            texts.append(label)
    return _ordered_unique_texts(texts)[:max(1, limit)]

def _neighbor_context_texts(graph: CanonicalTemporalMemoryGraph, *, entity_id: str, rows: list[dict[str, Any]], limit: int) -> list[str]:
    texts: list[str] = []
    for row in rows:
        for value in row.get('hsgg_neighbor_attributes', []) or []:
            text = str(value).strip()
            if text:
                texts.append(text)
    neighbor_ids: list[str] = []
    for sample in graph.relation_observed_samples:
        subject = str(getattr(sample, 'subject_entity_id', '') or '')
        object_id = str(getattr(sample, 'object_entity_id', '') or '')
        if subject == entity_id and object_id:
            neighbor_ids.append(object_id)
        elif object_id == entity_id and subject:
            neighbor_ids.append(subject)
    entity_by_id = {str(entity.entity_id): entity for entity in graph.entity_nodes}
    for neighbor_id in _ordered_unique_texts(neighbor_ids)[:max(1, limit)]:
        entity = entity_by_id.get(neighbor_id)
        if entity is None:
            texts.append(f'neighbor_entity={neighbor_id}')
            continue
        label = str(getattr(entity, 'label', '') or '').strip()
        texts.append(f'neighbor_entity={neighbor_id} label={label!r}')
    return _ordered_unique_texts(texts)[:max(1, limit)]

def _episode_entity_ids(episode: Any) -> list[str]:
    values = [getattr(episode, 'primary_entity_id', None), getattr(episode, 'subject_entity_id', None), getattr(episode, 'object_entity_id', None), *(getattr(episode, 'participant_entity_ids', None) or [])]
    return _ordered_unique_texts((str(value).strip() for value in values if str(value or '').strip()))

def _transition_event_entity_ids(event: Any) -> list[str]:
    values = [getattr(event, 'entity_id', None), getattr(event, 'subject_entity_id', None), getattr(event, 'object_entity_id', None)]
    return _ordered_unique_texts((str(value).strip() for value in values if str(value or '').strip()))

def _entity_time_range(entity: Any | None, *, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if entity is not None:
        return _time_range_payload(getattr(entity, 'time_range', None))
    frames = [frame for frame in (_optional_int(row.get('graph_frame_index')) for row in rows) if frame is not None]
    if not frames:
        return {}
    return {'start_frame': min(frames), 'end_frame': max(frames)}

def _time_range_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    payload: dict[str, Any] = {}
    for name in ('start_frame', 'end_frame', 'start_ts', 'end_ts', 'start_sec', 'end_sec'):
        if hasattr(value, name):
            payload[name] = getattr(value, name)
    return {key: item for key, item in payload.items() if item is not None}

def _frame_in_time_range(frame: int, time_range: dict[str, Any]) -> bool:
    start = _optional_int(time_range.get('start_frame'))
    end = _optional_int(time_range.get('end_frame'))
    if start is None and end is None:
        return True
    if start is not None and frame < start:
        return False
    if end is not None and frame > end:
        return False
    return True

def _time_range_text(value: dict[str, Any]) -> str:
    start = value.get('start_frame')
    end = value.get('end_frame')
    if start is not None and end is not None:
        return f'frames {start}-{end}' if start != end else f'frame {start}'
    return ''

def _constraint_terms(question_ir: dict[str, Any]) -> set[str]:
    return _role_terms([*_iter_text_values(question_ir.get('event_constraints')), *_iter_text_values(question_ir.get('spatial_constraints')), *_iter_text_values(question_ir.get('hard_constraints')), *_iter_text_values(question_ir.get('evidence_requirements'))])

def _classify_phrase_role(phrase: str, *, target_terms: set[str], attribute_terms: set[str], constraint_terms: set[str]) -> str:
    terms = _text_tokens(phrase)
    if terms & target_terms:
        return 'primary_target'
    if terms & attribute_terms:
        return 'target_attribute'
    if terms & constraint_terms:
        return 'event_constraint'
    return 'fallback'

def _normalize_phrase_role(value: Any) -> str:
    text = str(value or '').strip()
    if text in {'primary_target', 'target_attribute', 'event_constraint', 'spatial_constraint', 'hard_constraint', 'support', 'fallback'}:
        return text
    return ''

def _role_terms(values: list[Any]) -> set[str]:
    return _text_tokens(' '.join((str(value) for value in values if str(value or '').strip())))

def _iter_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target', 'target_state', 'relation', 'operation'):
            if value.get(key):
                values.append(str(value[key]))
        nested = value.get('target_entities')
        if isinstance(nested, list):
            values.extend((str(item) for item in nested))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_iter_text_values(item))
        return values
    return [str(value)]

def _compact_question_visual_phrase(question: str) -> str:
    text = _clean_phrase(question)
    text = re.sub('\\b(what|which|where|when|how many|how long|did|was|were|is|are)\\b', '', text, flags=re.I)
    text = re.sub('\\s+', ' ', text).strip(' ?.,:;')
    return text

def _clean_phrase(value: Any) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    text = re.sub('[_/]+', ' ', text)
    text = re.sub('\\b(robot|ego camera|you|i|we|saw|see|seen|seeing)\\b', '', text, flags=re.I)
    text = re.sub('\\s+', ' ', text).strip(' .,:;?')
    if len(text) > 120:
        text = text[:120].rsplit(' ', 1)[0]
    return text

def _text_tokens(value: Any) -> set[str]:
    tokens = re.findall('[a-z0-9]+', str(value or '').lower())
    return {variant for token in tokens if token and token not in _STOPWORDS for variant in _token_variants(token)}

def _token_variants(token: str) -> set[str]:
    variants = {token}
    if len(token) > 4 and token.endswith('ies'):
        variants.add(token[:-3] + 'y')
    if len(token) > 4 and token.endswith('es'):
        variants.add(token[:-2])
    if len(token) > 3 and token.endswith('s'):
        variants.add(token[:-1])
    return variants

def _ordered_unique_texts(values: Any) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered

def _short_text(value: Any, *, limit: int) -> str:
    text = ' '.join(str(value or '').split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(' ', 1)[0] + '...'

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
