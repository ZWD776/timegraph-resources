from __future__ import annotations
from typing import Any, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.rendering import qa_safe_hit
HANDLER_ID = 'nextqa_visual_handler_v2:descriptive_count'

def build(route_record: Mapping[str, Any], role_hits: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hits = role_hits.get('temporally_deduplicated_entities', [])
    entities: list[tuple[str, str, float, float]] = []
    selected: list[dict[str, Any]] = []
    seen_entities: set[str] = set()
    for hit in hits:
        metadata = hit.get('metadata') or {}
        entity_id = str(metadata.get('entity_id') or '')
        if not entity_id or entity_id in seen_entities:
            continue
        seen_entities.add(entity_id)
        label = str(metadata.get('label') or 'object')
        start_ts = float(metadata.get('start_ts', metadata.get('timestamp_sec', 0.0)) or 0.0)
        end_ts = float(metadata.get('end_ts', metadata.get('timestamp_sec', 0.0)) or 0.0)
        entities.append((entity_id, label, start_ts, end_ts))
        selected.append(hit)
        if len(selected) >= 64:
            break
    if entities:
        compact = '; '.join((f'{label} [{start_ts:.3f}-{end_ts:.3f}s]' for _entity_id, label, start_ts, end_ts in entities))
        evidence = [{'role_id': 'temporally_deduplicated_entities', 'candidate_id': 'count-deduplicated-set:' + str(route_record['question_id']), 'candidate_kind': 'deduplicated_entity_set', 'similarity': max((float(hit.get('similarity') or 0.0) for hit in selected)), 'text': f'Temporally deduplicated graph-bound candidate entities: {compact}.'}]
        status = 'completed'
    else:
        evidence = []
        status = 'no_renderable_visual_evidence'
    return {'schema_version': 'nextqa_visual_candidate_pack_v2', 'route_id': 'descriptive_count', 'handler_id': HANDLER_ID, 'handler_module': 'descriptive_count', 'status': status, 'boundary_available': False, 'role_status': {'temporally_deduplicated_entities': 'filled' if selected else 'empty'}, 'evidence': evidence, 'visual_hits': [qa_safe_hit(hit) for hit in selected], 'used_visual_candidate_ids': [str(hit['candidate_id']) for hit in selected], 'excluded_reason_counts': {'deduplicated_entity_hits': max(0, len(hits) - len(selected))}}
