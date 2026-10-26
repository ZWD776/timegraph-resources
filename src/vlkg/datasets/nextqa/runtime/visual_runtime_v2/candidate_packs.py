"""Role-budgeted candidate selection and candidate-pack contracts."""
from __future__ import annotations
from collections import Counter
from typing import Any, Callable, Mapping
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.policy_registry import get_route_policy
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.rendering import qa_safe_hit, render_hit
AcceptHit = Callable[[str, Mapping[str, Any]], bool]

def build_standard_candidate_pack(route_record: Mapping[str, Any], role_hits: Mapping[str, list[dict[str, Any]]], *, actual_handler_id: str, accept_hit: AcceptHit | None=None) -> dict[str, Any]:
    route_id = str(route_record['route_id'])
    policy = get_route_policy(route_id)
    if policy.handler_id != actual_handler_id:
        raise ValueError('handler dispatch does not match the persisted policy')
    accepted = accept_hit or (lambda _role, _hit: True)
    selected_rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    excluded = Counter()
    role_status: dict[str, str] = {}
    for role in policy.roles:
        hits = role_hits.get(role.role_id, [])
        by_kind: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
            by_kind.setdefault(str(hit['candidate_kind']), []).append(hit)
        selected_for_role = 0
        for kind, quota in role.kind_quotas:
            used_for_kind = 0
            for hit in by_kind.get(kind, []):
                candidate_id = str(hit['candidate_id'])
                if candidate_id in used_ids:
                    excluded['duplicate_candidate'] += 1
                    continue
                if not accepted(role.role_id, hit):
                    excluded['route_temporal_constraint'] += 1
                    continue
                text = render_hit(route_id, role.role_id, hit)
                if not text:
                    excluded['not_renderable_as_graph_bound_text'] += 1
                    continue
                selected_rows.append({'role_id': role.role_id, 'candidate_id': candidate_id, 'candidate_kind': kind, 'similarity': float(hit.get('similarity') or 0.0), 'text': text, 'safe_hit': qa_safe_hit(hit)})
                used_ids.add(candidate_id)
                selected_for_role += 1
                used_for_kind += 1
                if used_for_kind >= quota:
                    break
        role_status[role.role_id] = 'filled' if selected_for_role else 'empty'
    selected_rows = selected_rows[:policy.evidence_limit]
    boundary_available = any((row['candidate_kind'] in {'event_boundary', 'before_after_pair'} for row in selected_rows))
    status = 'completed'
    if route_id in {'causal_why', 'temporal_previous', 'temporal_next'} and (not boundary_available):
        status = 'boundary_unavailable_fallback' if selected_rows else 'boundary_unavailable'
    elif not selected_rows:
        status = 'no_renderable_visual_evidence'
    return {'schema_version': 'nextqa_visual_candidate_pack_v2', 'route_id': route_id, 'handler_id': actual_handler_id, 'handler_module': actual_handler_id.split(':', 1)[-1], 'status': status, 'boundary_available': boundary_available, 'role_status': role_status, 'evidence': [{key: value for key, value in row.items() if key != 'safe_hit'} for row in selected_rows], 'visual_hits': [row['safe_hit'] for row in selected_rows], 'used_visual_candidate_ids': [row['candidate_id'] for row in selected_rows], 'excluded_reason_counts': dict(sorted(excluded.items()))}
