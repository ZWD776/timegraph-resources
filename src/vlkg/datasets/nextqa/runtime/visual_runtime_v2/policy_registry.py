"""Fixed route-to-handler and role-budget registry for NExT-QA visual v2."""
from __future__ import annotations
from dataclasses import dataclass
from vlkg.datasets.nextqa.runtime.visual_runtime_v2.route_contracts import ROUTE_CAUSAL_HOW, ROUTE_CAUSAL_WHY, ROUTE_DESCRIPTIVE_COUNT, ROUTE_DESCRIPTIVE_DIRECT, ROUTE_DESCRIPTIVE_LOCATION, ROUTE_TEMPORAL_CONCURRENT, ROUTE_TEMPORAL_NEXT, ROUTE_TEMPORAL_PREVIOUS

@dataclass(frozen=True)
class RolePolicy:
    role_id: str
    kind_quotas: tuple[tuple[str, int], ...]
    search_top_k: int = 12

    @property
    def evidence_budget(self) -> int:
        return sum((quota for _kind, quota in self.kind_quotas))

@dataclass(frozen=True)
class RoutePolicy:
    route_id: str
    handler_id: str
    roles: tuple[RolePolicy, ...]
    temporal_policy: str
    evidence_limit: int = 12

def _role(role_id: str, *kind_quotas: tuple[str, int], search_top_k: int=12) -> RolePolicy:
    return RolePolicy(role_id, tuple(kind_quotas), search_top_k)
POLICIES = {ROUTE_CAUSAL_WHY: RoutePolicy(ROUTE_CAUSAL_WHY, 'nextqa_visual_handler_v2:causal_why', (_role('effect_event', ('event_boundary', 2), ('before_after_pair', 2)), _role('preceding_action_or_state', ('before_after_pair', 2), ('temporal_window', 2)), _role('actor_object_interaction', ('object', 2), ('tracklet', 2))), 'strict_effect_boundary_with_preceding_context'), ROUTE_CAUSAL_HOW: RoutePolicy(ROUTE_CAUSAL_HOW, 'nextqa_visual_handler_v2:causal_how', (_role('ordered_action_chain', ('temporal_window', 4), ('event_boundary', 1)), _role('mechanism_object', ('object', 2), ('frame', 1)), _role('actor_object_interaction', ('tracklet', 2), ('object', 2))), 'ordered_mechanism_windows'), ROUTE_TEMPORAL_PREVIOUS: RoutePolicy(ROUTE_TEMPORAL_PREVIOUS, 'nextqa_visual_handler_v2:temporal_previous', (_role('target_event_boundary', ('event_boundary', 3)), _role('preceding_window', ('before_after_pair', 4), ('temporal_window', 3), ('frame', 2))), 'strict_predecessor_of_target_boundary'), ROUTE_TEMPORAL_NEXT: RoutePolicy(ROUTE_TEMPORAL_NEXT, 'nextqa_visual_handler_v2:temporal_next', (_role('target_event_boundary', ('event_boundary', 3)), _role('following_window', ('before_after_pair', 4), ('temporal_window', 3), ('frame', 2))), 'strict_successor_of_target_boundary'), ROUTE_TEMPORAL_CONCURRENT: RoutePolicy(ROUTE_TEMPORAL_CONCURRENT, 'nextqa_visual_handler_v2:temporal_concurrent', (_role('same_time_window', ('temporal_window', 4), ('frame', 2)), _role('actor_object_relations', ('object', 3), ('tracklet', 3))), 'same_window_actor_object_context'), ROUTE_DESCRIPTIVE_COUNT: RoutePolicy(ROUTE_DESCRIPTIVE_COUNT, 'nextqa_visual_handler_v2:descriptive_count', (_role('temporally_deduplicated_entities', ('tracklet', 48), ('object', 16), search_top_k=64),), 'stable_entity_deduplication', evidence_limit=1), ROUTE_DESCRIPTIVE_LOCATION: RoutePolicy(ROUTE_DESCRIPTIVE_LOCATION, 'nextqa_visual_handler_v2:descriptive_location', (_role('target_entity', ('object', 2), ('tracklet', 2)), _role('spatial_anchor', ('object', 2), ('temporal_window', 2)), _role('scene_context', ('frame', 2), ('temporal_window', 2))), 'target_anchor_scene_context'), ROUTE_DESCRIPTIVE_DIRECT: RoutePolicy(ROUTE_DESCRIPTIVE_DIRECT, 'nextqa_visual_handler_v2:descriptive_direct', (_role('direct_target', ('object', 3), ('frame', 3)), _role('direct_temporal_context', ('tracklet', 3), ('temporal_window', 3))), 'direct_object_action_state_or_attribute')}

def get_route_policy(route_id: str) -> RoutePolicy:
    try:
        return POLICIES[route_id]
    except KeyError as exc:
        raise ValueError(f'unsupported NExT-QA visual route: {route_id!r}') from exc

def audit_policy_registry() -> dict[str, object]:
    handler_ids = [policy.handler_id for policy in POLICIES.values()]
    return {'route_count': len(POLICIES), 'handler_count': len(set(handler_ids)), 'unique_handler_per_route': len(handler_ids) == len(set(handler_ids)), 'all_role_budgets_positive': all((role.evidence_budget > 0 for policy in POLICIES.values() for role in policy.roles)), 'all_non_count_routes_within_limit': all((policy.route_id == ROUTE_DESCRIPTIVE_COUNT or sum((role.evidence_budget for role in policy.roles)) <= policy.evidence_limit for policy in POLICIES.values()))}
