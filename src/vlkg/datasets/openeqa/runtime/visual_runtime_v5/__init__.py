"""OpenEQA single-route visual runtime v5.

V5 persists one flat ``question_type``/``route_id`` and performs one direct
policy lookup. Versioned handler implementations remain selected by persisted
handler IDs, so frozen manifests stay reproducible without runtime question
reclassification.
"""
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.dispatcher import OPENEQA_VISUAL_RUNTIME_CONTRACT, ROUTE_STRATEGY_CONTRACT, fuse_single_route_visual_evidence, handler_provenance, route_handler_contract_issues, route_handler_id, route_visual_query_builder_id, route_visual_strategy_contract
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.route_contracts import ROUTE_CONTRACT, ROUTE_IDS, ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5, single_route_decision_from_ir, single_route_decision_issues
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.taxonomy import ROUTE_DEFINITIONS, TAXONOMY_CONTRACT
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.visual_strategy import VISUAL_QUERY_PLAN_CONTRACT, VISUAL_STRATEGIES, VISUAL_STRATEGY_CONTRACT, TARGET_LOCATION_BUILDER_V1, build_visual_query_plan, get_visual_strategy
from vlkg.datasets.openeqa.runtime.visual_runtime_v5.active_handlers import TARGET_LOCATION_HANDLER_ID
__all__ = ['OPENEQA_VISUAL_RUNTIME_CONTRACT', 'ROUTE_CONTRACT', 'ROUTE_DEFINITIONS', 'ROUTE_IDS', 'ROUTE_STRATEGY_CONTRACT', 'ROUTING_PROFILE_IR_FIRST_SINGLE_ROUTE_V5', 'TAXONOMY_CONTRACT', 'VISUAL_QUERY_PLAN_CONTRACT', 'VISUAL_STRATEGIES', 'VISUAL_STRATEGY_CONTRACT', 'TARGET_LOCATION_BUILDER_V1', 'TARGET_LOCATION_HANDLER_ID', 'build_visual_query_plan', 'fuse_single_route_visual_evidence', 'handler_provenance', 'route_handler_contract_issues', 'route_handler_id', 'route_visual_query_builder_id', 'route_visual_strategy_contract', 'get_visual_strategy', 'single_route_decision_from_ir', 'single_route_decision_issues']
