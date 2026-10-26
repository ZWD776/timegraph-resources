"""Allowlisted identity for the isolated NExT-QA route-wide visual strategy."""
from __future__ import annotations
from typing import Any, Mapping
CAUSAL_WHY_ROUTE_ID = 'causal_why'

def validate_route_strategy_identity(strategy_id: str, target_route_id: str, handler_id: str) -> str:
    """Return the strategy family after validating an exact allowlisted identity."""
    if target_route_id != CAUSAL_WHY_ROUTE_ID:
        raise ValueError('unsupported route strategy target')
    if target_route_id == CAUSAL_WHY_ROUTE_ID and strategy_id == 'causal_why_coherent_chain_v2' and (handler_id == 'nextqa_visual_handler_v4:causal_why_coherent_chain'):
        return 'causal_why_coherent_chain_v2'
    raise ValueError('unsupported route strategy identity')

def validate_strategy_evidence_manifest(manifest: Mapping[str, Any]) -> tuple[str, str, str, str]:
    strategy_id = str(manifest.get('strategy_id') or '').strip()
    target_route_id = str(manifest.get('target_route_id') or '').strip()
    handler_id = str(manifest.get('handler_id') or '').strip()
    family = validate_route_strategy_identity(strategy_id, target_route_id, handler_id)
    expected_schema = 'nextqa_visual_route_strategy_evidence_manifest_v1'
    if manifest.get('schema_version') != expected_schema:
        raise ValueError('strategy evidence manifest schema does not match its identity')
    return (strategy_id, target_route_id, handler_id, family)
