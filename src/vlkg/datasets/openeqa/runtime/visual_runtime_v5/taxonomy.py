"""Frozen OpenEQA V5 semantic question taxonomy.

Semantic details such as the queried attribute, state, spatial relation,
count filter, or OCR ownership remain IR parameters and must not create a
second runtime route.
"""
from __future__ import annotations
TAXONOMY_CONTRACT = 'openeqa_single_route_taxonomy.v1'
ROUTE_IDS = frozenset({'relation_bound_identity', 'direct_identity', 'entity_attribute', 'entity_state', 'entity_existence', 'surface_capacity', 'target_location', 'relation_query', 'entity_count', 'text_reading', 'functional_affordance', 'grounded_world_knowledge', 'scene_description', 'content_description'})
ROUTE_DEFINITIONS = {'relation_bound_identity': 'Identify an unknown entity using anchors and relations.', 'direct_identity': 'Identify an explicitly specified entity.', 'entity_attribute': 'Query an entity attribute carried by IR parameters.', 'entity_state': 'Query an entity state carried by target_state.', 'entity_existence': 'Verify whether an entity exists with negative coverage.', 'surface_capacity': 'Judge available surface space or activity capacity.', 'target_location': 'Locate a known target.', 'relation_query': 'Describe or verify relations between known entities.', 'entity_count': 'Count entities subject to optional IR constraints.', 'text_reading': 'Read text, dates, labels, or logos with optional ownership.', 'functional_affordance': 'Infer use, affordance, action, or recommendation.', 'grounded_world_knowledge': 'Use scene-grounded world knowledge.', 'scene_description': 'Classify or describe a room or whole scene.', 'content_description': 'Describe depicted content in an image or artwork.'}
__all__ = ['ROUTE_DEFINITIONS', 'ROUTE_IDS', 'TAXONOMY_CONTRACT']
