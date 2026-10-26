"""Accepted OpenEQA entity-attribute mainline boundary."""
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'entity_attribute.frozen_v3'
fuse_entity_attribute_frozen_evidence = fuse_frozen_v3
__all__ = ['HANDLER_ID', 'fuse_entity_attribute_frozen_evidence']
