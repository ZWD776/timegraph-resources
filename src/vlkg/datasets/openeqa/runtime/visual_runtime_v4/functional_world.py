"""OpenEQA functional-world strategy boundary."""
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'functional_world.frozen_v3'
fuse_functional_world_evidence = fuse_frozen_v3
__all__ = ['HANDLER_ID', 'fuse_functional_world_evidence']
