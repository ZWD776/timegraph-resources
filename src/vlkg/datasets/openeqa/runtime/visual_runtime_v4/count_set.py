"""OpenEQA count-set strategy boundary."""
from vlkg.datasets.openeqa.runtime.visual_runtime_v4.base import fuse_frozen_v3
HANDLER_ID = 'count_set.frozen_v3'
fuse_count_set_evidence = fuse_frozen_v3
__all__ = ['HANDLER_ID', 'fuse_count_set_evidence']
