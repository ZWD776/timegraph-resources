"""OpenEQA visual strategy runtime v3.

V3 keeps frozen v1 broad visual evidence and adds guarded spatial candidates.
It never changes NAVQA runtime modules or graph-only retrieval.
"""
from vlkg.datasets.openeqa.runtime.visual_runtime_v3.spatial_guarded import fuse_spatial_guarded_visual_evidence
OPENEQA_VISUAL_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v3'
__all__ = ['OPENEQA_VISUAL_RUNTIME_CONTRACT', 'fuse_spatial_guarded_visual_evidence']
