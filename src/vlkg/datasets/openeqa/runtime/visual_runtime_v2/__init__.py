"""OpenEQA visual strategy runtime v2.

V2 treats the frozen v1 runtime as a read-only baseline and adds only
OpenEQA-owned evidence verification layers.
"""
from vlkg.datasets.openeqa.runtime.visual_runtime_v2.spatial_geometry import fuse_spatial_verified_visual_evidence, requested_geometry_relations, verify_spatial_pairs
OPENEQA_VISUAL_RUNTIME_CONTRACT = 'openeqa_visual_runtime_v2'
__all__ = ['OPENEQA_VISUAL_RUNTIME_CONTRACT', 'fuse_spatial_verified_visual_evidence', 'requested_geometry_relations', 'verify_spatial_pairs']
