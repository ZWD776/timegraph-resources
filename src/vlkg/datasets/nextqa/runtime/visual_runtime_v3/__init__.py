"""Isolated NExT-QA visual strategy experiments, version 3."""
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain import HANDLER_ID, STRATEGY_ID, build_coherent_causal_chain_pack
from vlkg.datasets.nextqa.runtime.visual_runtime_v3.causal_why_coherent_chain_v2 import HANDLER_ID as V2_HANDLER_ID, STRATEGY_ID as V2_STRATEGY_ID, build_coherent_causal_chain_pack_v2
__all__ = ['HANDLER_ID', 'STRATEGY_ID', 'build_coherent_causal_chain_pack', 'V2_HANDLER_ID', 'V2_STRATEGY_ID', 'build_coherent_causal_chain_pack_v2']
