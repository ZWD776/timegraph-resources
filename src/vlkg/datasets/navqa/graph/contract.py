"""NAVQA graph recipe: original full CG6 with real sequence timestamps."""
from copy import deepcopy
from pathlib import Path

from vlkg.core.assets.asset_registry import asset_file


GRAPH_PROFILE = 'time_fact_core'
TIME_AXIS = 'elapsed_real_time'


def new_build_manifest(run_root: Path):
    import json
    current = json.loads(asset_file('navqa', 'graphs', 'manifest').read_text(encoding='utf-8'))
    result = deepcopy(current)
    result['default_output_root'] = str(run_root)
    for row in result['sequences']:
        seq = row['seq_id']
        row['temporal_graph_json'] = str(run_root / 'temporal' / f'seq{seq}' / 'temporal_graph.json')
        row['canonical_graph_json'] = str(run_root / 'canonical' / f'seq{seq}' / 'canonical_temporal_memory_graph.json')
    return result
