"""New-run defaults and resume contracts use distinct, explicit input sources."""
import importlib
import json
from pathlib import Path

import pytest

from vlkg.core.assets.asset_resume import recorded_or_selected

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    'build_visual_route_policy_v3', 'build_causal_why_coherent_chain_evidence',
    'run_causal_why_coherent_chain_qa', 'run_visual_enhanced_qa',
    'merge_visual_strategy_predictions', 'merge_visual_route_policy_v3',
)


@pytest.mark.parametrize('name', SCRIPTS)
def test_default_inputs_are_existing_parallel_assets(name):
    module = importlib.import_module('scripts.nextqa.' + name)
    extra = []
    if name == 'merge_visual_strategy_predictions':
        extra = ['--overlay-predictions', 'unused.json']
    if name == 'merge_visual_route_policy_v3':
        extra = ['--base-predictions', 'unused.json', '--target-overlay-predictions', 'unused.json']
    args = module.build_parser().parse_args(['--run-id', 'input-probe'] + extra)
    inputs = module.resolve_inputs(args)
    for path in inputs.values():
        if path is None:
            continue
        assert path.is_file(), path
        assert path.is_relative_to(ROOT / 'datasets/nextqa') or path.is_relative_to(ROOT / 'configs/datasets/nextqa')
    if hasattr(args, 'strategy_version'):
        assert args.strategy_version == 'v2'
    if hasattr(args, 'scope'):
        assert args.scope == 'full'


def test_flat_and_fingerprinted_resume_paths_are_preserved(tmp_path):
    path = tmp_path / 'run_config.json'
    old = tmp_path / 'original-input.json'
    path.write_text(json.dumps({'contract': {'flat_path': str(old), 'fingerprint': {'path': str(old)}}}))
    for field in ('flat_path', 'fingerprint'):
        assert recorded_or_selected(None, resume=True, run_config=path, field=field,
                                    selected=lambda: tmp_path / 'new-default.json') == old
    explicit = tmp_path / 'explicit.json'
    assert recorded_or_selected(explicit, resume=True, run_config=path, field='flat_path',
                                selected=lambda: old) == explicit


def test_full_evidence_builder_does_not_require_historical_pilot():
    from scripts.nextqa import build_causal_why_coherent_chain_evidence as builder
    args = builder.build_parser().parse_args(['--run-id', 'input-probe'])
    assert builder.resolve_inputs(args)['pilot_manifest'] is None
    args.scope = 'pilot'
    with pytest.raises(ValueError, match='explicit --pilot-manifest'):
        builder.resolve_inputs(args)
