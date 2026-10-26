from importlib import import_module
from pathlib import Path
import sys

import pytest

from scripts.datasets.pipeline import make_plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('dataset', ('navqa', 'nextqa', 'openeqa'))
@pytest.mark.parametrize('stage', ('build', 'index', 'ir', 'retrieve', 'answer'))
def test_equal_interface_produces_parseable_owned_commands(dataset, stage, monkeypatch):
    plan = make_plan(dataset, stage, 'interface-check', 'mock')
    assert plan['commands']
    for command in plan['commands']:
        script = Path(command[1])
        assert script.is_relative_to(ROOT / 'scripts/datasets' / dataset)
        module_name = '.'.join(script.relative_to(ROOT).with_suffix('').parts)
        module = import_module(module_name)
        if hasattr(module, 'build_parser'):
            module.build_parser().parse_args(command[2:])
        elif hasattr(module, 'parse_args'):
            monkeypatch.setattr(sys, 'argv', [str(script), *command[2:]])
            module.parse_args()
        elif hasattr(module, 'normalize_options'):
            module.normalize_options(command[2:])
        else:
            pytest.fail(f'command exposes no argument parser: {module_name}')
    for filename in plan['generated_files']:
        assert Path(filename).is_relative_to(Path(plan['output_root']))


@pytest.mark.parametrize('dataset', ('navqa', 'nextqa', 'openeqa'))
def test_frozen_evaluation_is_available_under_each_dataset(dataset):
    module = import_module(f'vlkg.datasets.{dataset}.audit.replay')
    result = module.replay()
    assert result['dataset'] == dataset
    assert set(result['arms']) == {'A', 'B', 'C'}
    assert result['calls_api'] is False
