"""Public core exports must work with optional backend imports forbidden."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from vlkg._optional_exports import resolve_optional_export


def test_core_imports_do_not_load_graphiti():
    root = Path(__file__).resolve().parents[1]
    code = '''
import importlib.abc
import sys
class DenyGraphiti(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('graphiti_core', 'vlkg.integrations.graphiti')):
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, DenyGraphiti())
from vlkg import DomainConfig, build_canonical_temporal_memory_graph
from vlkg.core.assets.asset_registry import asset_file
from vlkg.core.transformers import build_time_fact_core_graph
from vlkg.core.config.config import DomainConfig as DirectConfig
from vlkg.core.transformers.temporal_graph_to_canonical_graph import build_canonical_temporal_memory_graph as direct
assert DomainConfig is DirectConfig
assert build_canonical_temporal_memory_graph is direct
assert callable(build_time_fact_core_graph)
assert not any(name.startswith('graphiti_core') for name in sys.modules)
'''
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    result = subprocess.run([sys.executable, '-c', code], cwd=root, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_exports_preserve_missing_dependency_behavior(monkeypatch):
    error = ModuleNotFoundError('missing external runtime')
    calls = []
    def missing(module, package):
        calls.append(module)
        raise error
    monkeypatch.setattr('vlkg._optional_exports.import_module', missing)
    exports = {'Client': ('.optional', 'Client', False),
               'build_client': ('.optional', 'build_client', True)}
    namespace = {'__name__': 'example', '__package__': 'example'}
    assert resolve_optional_export('Client', exports, namespace) is None
    with pytest.raises(ModuleNotFoundError) as raised:
        namespace['build_client'](_error='caller keyword must not replace the import error')
    assert raised.value is error
    assert calls == ['.optional']
    with pytest.raises(AttributeError):
        resolve_optional_export('unknown', exports, namespace)


def test_requested_optional_exports_retain_identity(monkeypatch):
    import types
    client = type('Client', (), {})
    module = types.SimpleNamespace(Client=client)
    monkeypatch.setattr('vlkg._optional_exports.import_module', lambda *args: module)
    namespace = {'__name__': 'example', '__package__': 'example'}
    assert resolve_optional_export('Client', {'Client': ('.optional', 'Client', False)}, namespace) is client
