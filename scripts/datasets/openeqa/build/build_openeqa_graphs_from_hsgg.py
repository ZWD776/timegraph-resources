"""Strict source-neutral entry point for OpenEQA graph construction."""
from __future__ import annotations
import sys as _bootstrap_sys
from pathlib import Path as _BootstrapPath
_BOOTSTRAP_ROOT = next((p for p in _BootstrapPath(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()))
_bootstrap_sys.path[:0] = [str(_BOOTSTRAP_ROOT), str(_BOOTSTRAP_ROOT / 'src')]
import sys
import argparse
from pathlib import Path
import scripts.datasets.openeqa.build.build_hm3d_graphs_from_hsgg as legacy_builder
UNIFIED_CONFIG = 'datasets/openeqa/graph.yaml'
LEGACY_CONFIG = 'datasets/openeqa/graph.yaml'
CONFIG_ROOT = next((p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())) / 'configs'

def normalize_options(argv: list[str]) -> list[str]:
    """Parse guarded options, including --option=value, before forwarding extras."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--config', default=UNIFIED_CONFIG)
    parser.add_argument('--graph-build-adapter', choices=('none',), default='none')
    parser.add_argument('--graph-profile', choices=('time_fact_core',), default='time_fact_core')
    args, rest = parser.parse_known_args(argv)
    config = Path(args.config)
    if not config.is_absolute():
        config = CONFIG_ROOT / config
    allowed = {(CONFIG_ROOT / name).resolve() for name in (UNIFIED_CONFIG, LEGACY_CONFIG)}
    if config.resolve() not in allowed:
        parser.error(f'Unified OpenEQA requires --config {UNIFIED_CONFIG} (or the recorded legacy config)')
    result = [*rest, '--config', str(config.resolve()), '--graph-build-adapter', 'none', '--graph-profile', 'time_fact_core']
    if '--unified-openeqa-contract' not in result:
        result.append('--unified-openeqa-contract')
    return result

def main() -> None:
    original = sys.argv
    options = normalize_options(original[1:])
    try:
        sys.argv = [original[0], *options]
        legacy_builder.main()
    finally:
        sys.argv = original
if __name__ == '__main__':
    main()
