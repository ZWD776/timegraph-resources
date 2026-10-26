"""Place selected operational scripts under equal dataset/stage directories."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    plan = json.loads((ROOT / 'registry/migrations/dataset_script_move_plan.json').read_text(encoding='utf-8'))
    archive = ROOT / 'archive/legacy_20260928/migration/before_dataset_scripts'
    if archive.exists():
        raise FileExistsError('script migration already attempted')
    archive.mkdir(parents=True)
    paths = {row['source']: row['target'] for row in plan['entries']}
    aliases = {old[:-3].replace('/', '.'): new[:-3].replace('/', '.') for old, new in paths.items()}
    prepared = []
    root_expression = "next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())"
    for row in plan['entries']:
        p = ROOT / row['source']; target = ROOT / row['target']
        if not p.is_file() or target.exists():
            raise ValueError(f'script source missing or target occupied: {row}')
        before = p.read_bytes(); code = before.decode('utf-8-sig')
        code = re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', root_expression, code)
        for old, new in paths.items():
            code = code.replace(old, new).replace(old.replace('/', '\\'), new.replace('/', '\\'))
        compile(code, row['target'], 'exec')
        prepared.append((row, p, target, before, code))
    receipts = []
    for row, source, target, before, code in prepared:
        snapshot = archive / row['source']; snapshot.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(before)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target); target.write_text(code, encoding='utf-8', newline='\n')
        wrapper = ('"""Compatibility CLI forwarding to the canonical dataset entrypoint."""\n'
            'from pathlib import Path\nimport runpy\n\n'
            f'ROOT = {root_expression}\n'
            f'if __name__ == "__main__":\n    runpy.run_path(str(ROOT / {row["target"]!r}), run_name="__main__")\n')
        source.write_text(wrapper, encoding='utf-8', newline='\n')
        receipts.append({**row, 'source_sha256': hashlib.sha256(before).hexdigest(),
                         'canonical_sha256': hashlib.sha256(code.encode()).hexdigest(),
                         'wrapper_sha256': hashlib.sha256(wrapper.encode()).hexdigest()})
    for dataset in ('navqa', 'nextqa', 'openeqa'):
        for stage in plan['target_stages']:
            p = ROOT / f'scripts/datasets/{dataset}/{stage}'
            p.mkdir(parents=True, exist_ok=True)
            for folder in (p, p.parent):
                init = folder / '__init__.py'
                if not init.exists():
                    init.write_text('"""Dataset-owned operational entrypoints."""\n', encoding='utf-8')
    script_init = ROOT / 'scripts/__init__.py'
    script_init.write_text('"""Operational scripts with canonical compatibility imports."""\n'
        'from pathlib import Path\nimport sys\n'
        f'ROOT = {root_expression}\n'
        'sys.path.insert(0, str(ROOT / "src"))\n'
        'from vlkg._module_alias import install_aliases\n'
        '_SCRIPT_ALIASES = ' + repr(aliases) + '\ninstall_aliases(_SCRIPT_ALIASES)\n', encoding='utf-8', newline='\n')
    changed = 0
    for top in ('scripts', 'tests', 'src'):
        for p in (ROOT / top).rglob('*.py'):
            if p == script_init or p.name == 'migrate_dataset_scripts.py' or p.relative_to(ROOT).as_posix() in paths:
                continue
            before = p.read_bytes(); code = before.decode('utf-8-sig'); updated = code
            for old, new in paths.items():
                updated = updated.replace(old, new)
            for old, new in aliases.items():
                updated = re.sub(r'(\bfrom\s+|\bimport\s+)' + re.escape(old) + r'(?![A-Za-z0-9_])',
                                 lambda m, value=new: m.group(1) + value, updated)
            if updated != code:
                compile(updated, p.as_posix(), 'exec')
                snapshot = archive / 'callers' / p.relative_to(ROOT); snapshot.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(before)
                p.write_text(updated, encoding='utf-8', newline='\n'); changed += 1
    result = {'status': 'moved_pending_regression', 'files': receipts, 'callers_updated': changed,
              'aliases': aliases, 'old_cli_paths_are_wrappers_only': True, 'data_deleted': False}
    (ROOT / 'registry/migrations/dataset_script_move_receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('selected operational scripts', len(receipts), 'callers updated', changed)


if __name__ == '__main__':
    main()
