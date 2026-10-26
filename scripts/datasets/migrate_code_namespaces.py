"""Move implementations to core and symmetric dataset packages, keeping aliases."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    plan_path = ROOT / 'registry/migrations/code_namespace_move_plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    aliases = plan['aliases']
    archive = ROOT / 'archive/legacy_20260928/migration/code_namespace_originals'
    if archive.exists():
        raise FileExistsError('namespace migration already started; inspect receipt before retry')
    archive.mkdir(parents=True)
    receipts = []
    # Prepare all code changes and validate syntax before the first source move.
    prepared = []
    ordered = sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True)
    for row in plan['entries']:
        source, target = ROOT / row['source'], ROOT / row['target']
        if not source.is_file() or target.exists():
            raise ValueError(f'namespace source missing or destination exists: {row}')
        if not source.resolve().is_relative_to(ROOT) or not target.resolve().is_relative_to(ROOT):
            raise ValueError('namespace path escapes workspace')
        before = source.read_bytes()
        code = before.decode('utf-8-sig')
        for old, new in ordered:
            code = re.sub(re.escape(old) + r'(?![A-Za-z0-9_])', lambda match, value=new: value, code)
        if row['source'].startswith('src/vlkg/') and '/eval_adapters/' not in row['source']:
            changed_root = bool(re.search(r'Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', code))
            code = re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', 'workspace_root()', code)
            if 'Path(__file__).resolve().parent.parent.parent' in code:
                code = code.replace('Path(__file__).resolve().parent.parent.parent', 'workspace_root()')
                changed_root = True
            if changed_root:
                tree = ast.parse(code)
                futures = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == '__future__']
                lines = code.splitlines(keepends=True)
                offset = max((n.end_lineno for n in futures), default=0)
                lines.insert(offset, '\nfrom vlkg.workspace import workspace_root\n')
                code = ''.join(lines)
        compile(code, row['target'], 'exec')
        prepared.append((row, source, target, before, code))
    for row, source, target, before, code in prepared:
        snapshot = archive / row['source']
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(before)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Paths were resolved and verified above; no content is discarded.
        os.rename(source, target)
        target.write_text(code, encoding='utf-8', newline='\n')
        receipts.append({**row, 'original_sha256': hashlib.sha256(before).hexdigest(),
                         'current_sha256': hashlib.sha256(code.encode()).hexdigest(),
                         'snapshot': snapshot.relative_to(ROOT).as_posix()})
    # Declare actual package boundaries; package initializers carry no strategies.
    for row in plan['entries']:
        parent = (ROOT / row['target']).parent
        while parent != ROOT / 'src/vlkg':
            initializer = parent / '__init__.py'
            if not initializer.exists():
                initializer.write_text('"""Canonical project package."""\n', encoding='utf-8')
            parent = parent.parent
    root_init = ROOT / 'src/vlkg/__init__.py'
    old = root_init.read_bytes()
    (archive / 'root_init.before.py').write_bytes(old)
    bootstrap = 'from ._module_alias import install_aliases\n\n_PUBLIC_MODULE_ALIASES = ' + repr(aliases) + '\n\ninstall_aliases(_PUBLIC_MODULE_ALIASES)\n\n'
    root_init.write_text(bootstrap + old.decode('utf-8-sig'), encoding='utf-8', newline='\n')
    # Current caller source uses canonical imports. Compatibility locations stay
    # for external callers and old pickles, not duplicate implementation files.
    callers = []
    for top in ('src', 'scripts', 'tests'):
        for p in (ROOT / top).rglob('*.py'):
            if p == root_init or p.name == 'migrate_code_namespaces.py':
                continue
            before = p.read_bytes()
            code = before.decode('utf-8-sig')
            updated = code
            for old_name, new_name in ordered:
                # Avoid rewriting the alias implementation or historical names
                # in diagnostic text; change actual absolute import statements.
                updated = re.sub(r'(?m)(\bfrom\s+|\bimport\s+)' + re.escape(old_name) + r'(?![A-Za-z0-9_])',
                                 lambda m, value=new_name: m.group(1) + value, updated)
            if updated != code:
                compile(updated, p.as_posix(), 'exec')
                snapshot = archive / 'callers' / p.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(before)
                p.write_text(updated, encoding='utf-8', newline='\n')
                callers.append({'path': p.relative_to(ROOT).as_posix(), 'original_sha256': hashlib.sha256(before).hexdigest(),
                                'current_sha256': hashlib.sha256(updated.encode()).hexdigest()})
    result = {'contract': 'canonical_code_namespace_move_receipt_v1', 'status': 'moved_pending_regression',
              'files': receipts, 'callers_updated': callers, 'aliases': aliases,
              'old_implementations_preserved': True, 'data_assets_deleted': False,
              'core_imports_dataset_strategies_allowed': False}
    (ROOT / 'registry/migrations/code_namespace_move_receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('canonical implementation files', len(receipts), 'callers updated', len(callers))


if __name__ == '__main__':
    main()
