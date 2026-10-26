"""Put each dataset implementation under the same five stage namespaces."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    plan = json.loads((ROOT / 'registry/migrations/dataset_stage_namespace_plan.json').read_text(encoding='utf-8'))
    mapping = sorted(plan['aliases'].items(), key=lambda item: len(item[0]), reverse=True)
    archive = ROOT / 'archive/legacy_20260928/migration/before_dataset_stages'
    if archive.exists():
        raise FileExistsError('stage migration already attempted')
    archive.mkdir(parents=True)
    def canonical(name):
        for old, new in mapping:
            if name == old or name.startswith(old + '.'):
                return new + name[len(old):]
        return name
    prepared = []
    for row in plan['entries']:
        source, target = ROOT / row['source'], ROOT / row['target']
        if not source.is_file() or target.exists():
            raise ValueError('missing stage source or occupied target')
        before = source.read_bytes()
        tree = ast.parse(before.decode('utf-8-sig'))
        module = row['source'][4:-3].replace('/', '.')
        package = module[:-len('.__init__')] if source.name == '__init__.py' else module.rsplit('.', 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                bits = package.split('.')
                base = '.'.join(bits[:len(bits)-node.level+1])
                node.module = canonical(base + ('.' + node.module if node.module else ''))
                node.level = 0
            elif isinstance(node, ast.ImportFrom) and node.module:
                node.module = canonical(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    alias.name = canonical(alias.name)
        code = ast.unparse(tree) + '\n'
        if re.search(r'Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', code):
            code = re.sub(r'Path\(__file__\)\.resolve\(\)\.parents\[\d+\]', 'workspace_root()', code)
            lines = code.splitlines(keepends=True)
            futures = [node.end_lineno for node in ast.parse(code).body if isinstance(node, ast.ImportFrom) and node.module == '__future__']
            lines.insert(max(futures, default=0), 'from vlkg.workspace import workspace_root\n')
            code = ''.join(lines)
        compile(code, row['target'], 'exec')
        prepared.append((row, source, target, before, code))
    receipts = []
    for row, source, target, before, code in prepared:
        snapshot = archive / row['source']
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(before)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(source, target)
        target.write_text(code, encoding='utf-8', newline='\n')
        receipts.append({**row, 'before_sha256': hashlib.sha256(before).hexdigest(),
                         'after_sha256': hashlib.sha256(code.encode()).hexdigest()})
    for dataset in ('navqa', 'nextqa', 'openeqa'):
        for group in plan['stages']:
            folder = ROOT / f'src/vlkg/datasets/{dataset}/{group}'
            folder.mkdir(parents=True, exist_ok=True)
            init = folder / '__init__.py'
            if not init.exists():
                init.write_text(f'"""{dataset} {group} boundary."""\n', encoding='utf-8')
    root_init = ROOT / 'src/vlkg/__init__.py'
    before = root_init.read_bytes()
    (archive / 'root_init.before.py').write_bytes(before)
    tree = ast.parse(before.decode('utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_PUBLIC_MODULE_ALIASES' for t in node.targets):
            values = ast.literal_eval(node.value)
            values.update(plan['aliases'])
            node.value = ast.parse(repr(values), mode='eval').body
    root_init.write_text(ast.unparse(ast.fix_missing_locations(tree)) + '\n', encoding='utf-8', newline='\n')
    callers = 0
    for top in ('src', 'scripts', 'tests'):
        for p in (ROOT / top).rglob('*.py'):
            if p == root_init or p.name == 'migrate_dataset_stages.py':
                continue
            before = p.read_bytes(); code = before.decode('utf-8-sig'); updated = code
            for old, new in mapping:
                updated = re.sub(r'(\bfrom\s+|\bimport\s+)' + re.escape(old) + r'(?![A-Za-z0-9_])',
                                 lambda m, value=new: m.group(1) + value, updated)
            if updated != code:
                compile(updated, p.as_posix(), 'exec')
                snapshot = archive / 'callers' / p.relative_to(ROOT)
                snapshot.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(before)
                p.write_text(updated, encoding='utf-8', newline='\n'); callers += 1
    result = {'status': 'moved_pending_regression', 'files': receipts,
              'callers_updated': callers, 'aliases': plan['aliases'], 'old_code_preserved': True}
    (ROOT / 'registry/migrations/dataset_stage_namespace_receipt.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('stage implementation files', len(receipts), 'callers', callers)


if __name__ == '__main__':
    main()
