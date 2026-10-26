"""Resolve retained script defaults through reviewed active asset mappings."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def evaluate_path(node):
    if isinstance(node, ast.Name) and node.id in ('ROOT', 'REPO_ROOT'):
        return ROOT
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Path' and len(node.args) == 1:
        value = evaluate_path(node.args[0])
        return Path(value) if value is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        a, b = evaluate_path(node.left), evaluate_path(node.right)
        return Path(a) / b if a is not None and b is not None else None
    return None


def nextqa_mapping():
    mapping = {}
    for name in ('nextqa_mainline_copy_plan', 'nextqa_repaired_source_copy_plan'):
        for row in json.loads((ROOT / f'registry/migrations/{name}.json').read_text(encoding='utf-8'))['entries']:
            mapping[row['source']] = row['target']
    m = json.loads((ROOT / 'registry/migrations/nextqa_rebuild_inputs.json').read_text(encoding='utf-8'))
    for row in m['rows']:
        mapping[row['source']] = row['target']
    for row in m.get('bundles', []):
        mapping[row['source_root']] = row['target_root']
    mapping['.scratch/nextqa/graphs/nextqa_cg6_human_v2_full570_20260812a'] = 'datasets/nextqa/graphs/nextqa-cg6-human-v2-time-fact-core-v1/payload/runtime_paths_v3'
    mapping['.scratch/nextqa/visual_indexes/nextqa_visual_indexes_v2_strict_pairs_full570_20260818a'] = 'datasets/nextqa/indexes/nextqa-visual-index-v2-strict-pairs-full570/payload/runtime_paths_v3'
    return mapping


def main():
    archive = ROOT / 'archive/legacy_20260928/migration/before_retained_default_rewrite'
    if archive.exists() and any(archive.rglob('*.py')):
        raise FileExistsError('default rewrite already attempted')
    archive.mkdir(parents=True, exist_ok=True)
    from scripts.datasets.rebase_navqa_runtime import mapping as navqa_mapping
    from scripts.datasets.rebase_openeqa_runtime import mapping as openeqa_mapping
    mappings = {'navqa': navqa_mapping(), 'nextqa': nextqa_mapping(), 'openeqa': openeqa_mapping()}
    receipts = []
    for dataset in mappings:
        ordered = sorted(mappings[dataset].items(), key=lambda x: len(x[0]), reverse=True)
        def relocate(value):
            path = Path(value)
            if path.is_absolute():
                if not path.is_relative_to(ROOT):
                    return None
                text = path.relative_to(ROOT).as_posix()
            else:
                text = str(value).replace('\\', '/')
            for old, new in ordered:
                if text == old or text.startswith(old + '/'):
                    return (ROOT / (new + text[len(old):])).as_posix()
            if dataset == 'navqa' and text == '.scratch/navqa':
                return (ROOT / 'datasets/navqa/source/navqa-coda-every10-v1/payload/navqa').as_posix()
            if text in (f'.scratch/{dataset}/qa_runs', f'.scratch/{dataset}'):
                return (ROOT / f'datasets/{dataset}/runs/new_runs/payload').as_posix()
            return None
        for p in (ROOT / f'scripts/datasets/{dataset}').rglob('*.py'):
            before = p.read_bytes(); code = before.decode('utf-8-sig'); tree = ast.parse(code)
            changes = []
            class Rewrite(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    value = evaluate_path(node)
                    if isinstance(value, Path) and (updated := relocate(value)):
                        changes.append((str(value), updated))
                        return ast.copy_location(ast.Call(ast.Name('Path', ast.Load()), [ast.Constant(updated)], []), node)
                    return self.generic_visit(node)
                def visit_Constant(self, node):
                    if isinstance(node.value, str) and (updated := relocate(node.value)):
                        changes.append((node.value, updated)); node.value = updated
                    return node
            tree = Rewrite().visit(tree)
            if changes:
                updated = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
                compile(updated, p.as_posix(), 'exec')
                snapshot = archive / p.relative_to(ROOT); snapshot.parent.mkdir(parents=True, exist_ok=True); snapshot.write_bytes(before)
                p.write_text(updated, encoding='utf-8', newline='\n')
                receipts.append({'path': p.relative_to(ROOT).as_posix(), 'before_sha256': hashlib.sha256(before).hexdigest(),
                                 'after_sha256': hashlib.sha256(updated.encode()).hexdigest(), 'path_changes': changes})
    (ROOT / 'registry/migrations/retained_default_rewrite.json').write_text(json.dumps({'status': 'rewritten_pending_regression', 'files': receipts}, indent=2) + '\n', encoding='utf-8')
    print('retained script default files rewritten', len(receipts))


if __name__ == '__main__':
    main()
