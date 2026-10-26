"""Seal historical source snapshots and remove unused graph-version branches.

Shared lexical helpers required by retained callers are kept under neutral
names. The original full source is recoverable inside the single archive root.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[2]
MARKER = re.compile(r'cg[478]', re.I)
FILES = (
    'src/vlkg/integrations/cognee/canonical_hybrid_retrieval.py',
    'src/vlkg/eval_adapters/openeqa/canonical_hybrid_retrieval.py',
    'scripts/datasets/navqa/qa/run_navqa_mainline_all_sequences.py',
    'scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py',
    'scripts/datasets/navqa/qa/run_navqa_route.py',
)


def legacy(value):
    return isinstance(value, str) and MARKER.search(value) is not None


class Pruner(ast.NodeTransformer):
    def __init__(self, required):
        self.required = required
        self.renamed = {name: re.sub(r'_cg[478]_direct_', '_generic_', name, flags=re.I)
                        for name in required}
        self.removed = []

    def visit_FunctionDef(self, node):
        if legacy(node.name) and node.name not in self.required:
            self.removed.append(node.name)
            return None
        node.name = self.renamed.get(node.name, node.name)
        return self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_arguments(self, node):
        pairs = [(a, d) for a, d in zip(node.kwonlyargs, node.kw_defaults) if not legacy(a.arg)]
        node.kwonlyargs = [a for a, _ in pairs]
        node.kw_defaults = [d for _, d in pairs]
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if any(legacy(n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else '')
               for target in node.targets for n in ast.walk(target)):
            return None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if any(legacy(n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else '')
               for n in ast.walk(node.target)):
            return None
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.renamed:
            node.id = self.renamed[node.id]
        elif legacy(node.id) and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(False), node)
        return node

    def visit_Attribute(self, node):
        if legacy(node.attr) and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(False), node)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if any(isinstance(x, ast.Constant) and legacy(x.value) for x in node.comparators):
            return ast.copy_location(ast.Constant(False), node)
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'startswith' and node.args:
            if isinstance(node.args[0], ast.Constant) and legacy(node.args[0].value):
                return ast.copy_location(ast.Constant(False), node)
        node.keywords = [kw for kw in node.keywords if not legacy(kw.arg)]
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
            call = node.value
            if call.func.attr == 'add_argument' and call.args and isinstance(call.args[0], ast.Constant):
                if legacy(call.args[0].value):
                    return None
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and legacy(node.value.value):
            return ast.copy_location(ast.Expr(ast.Constant('Retained mainline graph runtime.')), node)
        return self.generic_visit(node)

    def visit_Dict(self, node):
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and legacy(k.value))]
        node.keys, node.values = [k for k, _ in pairs], [v for _, v in pairs]
        return self.generic_visit(node)

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and node.test.value is False:
            return node.orelse
        if not node.body:
            node.body = [ast.Pass()]
        return node


def retain_helpers(tree):
    definitions = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    old = {name for name in definitions if legacy(name)}
    # Remove the historical dispatch before computing the helpers needed by
    # otherwise retained functions and classes.
    preview = Pruner(old).visit(deepcopy(tree))
    required = set()
    for node in preview.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and legacy(node.name):
            continue
        required |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & old
    # The preview has neutral names; account for them as well.
    neutral = {re.sub(r'_cg[478]_direct_', '_generic_', name, flags=re.I): name for name in old}
    for node in preview.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in neutral:
            continue
        required |= {neutral[n.id] for n in ast.walk(node) if isinstance(n, ast.Name) and n.id in neutral}
    while True:
        expanded = required | {n.id for name in required for n in ast.walk(definitions[name])
                               if isinstance(n, ast.Name) and n.id in old}
        if expanded == required:
            return required
        required = expanded


def main():
    out = ROOT / 'archive/legacy_20260928/migration/graph_runtime_before_isolation'
    report = []
    prepared = []
    for name in FILES:
        source = ROOT / name
        before = source.read_bytes()
        tree = ast.parse(before.decode('utf-8-sig'))
        required = retain_helpers(tree)
        pruner = Pruner(required)
        current = ast.fix_missing_locations(pruner.visit(tree))
        code = ast.unparse(current) + '\n'
        compile(code, name, 'exec')
        prepared.append((source, before, code))
        report.append({'source': name, 'original_sha256': hashlib.sha256(before).hexdigest(),
                       'required_helpers_renamed': pruner.renamed, 'removed_functions': pruner.removed,
                       'result_sha256': hashlib.sha256(code.encode()).hexdigest(),
                       'status': 'edited_pending_regression'})
    # All transformations are prepared and syntactically checked before edits.
    for source, before, code in prepared:
        snapshot = out / source.relative_to(ROOT)
        if snapshot.exists():
            raise FileExistsError('isolation already attempted; review saved sources before rerunning')
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(before)
        source.write_text(code, encoding='utf-8', newline='\n')
    (out / 'changes.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps([{'source': r['source'], 'removed_count': len(r['removed_functions']),
                      'retained_helpers': r['required_helpers_renamed']} for r in report], indent=2))


if __name__ == '__main__':
    main()
