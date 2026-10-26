"""Update existing tests to canonical implementation/config paths after relocation."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import shutil

from scripts.datasets.migrate_asset_storage import ROOT, write_json


def replace_chain(text, old, new):
    parts = old.split('/')
    pattern = r'\s*/\s*'.join(r'''["']''' + re.escape(part) + r'''["']''' for part in parts)
    replacement = ' / '.join(repr(part) for part in new.split('/'))
    return re.sub(pattern, replacement, text)


def main():
    mappings = {}
    for dataset in ('navqa', 'nextqa', 'openeqa'):
        for file in (ROOT / f'scripts/datasets/{dataset}').rglob('*.py'):
            if file.name != '__init__.py':
                mappings[f'scripts/{dataset}/{file.name}'] = file.relative_to(ROOT).as_posix()
        for file in (ROOT / f'src/vlkg/datasets/{dataset}').rglob('*.py'):
            if file.name != '__init__.py':
                mappings[f'src/vlkg/eval_adapters/{dataset}/{file.name}'] = file.relative_to(ROOT).as_posix()
    mappings.update({
        'scripts/eval/run_navqa_official_qa_on_canonical_graph.py': 'scripts/datasets/navqa/qa/run_navqa_official_qa_on_canonical_graph.py',
        'scripts/eval/run_navqa_llm_question_ir_experiment.py': 'scripts/datasets/navqa/retrieval/build_question_ir.py',
        'scripts/nextqa/build_graph_pilot.py': 'scripts/datasets/nextqa/build/build_graph_full.py',
        'configs/nextqa_cg6_local_tracklet_v1.yaml': 'configs/datasets/nextqa/graph.yaml',
        'configs/nextqa_graph_qa_luna_v1.yaml': 'configs/datasets/nextqa/graph_qa.yaml',
        'configs/nextqa_graph_qa_luna_evidence24_v1.yaml': 'configs/datasets/nextqa/retrieval.yaml',
        'configs/nextqa_graph_only_unified_v1.yaml': 'configs/datasets/nextqa/no_routing_qa.yaml',
        'configs/nextqa_visual_qa_luna_v2.yaml': 'configs/datasets/nextqa/visual_qa.yaml',
        'configs/navqa_frozen_reference_v1.json': 'archive/legacy_20260928/shared/information/root_configs/navqa_frozen_reference_v1.json',
    })
    changes = []
    for p in (ROOT / 'tests').rglob('*.py'):
        text = original = p.read_text(encoding='utf-8')
        for old, new in mappings.items():
            text = replace_chain(text, old, new)
            text = text.replace(repr(old), repr(new)).replace('"' + old + '"', '"' + new + '"')
        text = text.replace('registry/migrations/', 'archive/legacy_20260928/governance/registry/migrations/')
        text = text.replace('registry/audits/nextqa_prearchive/summary.json', 'archive/legacy_20260928/governance/registry/audits/nextqa_prearchive/summary.json')
        text = text.replace('registry/gates/nextqa_prearchive_gate.json', 'archive/legacy_20260928/governance/registry/gates/nextqa_prearchive_gate.json')
        if p.name == 'test_isolation.py':
            text = replace_chain(text, 'src/vlkg/eval_adapters/nextqa/visual_runtime_v1', 'src/vlkg/datasets/nextqa/runtime/visual_runtime_v1')
            text = replace_chain(text, 'scripts/nextqa', 'scripts/datasets/nextqa')
            for name in ('uses_visual_embeddings_for_graph_construction', 'qa_reads_rgb', 'qa_reads_object_crop', 'qa_reads_embedding_cache', 'qa_uses_faiss'):
                text = text.replace('"' + name + '":', "'" + name + "':")
        if p.name == 'test_openeqa_ir_first_single_route.py':
            text = text.replace('vlkg.eval_adapters.openeqa.visual_runtime_v5', 'vlkg.datasets.openeqa.runtime.visual_runtime_v5')
        if text != original:
            snapshot = ROOT / 'archive/legacy_20260928/shared/information/test_paths_before_closeout' / p.relative_to(ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            if snapshot.exists():
                raise FileExistsError(snapshot)
            snapshot.write_text(original, encoding='utf-8')
            p.write_text(text, encoding='utf-8')
            changes.append(p.relative_to(ROOT).as_posix())
    prompt_source = ROOT / 'archive/legacy_20260928/shared/code/old_script_locations/openeqa/prompts'
    prompt_target = ROOT / 'scripts/datasets/openeqa/audit/prompts'
    prompt_target.mkdir(parents=True, exist_ok=True)
    for file in prompt_source.glob('*.txt'):
        target = prompt_target / file.name
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(file, target)
    write_json(ROOT / 'registry/audits/test_paths_closeout.json', {
        'status': 'updated_pending_regression', 'files': changes,
        'scope': 'Workspace/code/config path updates and unchanged evaluator prompt copies; no production graph/QA algorithm changes.'})
    print(json.dumps({'test_files_updated': len(changes), 'judge_prompts_restored': len(list(prompt_target.glob('*.txt')))}))


if __name__ == '__main__':
    main()
