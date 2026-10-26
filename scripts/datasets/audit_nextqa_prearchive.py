"""Build a reviewable keep/hold/archive-candidate inventory; never move files."""
from __future__ import annotations

import ast
from collections import Counter, defaultdict, deque
import hashlib
import gzip
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'registry/audits/nextqa_prearchive'
TEXT_SUFFIXES={'.py','.json','.yaml','.yml','.md','.tex','.toml','.txt','.ps1'}
ENTRYPOINTS={
    'current_inference': ['scripts/datasets/nextqa_mainline.py','scripts/datasets/nextqa/retrieval/run_graph_retrieval_pilot.py',
        'scripts/datasets/nextqa/qa/run_graph_only_qa_pilot.py','scripts/datasets/nextqa/retrieval/build_visual_evidence_v2.py','scripts/datasets/nextqa/qa/run_visual_enhanced_qa.py'],
    'rebuild': ['scripts/datasets/nextqa/build/build_graph_full.py','scripts/datasets/nextqa/build/build_visual_indexes.py','scripts/datasets/nextqa/build/build_visual_pair_index_v2.py'],
    'final_C_recipe': ['scripts/datasets/nextqa/retrieval/build_causal_why_coherent_chain_evidence.py','scripts/datasets/nextqa/qa/run_causal_why_coherent_chain_qa.py',
        'scripts/datasets/nextqa/qa/merge_visual_strategy_predictions.py','scripts/datasets/nextqa/retrieval/build_visual_route_policy_v3.py','scripts/datasets/nextqa/qa/merge_visual_route_policy_v3.py'],
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def inside(path: str, root: str) -> bool:
    path,root=path.replace('\\','/').casefold(),root.replace('\\','/').rstrip('/').casefold()
    return path==root or path.startswith(root+'/')


def relative(value: str) -> str | None:
    text=value.replace('\\','/')
    prefix=ROOT.as_posix()
    if text.casefold().startswith(prefix.casefold()+'/'):
        return text[len(prefix)+1:]
    if text.startswith(('.scratch/nextqa/','data/next-qa/','configs/','src/','scripts/')):
        return text
    return None


def classify(path: str, copied: dict, holds: list[str]) -> str:
    if any(inside(path,h) or inside(h,path) for h in holds):
        return 'hold_rebuild_or_recipe_dependency'
    if any(inside(path,source) for source in copied):
        return 'archive_candidate_verified_copy'
    return 'review_nonselected_asset'


def source_closure():
    files={}
    for top in ('src','scripts'):
        for p in (ROOT/top).rglob('*.py'):
            rel=p.relative_to(ROOT)
            bits=list(rel.with_suffix('').parts)
            if bits[0]=='src':bits=bits[1:]
            if bits[-1]=='__init__':bits=bits[:-1]
            files['.'.join(bits)]=p
    queue=deque(p for paths in ENTRYPOINTS.values() for p in paths)
    seen=set();edges=[];errors=[]
    def enqueue(module,source):
        if module in files:
            target=files[module].relative_to(ROOT).as_posix()
            edges.append({'from':source,'to':target})
            queue.append(target)
        parts=module.split('.')
        for i in range(1,len(parts)):
            parent='.'.join(parts[:i])
            if parent in files and files[parent].name=='__init__.py':
                queue.append(files[parent].relative_to(ROOT).as_posix())
    while queue:
        rel=queue.popleft()
        if rel in seen:continue
        seen.add(rel);p=ROOT/rel
        if not p.is_file():errors.append({'path':rel,'error':'missing source'});continue
        try:tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        except (SyntaxError,UnicodeError) as exc:errors.append({'path':rel,'error':str(exc)});continue
        bits=list(Path(rel).with_suffix('').parts)
        if bits[0]=='src':bits=bits[1:]
        package=bits[:-1]
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                if node.level:
                    base=package[:len(package)-node.level+1]
                    if node.module:base+=node.module.split('.')
                    module='.'.join(base)
                else:module=node.module or ''
                modules=[module]+[module+'.'+a.name for a in node.names if a.name!='*']
            for module in modules:
                enqueue(module,rel)
                if '.' not in module:
                    sibling=p.parent/(module+'.py')
                    if sibling.is_file():queue.append(sibling.relative_to(ROOT).as_posix())
    return {'scope':'Conservative static import closure plus package initializers; dynamic imports may need further review.',
            'entrypoints':ENTRYPOINTS,'files':sorted(seen),'edges':edges,'errors':errors}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plans=[read(ROOT/'registry/migrations/nextqa_mainline_copy_plan.json'),
           read(ROOT/'registry/migrations/nextqa_repaired_source_copy_plan.json')]
    copied={e['source']:e['target'] for p in plans for e in p['entries']}
    rebuild_inputs=read(ROOT/'registry/migrations/nextqa_rebuild_inputs.json')
    for row in rebuild_inputs['rows']:
        copied[row['source']]=row['target']
    for bundle in rebuild_inputs.get('bundles',[]):
        copied[bundle['source_root']]=bundle['target_root']
    receipt=read(ROOT/'registry/migrations/nextqa_mainline_activation.json')
    receipt_blob=(ROOT/receipt['receipt']).read_bytes()
    if hashlib.sha256(receipt_blob).hexdigest()!=receipt['receipt_sha256']:
        raise RuntimeError('copy receipt fingerprint mismatch')
    copy_rows=json.loads(gzip.decompress(receipt_blob))['rows']
    copy_rows += read(ROOT/'registry/migrations/nextqa_repaired_source_copy_receipt.json')['rows']
    copy_rows += rebuild_inputs['rows']
    copy_source_set={row['source'] for row in copy_rows}
    for row in copy_rows:
        target=ROOT/row['target']
        if not target.is_file() or target.stat().st_size!=row['bytes']:
            raise RuntimeError(f'copied counterpart missing or size changed: {row["target"]}')
    for source in copied:
        if not any(inside(path,source) for path in copy_source_set):
            raise RuntimeError(f'no copy receipt evidence for {source}')
    facts=[]
    manifest=yaml.safe_load((ROOT/'registry/datasets/nextqa.yaml').read_text(encoding='utf-8'))
    # Check frozen recipe input fingerprints against migrated counterparts.
    for plan in plans:
        for entry in plan['entries']:
            cfg=ROOT/entry['target']/'run_config.json'
            if not cfg.is_file():continue
            contract=read(cfg).get('contract',{})
            for key,value in contract.items():
                if not isinstance(value,dict) or not isinstance(value.get('path'),str):continue
                path=relative(value['path'])
                if path is None:continue
                match=next((s for s in sorted(copied,key=len,reverse=True) if inside(path,s)),None)
                target=copied[match]+path[len(match):] if match else None
                role='historical_seed_or_resume_provenance' if key.startswith('seed') else 'recipe_input'
                exists=(ROOT/path).is_file()
                record={'run_config':cfg.relative_to(ROOT).as_posix(),'input_key':key,'path':path,
                        'role':role,'exists':exists,'copied_target':target,'recorded_sha256':value.get('sha256')}
                # Bound hashing cost: large source frames/graphs were verified during copy.
                if target and (ROOT/target).is_file() and (ROOT/target).stat().st_size<300*1024**2:
                    actual=hashlib.sha256((ROOT/target).read_bytes()).hexdigest()
                    record['copied_hash_matches_recorded']=actual==value.get('sha256')
                facts.append(record)
    executions={p.stem.removeprefix('execution_'):read(p) for p in OUT.glob('execution_*.json')}
    holds=[]
    for report in executions.values():
        for row in report['denied_legacy_accesses']:
            path=relative(row['path'])
            if path:holds.append(str(Path(path).parent).replace('\\','/'))
    holds.extend(f['path'] if Path(f['path']).suffix=='' else Path(f['path']).parent.as_posix()
                 for f in facts if f['role']=='recipe_input' and f['copied_target'] is None
                 and f['path'].startswith(('.scratch/nextqa/','data/next-qa/')))
    holds=sorted(set(holds))
    inventory=[]
    for base in (ROOT/'.scratch/nextqa',ROOT/'data/next-qa'):
        for stage in sorted(base.iterdir()):
            units=list(stage.iterdir()) if stage.is_dir() else [stage]
            if not units:units=[stage]
            for unit in sorted(units):
                rel=unit.relative_to(ROOT).as_posix();size=0;count=0
                if unit.is_file():size=unit.stat().st_size;count=1
                else:
                    for d,dirs,files in os.walk(unit):
                        dirs[:]=[x for x in dirs if not (Path(d,x).lstat().st_file_attributes & 0x400)]
                        for name in files:
                            path=Path(d,name);size+=path.stat().st_size;count+=1
                inventory.append({'path':rel,'bytes':size,'files':count,
                    'disposition':classify(rel,copied,holds),'may_delete':False})
    closure=source_closure()
    closure_files=set(closure['files'])
    observed_sources=set()
    for key in ('replay','resources','graph_mock'):
        for name in executions.get(key,{}).get('workspace_accesses',{}):
            path=Path(name)
            if '__pycache__' in path.parts and path.suffix=='.pyc':
                path=path.parent.parent/(path.name.split('.cpython-')[0]+'.py')
            if path.suffix=='.py' and not path.parts[0].startswith('.') and (ROOT/path).is_file():
                observed_sources.add(path.as_posix())
    closure['observed_project_sources']=sorted(observed_sources)
    retained_sources={p.casefold() for p in closure_files|observed_sources}
    code_inventory=[]
    for top in ('scripts/nextqa','src/vlkg/eval_adapters/nextqa'):
        for p in sorted((ROOT/top).rglob('*.py')):
            path=p.relative_to(ROOT).as_posix()
            code_inventory.append({'path':path,'disposition':'retain_dependency' if path.casefold() in retained_sources else 'review_not_in_observed_or_static_closure',
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'may_delete':False})
    # Global callers and mixed documentation: paths are references, not deletion instructions.
    references=[]
    for top in ('configs','scripts','src','tests','docs','TimeGraph','iclr2027_code_release_cg6','aaai27_paper_draft','iclr2027_paper'):
        for p in (ROOT/top).rglob('*'):
            if not p.is_file() or p.suffix not in TEXT_SUFFIXES:continue
            rel=p.relative_to(ROOT).as_posix()
            if rel.startswith('docs/plans/dataset_cleanup_20260928/') or 'tectonic_cache' in rel or p.stat().st_size>5*1024**2:continue
            text=p.read_text(encoding='utf-8-sig',errors='replace')
            hits=[i for i,line in enumerate(text.splitlines(),1) if ('.scratch' in line or 'data/next-qa' in line or 'data\\next-qa' in line) and ('nextqa' in line or 'next-qa' in line)]
            if hits:references.append({'path':rel,'lines':hits,'in_static_closure':rel in closure_files,
                                       'action':'review_active_argument_override' if rel in closure_files else 'seal_or_rewrite_reference_before_old_root_retirement'})
    totals=Counter();counts=Counter()
    for row in inventory:totals[row['disposition']]+=row['bytes'];counts[row['disposition']]+=1
    summary={'status':'not_ready_for_blanket_archive','current_inference_guarded_cases':{k:v['status'] for k,v in executions.items()},
             'hold_roots':holds,'inventory_units':len(inventory),'inventory_by_disposition':dict(counts),
             'logical_bytes_by_disposition':dict(totals),'static_source_closure_files':len(closure_files),
             'observed_project_source_files':len(observed_sources),
             'observed_third_party_source_files':sorted(p for p in observed_sources if p.startswith(('graphiti-main/','cognee-main/'))),
             'nextqa_source_files':len(code_inventory),'global_reference_files':len(references),
             'recipe_input_fingerprints':len(facts),'unmigrated_recipe_inputs':[r for r in facts if r['role']=='recipe_input' and r['copied_target'] is None],
             'historical_fingerprint_mismatches':[r for r in facts if r.get('copied_hash_matches_recorded') is False],
             'copy_receipt_rows_rechecked_for_existence_and_size':len(copy_rows),
             'may_delete':False,'archive_candidates_are_not_approved_actions':True,
             'limits':['Static closure over-approximates imported code and does not prove dynamic completeness.',
                       'Execution guard covers only exercised offline branches, not full live QA or index regeneration.',
                       'Archived reports/resume provenance require preservation even when no runtime read occurs.',
                       'Copy receipt signatures and destination sizes were checked; large payload bytes were not all rehashed in this audit.',
                       'Review_nonselected_asset is not permission to remove it.']}
    write('recipe_inputs.json',facts);write('asset_inventory.json',inventory);write('code_closure.json',closure)
    write('code_inventory.json',code_inventory);write('global_references.json',references);write('summary.json',summary)
    lines=['# NExT-QA 封存前依赖审计','',
      '**结论：当前推理的已测路径可以脱离旧数据目录，但尚不能整体封存旧目录。**','',
      '## 动态验证','', '| 场景 | 结果 |','| --- | --- |']
    lines += [f'| {k} | {v["status"]} |' for k,v in executions.items()]
    lines += ['', '## 明确保留的旧目录', '']+[f'- `{p}`' for p in holds]
    lines += ['', '## 资产分类', '', '| 分类 | 单元数 | 逻辑 GiB |', '| --- | ---: | ---: |']
    lines += [f'| {key} | {counts[key]} | {totals[key]/1024**3:.3f} |' for key in sorted(counts)]
    lines += ['', '大小按文件逻辑长度统计；同盘封存不会释放这些空间。没有生成可执行删除清单。', '',
      '## 下一步', '',
      '1. 按 nextqa_archive_disposition.json 逐项处理非主线资产、调用方和混合报告。',
      '2. 保留最终 C 的原始依赖、修复过程、seed/resume 合同及已验证的重组合证据。',
      '3. 将静态闭包中的必要模块保留；混有主线和历史策略的模块拆分后再封存。',
      '4. 更新跨项目报告、测试和 release 副本引用，才可将旧目录整体封存。', '',
      '两份旧 run_config 的 source manifest hash 与最终修复版不同；原记录须保留，不允许改写成当前 hash。', '',
      ('运行时仍观测到本地第三方源码；须保留观测依赖。' if summary['observed_third_party_source_files'] else
       '本轮已执行场景没有读取本地 graphiti-main/cognee-main 源码；这不证明其他显式 backend 入口无需这些资产。'), '',
      f'静态保留闭包为 {len(closure_files)} 个源码文件，跨目录旧路径引用涉及 {len(references)} 个文件。详见旁边 JSON 清单。',
      '', '## 范围限制','',*['- '+s for s in summary['limits']]]
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='unmigrated_recipe_inputs'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
