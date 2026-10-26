"""Classify old assets and references into a non-executable archive proposal."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'registry/audits/nextqa_prearchive'
ARCHIVE = 'archive/legacy_20260928/nextqa'

# Each group was reviewed against the retained graph/IR/retrieval/A/B/C recipe.
# Original repair and seed records remain evidence even when not runtime inputs.
STAGES = {
    'graph_pilots': ('historical_graph_experiment', '早期或小规模建图试验；当前仅使用 human v2 full570。'),
    'graph_retrieval': ('historical_retrieval_provenance', '旧检索或补丁检索；当前 IRv3 b 已复制，旧 seed/resume 记录原样保留。'),
    'graph_retrieval_pilots': ('historical_retrieval_experiment', '小规模或替代检索分支，未选入当前主线。'),
    'graphs_special': ('repair_provenance', '29 视频修复中间图；修复结果已并入当前 570 视频图，保留中间证据。'),
    'in_place_repairs': ('repair_provenance', '原位修复过程及备份；不得改写原记录或以当前 hash 替代历史 hash。'),
    'special_patch_overlays': ('repair_provenance', '旧 29 视频补丁预测；当前 A/B/C 使用已修复完整主线。'),
    'special_patch_preparation': ('repair_provenance', '旧补丁准备材料，封存以保持修复过程可追溯。'),
    'special_splice': ('repair_provenance', '未选中的 normalization dry-run 和准备报告；实际 HSGG 输入已另登记保留。'),
    'official_raw_check_20260816': ('source_validation_provenance', '官方视频校验、下载和代表帧；当前 official 标签及修复输入已独立保留。'),
    'question_ir': ('historical_ir_provenance', 'IRv2、IRv3 a seed 或试运行；当前使用 IRv3 b，原 seed 合同必须封存保留。'),
    'visual_evidence': ('historical_visual_strategy', '旧视觉证据、causal v1、temporal-next 分支或 pilot；最终 C 的 causal v2 已保留。'),
    'visual_routes': ('historical_routes', '旧 routes v1；当前 routes v2 已保留。'),
    'visual_indexes': ('historical_index_experiment', '小规模 v1 索引试验；当前 strict-pairs v2 及所需 v1 shards 已保留。'),
    'visual_graph_baseline': ('historical_baseline', '旧视觉图基线 v1；当前冻结基线 v2 已保留。'),
    'qa': ('historical_qa_strategy', '早期 QA、mock、补丁或未选策略输出；当前 A/B/C 及最终 C 依赖另有保留副本。'),
    'qa_pilots': ('historical_qa_experiment', '小规模 QA 或 causal v1；当前 causal v2 全 1927 题输出另行保留。'),
    'qa_evaluations': ('historical_evaluation', '旧版本、pilot、替代组合评测；当前 A/B/C 评测已保留。'),
    'qa_error_audits': ('historical_diagnostic', '错误诊断及版本比较报告，封存原始结果。'),
    'ablation': ('historical_ablation', '未选 no-routing/core 子集或 pilot；选定 B 与正式 A/B/C ablation 已保留。'),
    'audit': ('historical_diagnostic', '原始源校验和修复审计报告，封存保留。'),
    'visual_readiness': ('historical_diagnostic', '旧视觉可运行性诊断，当前执行审计已单独登记。'),
    'engineering': ('historical_engineering', '早期关联诊断、跨数据集比较和 endpoint 检查脚本。'),
    'external_baselines': ('historical_external_baseline', '外部基线分析资产，未作为当前主线运行输入。'),
    'dry_run': ('historical_smoke', '早期 dry-run 输出。'),
    'pytest': ('historical_smoke', '早期测试临时输出。'),
    'state_snapshots': ('historical_snapshot', '原始工作状态快照；封存保留，不恢复覆盖当前代码。'),
    'launcher_logs': ('historical_logs', '旧运行启动日志与 pid 记录。'),
    '__pycache__': ('historical_temporary', '旧审计字节码；本阶段与其他历史资料一并封存，不删除。'),
    '_tmp_predictions_copy.txt': ('historical_temporary', '旧临时预测副本；本阶段封存，不删除。'),
    'audit_special_hsgg_patches_20260816.py': ('repair_provenance', '修复审计原脚本，封存以保留原过程。'),
}


def archive_target(source):
    source = source.replace('\\', '/')
    parts = source.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError(f'unsafe source: {source}')
    if source.startswith('.scratch/nextqa/'):
        suffix = 'scratch/nextqa/' + source[len('.scratch/nextqa/'):]
    elif source.startswith('data/next-qa/'):
        suffix = source
    else:
        raise ValueError(f'outside reviewed old roots: {source}')
    return ARCHIVE + '/assets/' + suffix


def reference_action(row):
    path = row['path']
    if path.startswith('scripts/datasets/'):
        return 'retain_expected_audit_control_reference'
    if path == 'configs/dataset_asset_registry.yaml':
        return 'retain_original_location_provenance'
    if path.startswith('src/') or row['in_static_closure']:
        return 'migrate_active_defaults_preserve_resume_contract'
    if path.startswith('tests/'):
        return 'review_fixture_or_compatibility_reference'
    if path.startswith(('docs/', 'aaai27_paper_draft/', 'iclr2027_paper/')):
        return 'snapshot_then_rewrite_current_document'
    if path.startswith('iclr2027_code_release_cg6/'):
        return 'seal_release_snapshot_after_cross_dataset_review'
    return 'review_shared_callers_then_seal_historical_code'


def main():
    inventory = json.loads((AUDIT / 'asset_inventory.json').read_text(encoding='utf-8'))
    refs = json.loads((AUDIT / 'global_references.json').read_text(encoding='utf-8'))
    assets = []
    for row in inventory:
        source = row['path']
        target = archive_target(source)
        if (ROOT / target).exists():
            raise FileExistsError(f'archive target already exists: {target}')
        if row['disposition'] == 'archive_candidate_verified_copy':
            category, reason = 'verified_mainline_original', '当前主线或依赖已复制；旧原件封存，历史合同原样保留。'
        elif row['disposition'] == 'review_nonselected_asset':
            stage = source.split('/')[2]
            if stage not in STAGES:
                raise ValueError(f'unreviewed asset stage: {stage}')
            category, reason = STAGES[stage]
        else:
            raise ValueError(f'unresolved runtime hold: {source}')
        assets.append({**row, 'proposed_disposition': 'seal_preserving_bytes',
                       'category': category, 'reason_cn': reason, 'archive_target': target,
                       'move_allowed': False, 'may_delete': False})
    classified_refs = []
    for row in refs:
        p = ROOT / row['path']
        classified_refs.append({**row, 'proposed_action': reference_action(row),
                                'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                'rewrite_or_move_allowed': False})
    counts = Counter(r['category'] for r in assets)
    total_bytes = Counter()
    for row in assets:
        total_bytes[row['category']] += row['bytes']
    report = {
        'contract': 'nextqa_archive_disposition_v1', 'status': 'classified_proposal_reference_migration_pending',
        'archive_root': ARCHIVE, 'asset_units': len(assets), 'unclassified_units': 0,
        'asset_counts': dict(counts), 'asset_bytes': dict(total_bytes),
        'reference_files': len(classified_refs),
        'reference_actions': dict(Counter(r['proposed_action'] for r in classified_refs)),
        'assets': assets, 'references': classified_refs,
        'move_allowed': False, 'deletion_allowed': False,
        'execution_prerequisites': [
            'Migrate or explicitly seal callers and mixed documents before retiring old roots.',
            'Recheck source existence, overlap, reparse points, destination collisions and file hashes immediately before any move.',
            'Keep historical seed/resume manifests and fingerprints immutable inside the archive.',
            'Preserve full model rebuild/live QA uncertainty; offline probes do not cover every runtime branch.',
        ],
        'scope': 'Classification only; this command never moves, deletes or rewrites source assets.'}
    out = ROOT / 'registry/migrations/nextqa_archive_disposition.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    summary = json.loads((AUDIT / 'summary.json').read_text(encoding='utf-8'))
    gate_path = ROOT / 'registry/gates/nextqa_prearchive_gate.json'
    gate = json.loads(gate_path.read_text(encoding='utf-8'))
    gate.update({
        'contract': 'nextqa_prearchive_gate_v3',
        'status': 'assets_classified_reference_migration_pending',
        'archive_disposition': out.relative_to(ROOT).as_posix(),
        'shared_code_hold': sorted({p.split('/')[0] for p in summary['observed_third_party_source_files']}),
        'shared_code_retirement_allowed': False,
        'shared_code_scope': 'Observed six offline NExT-QA cases only; explicit backends and other datasets still require review.',
        'unclassified_asset_units': 0, 'unclassified_asset_bytes': 0,
        'nonmainline_seal_proposal_units': sum(r['category'] != 'verified_mainline_original' for r in assets),
        'nonmainline_seal_proposal_bytes': sum(r['bytes'] for r in assets if r['category'] != 'verified_mainline_original'),
        'global_reference_files': len(classified_refs),
        'reference_action_counts': report['reference_actions'],
        'remaining_blockers': [
            'Active defaults, historical callers, mixed documents and release copies still require migration or sealing.',
            'Archive proposals require a final file-hash, overlap and reparse-point check before any move.',
            'Full model-backed index rebuild and live QA were not rerun.',
            'Historical run fingerprints must remain immutable.',
        ],
    })
    gate_path.write_text(json.dumps(gate, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    metadata = ROOT / 'datasets/nextqa/reports/nextqa-reports-v1/metadata'
    for name, value in [('nextqa_prearchive_gate.json', gate), ('nextqa_prearchive_summary.json', summary)]:
        (metadata / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    lines = ['# NExT-QA 旧资产封存分流方案', '',
             '241 个旧资产单元均已逐项登记拟封存地址。此清单不授权移动或删除。', '',
             '| 用途分类 | 单元数 | GiB |', '| --- | ---: | ---: |']
    lines += [f'| {key} | {counts[key]} | {total_bytes[key]/1024**3:.3f} |' for key in sorted(counts)]
    lines += ['', '## 引用处理', '', '| 处理方式 | 文件数 |', '| --- | ---: |']
    lines += [f'| {key} | {value} |' for key, value in sorted(report['reference_actions'].items())]
    lines += ['', '修复过程、旧 IR seed、运行合同和评测证据均保留原字节。旧主线原件的封存不会释放空间。', '',
              '仍需处理调用方、混合文档与 release 副本；在这些工作完成之前，旧数据根不能整体搬走。', '',
              '本地 Graphiti 是否属于当前导入链，以最新执行审计为准；其他历史或显式 backend 入口不在六项离线审计覆盖范围内。']
    (AUDIT / 'ARCHIVE_DISPOSITION_CN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('assets', 'references')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
