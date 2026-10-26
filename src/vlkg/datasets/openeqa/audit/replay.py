"""Validate frozen paired judge records and recompute their aggregate scores."""
import json
from pathlib import Path
from statistics import mean
from vlkg.core.assets.asset_registry import asset_file
from vlkg.workspace import workspace_root


def replay():
    root = workspace_root()
    arms, ids = {}, {}
    for arm in 'ABC':
        directory = root / 'datasets/openeqa/reports/openeqa-reports-v1/payload/judges' / arm
        scores = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
        summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
        predictions = json.loads(asset_file('openeqa', 'runs', arm).read_text(encoding='utf-8'))
        ids[arm] = {r['question_id'] for r in scores}
        if len(ids[arm]) != len(scores) or len(scores) != 1636:
            raise ValueError('judge question set changed')
        by_id = {r['question_id']: r for r in predictions}
        if set(by_id) != ids[arm]:
            raise ValueError('prediction and judge question sets differ')
        for row in scores:
            predicted = by_id[row['question_id']]
            actual = predicted.get('prediction') or predicted.get('pred_answer') or predicted.get('answer')
            if str(actual).strip() != str(row['prediction']).strip():
                raise ValueError('judge record no longer matches its prediction')
        normalized = round(mean(row['normalized_score'] for row in scores), 4)
        if normalized != summary['normalized_score_mean']:
            raise ValueError('stored judge aggregation differs')
        arms[arm] = {'total': len(scores), 'normalized_score_mean': normalized}
    if not (ids['A'] == ids['B'] == ids['C']):
        raise ValueError('matched A/B/C coverage differs')
    return {'dataset': 'openeqa', 'arms': arms, 'scope': 'Frozen project-defined LLM-Judge scores; no new judging.', 'calls_api': False}
