"""Replay frozen NAVQA official-style evaluation evidence without a judge call."""
import json
from vlkg.core.assets.asset_registry import asset_file


def replay():
    arms = {}
    for arm in ('A', 'B', 'C'):
        data = json.loads(asset_file('navqa', 'runs', arm).read_text(encoding='utf-8'))
        if arm == 'C':
            metric = data['navqa_official_style_accuracy']
            correct, total = metric['correct'], metric['total']
        else:
            correct = sum(row['new']['correct'] for row in data['rows'])
            total = sum(row['new']['total'] for row in data['rows'])
        arms[arm] = {'correct': correct, 'total': total, 'accuracy': correct / total}
    if [(arms[a]['correct'], arms[a]['total']) for a in 'ABC'] != [(73,139),(87,139),(101,139)]:
        raise ValueError('selected NAVQA evaluation identities changed')
    return {'dataset': 'navqa', 'arms': arms, 'scope': 'Frozen official-style evaluation replay; no new LLM judging.', 'calls_api': False}
