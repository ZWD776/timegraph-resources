"""Re-evaluate selected NExT-QA predictions with the dataset evaluator."""
import json
from vlkg.core.assets.asset_registry import asset_file
from vlkg.datasets.nextqa.qa.evaluation import evaluate_exact_choice_accuracy


def replay():
    def load(layer, name):
        return json.loads(asset_file('nextqa', layer, name).read_text(encoding='utf-8'))
    questions, labels = load('source', 'questions'), load('source', 'labels')
    arms = {}
    for arm in 'ABC':
        result = evaluate_exact_choice_accuracy(load('runs', arm), labels, questions)
        arms[arm] = {'correct': result['correct_question_count'], 'total': len(questions), 'accuracy': result['overall_accuracy']}
    if [arms[a]['correct'] for a in 'ABC'] != [3024,3257,3334]:
        raise ValueError('selected NExT-QA evaluation identities changed')
    return {'dataset': 'nextqa', 'arms': arms, 'scope': 'Exact-choice evaluator replay.', 'calls_api': False}
