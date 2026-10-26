"""Question-only semantic IR for the NExT-QA multiple-choice task."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping
TOKEN_RE = re.compile('[a-z0-9]+')
QUESTION_ID_RE = re.compile('^nextqa-val/(?P<video_id>[^/]+)/q[^/]+$')
ACTOR_TERMS = ('camera person', 'the person holding the camera', 'children', 'people', 'woman', 'lady', 'girl', 'boy', 'baby', 'men', 'man', 'person', 'driver')
OBJECT_TERMS = ('toy cars', 'toy car', 'play area', 'brown table', 'grey seat', 'purple thing', 'opposite road', 'camera', 'bicycle', 'bike', 'vehicles', 'vehicle', 'truck', 'barrier', 'window', 'stairs', 'sandals', 'sofa', 'table', 'ground', 'slope', 'floor', 'guitar', 'trumpet', 'restaurant', 'seat', 'soil', 'wheel', 'road', 'room', 'feet', 'cars', 'car', 'fan', 'dog', 'flower', 'bottle', 'train', 'fence', 'porch', 'scooter', 'keyboard', 'music', 'lights')
ACTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (('hang_out', ('hanging out', 'hang out')), ('look', ('looked', 'looking', 'looks', 'look')), ('move', ('moved', 'moving', 'moves', 'move')), ('hold', ('holding', 'held', 'hold')), ('stop', ('stationary', 'stopped', 'stops', 'stop')), ('cycle', ('cycling', 'cycled', 'cycle')), ('ride', ('riding', 'rode', 'ride')), ('bend', ('bending', 'bent', 'bend')), ('approach', ('approached', 'approaching', 'approach')), ('point', ('pointed', 'pointing', 'point')), ('walk', ('walked', 'walking', 'walk')), ('put', ('putting', 'put')), ('play', ('playing', 'played', 'play')), ('pass', ('passing', 'passed', 'pass')), ('come', ('came', 'comes', 'come')), ('take', ('taken', 'took', 'take')), ('get', ('getting', 'got', 'get')), ('carry', ('carried', 'carrying', 'carry')), ('transport', ('transporting', 'transport')), ('signal', ('signaled', 'signalling', 'signal')), ('place', ('placed', 'place')), ('react', ('reacted', 'react')), ('film', ('filmed', 'filming', 'film')), ('drive', ('drove', 'driven', 'driving', 'drive')), ('smile', ('smiling', 'smiled', 'smile')), ('flash', ('flashing', 'flashed', 'flash')), ('sit', ('sitting', 'sat', 'sit')), ('wave', ('waving', 'waved', 'wave')), ('talk', ('talking', 'talked', 'talk')), ('dance', ('dancing', 'danced', 'dance')), ('perform', ('performance', 'performing', 'perform')), ('touch', ('touching', 'touched', 'touch')), ('support', ('supporting', 'supported', 'support')), ('feel', ('feeling', 'felt', 'feel')))
TEMPORAL_MARKERS = (('after', 'after'), ('right before', 'before'), ('before', 'before'), ('while', 'concurrent'), ('when', 'concurrent'), (' as ', 'concurrent'))

@dataclass(frozen=True)
class QuestionRetrievalInput:
    """Strict projection consumed by graph retrieval."""
    question_id: str
    video_id: str
    normalized_question: str
    target_entities: tuple[str, ...]
    actors: tuple[str, ...]
    actions: tuple[str, ...]
    objects: tuple[str, ...]
    temporal_direction: str
    temporal_anchor: str
    referenced_event: str
    causal_query: str
    count_query: bool
    location_query: bool
    attribute_query: str
    answer_format: str

    def to_dict(self) -> dict[str, Any]:
        return {'question_id': self.question_id, 'video_id': self.video_id, 'normalized_question': self.normalized_question, 'target_entities': list(self.target_entities), 'actors': list(self.actors), 'actions': list(self.actions), 'objects': list(self.objects), 'temporal_direction': self.temporal_direction, 'temporal_anchor': self.temporal_anchor, 'referenced_event': self.referenced_event, 'causal_query': self.causal_query, 'count_query': self.count_query, 'location_query': self.location_query, 'attribute_query': self.attribute_query, 'answer_format': self.answer_format}

def normalize_question_text(value: str) -> str:
    return ' '.join(str(value or '').strip().lower().split()).rstrip('?.!')

def build_question_ir_record(row: Mapping[str, Any]) -> dict[str, Any]:
    question_id = str(row.get('question_id') or '').strip()
    video_id = str(row.get('video_id') or '').strip()
    match = QUESTION_ID_RE.fullmatch(question_id)
    if match is None or match.group('video_id') != video_id:
        raise ValueError(f'invalid stable NExT-QA question identity: {question_id!r}')
    question = normalize_question_text(str(row.get('normalized_question') or ''))
    if not question:
        raise ValueError(f'question text is empty: {question_id}')
    choices = [str(value).strip() for value in row.get('answer_choices', [])]
    if len(choices) != 5 or any((not value for value in choices)):
        raise ValueError(f'question must have five non-empty choices: {question_id}')
    actors = _terms_in_text(question, ACTOR_TERMS)
    objects = _terms_in_text(question, OBJECT_TERMS)
    actions = _actions_in_text(question)
    temporal_direction = _temporal_direction(question)
    temporal_anchor = _temporal_anchor(question)
    referenced_event = _referenced_event(question)
    causal_query = _causal_query(question)
    targets = _dedupe((*actors, *objects))
    return {'schema_version': 'nextqa_question_ir_v2', 'question_id': question_id, 'video_id': video_id, 'official_type': str(row.get('official_type') or '').strip(), 'normalized_question': question, 'target_entities': list(targets), 'actors': list(actors), 'actions': list(actions), 'objects': list(objects), 'temporal_direction': temporal_direction, 'temporal_anchor': temporal_anchor, 'referenced_event': referenced_event, 'causal_query': causal_query, 'count_query': question.startswith('how many '), 'location_query': question.startswith('where '), 'attribute_query': _attribute_query(question), 'answer_choices': choices, 'answer_format': str(row.get('answer_format') or 'multiple_choice_5'), 'ir_builder': 'nextqa_question_only_rule_v2', 'retrieval_uses_official_type': False, 'retrieval_uses_answer_choices': False, 'uses_gold_answer': False}

def build_question_ir_records(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    records = [build_question_ir_record(row) for row in rows]
    ids = [record['question_id'] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError('question runtime contains duplicate question IDs')
    return records

def retrieval_input_from_ir(record: Mapping[str, Any]) -> QuestionRetrievalInput:
    """Discard choices and diagnostic type before graph retrieval."""
    return QuestionRetrievalInput(question_id=str(record['question_id']), video_id=str(record['video_id']), normalized_question=normalize_question_text(str(record['normalized_question'])), target_entities=tuple(_strings(record.get('target_entities'))), actors=tuple(_strings(record.get('actors'))), actions=tuple(_strings(record.get('actions'))), objects=tuple(_strings(record.get('objects'))), temporal_direction=str(record.get('temporal_direction') or 'none'), temporal_anchor=str(record.get('temporal_anchor') or 'none'), referenced_event=str(record.get('referenced_event') or ''), causal_query=str(record.get('causal_query') or 'none'), count_query=bool(record.get('count_query', False)), location_query=bool(record.get('location_query', False)), attribute_query=str(record.get('attribute_query') or 'none'), answer_format=str(record.get('answer_format') or 'multiple_choice_5'))

def build_retrieval_ir_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [retrieval_input_from_ir(record).to_dict() for record in records]

def retrieval_input_fingerprint(record: Mapping[str, Any]) -> str:
    payload = retrieval_input_from_ir(record).to_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()

def audit_nonsemantic_field_isolation(record: Mapping[str, Any]) -> dict[str, bool]:
    baseline = retrieval_input_fingerprint(record)
    type_perturbed = deepcopy(dict(record))
    type_perturbed['official_type'] = '__PERTURBED_TYPE__'
    choices_perturbed = deepcopy(dict(record))
    choices_perturbed['answer_choices'] = [f'perturbed-{index}' for index in range(5)]
    return {'official_type_invariant': baseline == retrieval_input_fingerprint(type_perturbed), 'answer_choices_invariant': baseline == retrieval_input_fingerprint(choices_perturbed)}

def _terms_in_text(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for term in terms:
        match = re.search(f'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])', text)
        if match is not None:
            matches.append((match.start(), term))
    matches.sort(key=lambda item: (item[0], -len(item[1]), item[1]))
    accepted: list[str] = []
    for _, term in matches:
        if any((term != current and term in current for current in accepted)):
            continue
        accepted.append(term)
    return tuple(accepted)

def _actions_in_text(text: str) -> tuple[str, ...]:
    matches: list[tuple[int, str]] = []
    for action, aliases in ACTION_ALIASES:
        starts = [match.start() for alias in aliases if (match := re.search(f'(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])', text)) is not None]
        if starts:
            matches.append((min(starts), action))
    actions = [action for _, action in sorted(matches)]
    if 'carried out' in text and 'carry' in actions:
        actions.remove('carry')
    if re.search('\\bvideo\\s+(?:(?:is|was)\\s+)?taken\\b', text) and 'take' in actions:
        actions.remove('take')
    return tuple(actions)

def _temporal_direction(text: str) -> str:
    for marker, direction in TEMPORAL_MARKERS:
        if marker.strip() in {'after', 'before', 'right before', 'while', 'when'}:
            if re.search(f'\\b{re.escape(marker)}\\b', text):
                return direction
        elif marker in text:
            return direction
    return 'none'

def _temporal_anchor(text: str) -> str:
    if 'beginning' in text:
        return 'video_beginning'
    if 'middle' in text:
        return 'video_middle'
    if 'at the end' in text or 'end of the video' in text:
        return 'video_end'
    return 'none'

def _referenced_event(text: str) -> str:
    match = re.search('\\b(?:after|before|while|when|as)\\b\\s+(.+)', text)
    if match is None:
        return ''
    value = match.group(1)
    value = re.sub('\\s+(?:at|near|in)\\s+(?:the\\s+)?(?:beginning|middle|end)(?:\\s+of\\s+the\\s+video)?$', '', value)
    return value.strip()

def _causal_query(text: str) -> str:
    if text.startswith('why '):
        return 'why'
    if text.startswith('how ') and (not text.startswith('how many ')):
        return 'how'
    return 'none'

def _attribute_query(text: str) -> str:
    if re.search('\\b(?:color|colour)\\b', text):
        return 'color'
    return 'none'

def _strings(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]

def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys((value for value in values if value)))
