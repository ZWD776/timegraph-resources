from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from vlkg.datasets.navqa.qa.position_reference import FIELD_CATEGORY, FIELD_PARSABLE_ANSWER, FIELD_QUESTION, FIELD_SEQ_ID, FIELD_TEXT_ANSWER, FIELD_TIMESTAMP, FIELD_TYPE, FIELD_UUID, timestamp_from_hms_near_reference

@dataclass(frozen=True)
class OfficialAnswerLabel:
    uuid: str
    question_type: str
    question_category: str
    evaluation_key: str
    value: Any
    display: str
    semantics: str
    uses_official_answer_timestamp: bool = False
    inject_into_graph: bool = False

def load_position_labels(reference: dict[str, Any] | None) -> dict[str, list[float]]:
    if not reference:
        return {}
    labels: dict[str, list[float]] = {}
    for item in reference.get('position_answers', []):
        if not isinstance(item, dict):
            continue
        uuid = str(item.get('uuid', '')).strip()
        position = item.get('position')
        if not uuid or not isinstance(position, list | tuple) or len(position) != 3:
            continue
        labels[uuid] = [float(position[0]), float(position[1]), float(position[2])]
    return labels

def materialize_official_answer_label(row: dict[str, str], *, question_metadata: dict[str, dict[str, Any]] | None=None, position_labels: dict[str, list[float]] | None=None) -> OfficialAnswerLabel:
    """Build an official-format NaVQA label without adding data to graph memory."""
    metadata = question_metadata or {}
    positions = position_labels or {}
    uuid = str(row.get(FIELD_UUID, '')).strip()
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    category = str(row.get(FIELD_CATEGORY, '')).strip()
    if qtype == 'position':
        value = positions.get(uuid)
        if value is None:
            display = '[position]'
        else:
            display = '[' + ', '.join((f'{item:.6f}' for item in value)) + ']'
        return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key='position', value=value, display=display, semantics='ego_pose_at_semantic_event_time', uses_official_answer_timestamp=value is not None)
    if qtype == 'time':
        qa_item = metadata.get(uuid)
        timestamps = _answer_timestamp_values(row.get(FIELD_TIMESTAMP, ''))
        value: float | None = None
        if qa_item is not None and len(timestamps) == 1:
            event_time = timestamp_from_hms_near_reference(timestamps[0], question_start_time=float(qa_item['start_time']), reference_time=float(qa_item.get('end_time', qa_item['start_time'])))
            value = round((float(qa_item['end_time']) - event_time) / 60.0, 2)
        return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key='time', value=value, display=f'{value} minutes ago' if value is not None else '[minutes] ago', semantics='minutes_before_question_end_time', uses_official_answer_timestamp=value is not None)
    if qtype == 'duration':
        raw_value = str(row.get(FIELD_PARSABLE_ANSWER, '')).strip()
        value = _float_or_none(raw_value)
        return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key='duration', value=value, display=f'{value:g} minutes' if value is not None else raw_value, semantics='event_duration_minutes', uses_official_answer_timestamp=False)
    if qtype == 'binary':
        value = str(row.get(FIELD_PARSABLE_ANSWER, '')).strip().lower()
        return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key='binary', value=value or None, display=value or '', semantics='yes_no', uses_official_answer_timestamp=False)
    if qtype == 'text':
        value = str(row.get(FIELD_TEXT_ANSWER, '')).strip()
        return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key='text', value=value, display=value, semantics='free_text_human_judged', uses_official_answer_timestamp=False)
    return OfficialAnswerLabel(uuid=uuid, question_type=qtype, question_category=category, evaluation_key=qtype or 'unknown', value=None, display='', semantics='unknown', uses_official_answer_timestamp=False)

def official_answer_format_instruction(row: dict[str, str]) -> str:
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    parsable = str(row.get(FIELD_PARSABLE_ANSWER, '')).strip().lower()
    if qtype == 'time' or '[minutes] ago' in parsable:
        return "NaVQA evaluation adapter: answer the event using graph evidence, and keep enough timestamp/frame evidence for a separate adapter to render the official '<number> minutes ago' field relative to the question end time."
    if qtype == 'duration':
        return 'NaVQA evaluation adapter: answer with the graph-backed event interval and duration in minutes when evidence supports it.'
    if qtype == 'position' or '[position]' in parsable:
        return 'NaVQA evaluation adapter: [position] means the robot/ego pose at the semantic event time. Do not infer object 3D coordinates or inject pose data into graph memory; provide graph-backed scene/frame/time evidence for a separate ego-pose renderer.'
    if qtype == 'binary':
        return "NaVQA evaluation adapter: for binary yes/no questions, the final answer_text must be exactly 'yes' or exactly 'no', grounded in the question window. Put uncertainty or supporting evidence only in structured metadata, not in answer_text."
    if qtype == 'text':
        return 'NaVQA evaluation adapter: answer with concise natural language evidence.'
    return ''

def question_adaptation_context(row: dict[str, str]) -> str:
    qtype = str(row.get(FIELD_TYPE, '')).strip().lower()
    if qtype == 'position':
        return '\n'.join(['NaVQA position adaptation, not answer evidence:', 'The official [position] placeholder denotes the robot/ego pose at the semantic event time.', 'Treat this as event-time localization: first find the graph-backed scene, frame, or timestamp where the queried target/event is observed.', 'The question asks for a semantic target/place, while the official output field is ego pose at that target memory.', 'Do not infer object 3D coordinates, map coordinates, or x/y/z values from semantic graph evidence.', 'The graph should retrieve the target observation time/place; coordinate rendering belongs outside the graph.'])
    if qtype == 'time':
        return '\n'.join(['NaVQA time adaptation, not answer evidence:', 'Official time answers are minutes before the question end time.', 'The graph should retrieve the event time; relative-time rendering belongs outside the graph.'])
    if qtype == 'duration':
        return '\n'.join(['NaVQA duration adaptation, not answer evidence:', 'Official duration answers are minutes for an event interval.', 'The graph should retrieve the start/end event evidence when possible.'])
    if qtype == 'binary':
        return 'NaVQA binary adaptation, not answer evidence: evaluate the condition in the relevant question window, not merely anywhere in the video.'
    return ''

def official_label_to_dict(label: OfficialAnswerLabel) -> dict[str, Any]:
    return {'uuid': label.uuid, 'question_type': label.question_type, 'question_category': label.question_category, 'evaluation_key': label.evaluation_key, 'value': label.value, 'display': label.display, 'semantics': label.semantics, 'uses_official_answer_timestamp': label.uses_official_answer_timestamp, 'inject_into_graph': label.inject_into_graph}

def _answer_timestamp_values(value: Any) -> list[str]:
    return [item.strip() for item in str(value or '').split(',') if item.strip()]

def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
