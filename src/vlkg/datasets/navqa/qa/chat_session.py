from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any
from vlkg.datasets.navqa.qa.graph_qa_pipeline import answer_question
from vlkg.core.models.temporal_graph import StateEpisode
from vlkg.core.graph_retrieval.query_engine import TemporalGraphQueryEngine
from vlkg.core.graph_retrieval.query_parser import parse_question, repair_question_text

def _zh(value: str) -> str:
    return value.encode('ascii').decode('unicode_escape')
FOLLOWUP_ENTITY_PATTERNS = ('\\bit\\b', '\\bthis\\b', '\\bthat\\b', '\\bthey\\b', '\\bhe\\b', '\\bshe\\b', '\\bhim\\b', '\\bher\\b', '\\bthem\\b', _zh('\\u5b83'), _zh('\\u4ed6'), _zh('\\u5979'), _zh('\\u5b83\\u4eec'), _zh('\\u4ed6\\u4eec'), _zh('\\u5979\\u4eec'), _zh('\\u8fd9\\u4e2a'), _zh('\\u90a3\\u4e2a'), _zh('\\u8fd9\\u4e9b'), _zh('\\u90a3\\u4e9b'))
FOLLOWUP_FRAME_PATTERNS = (_zh('\\u90a3\\u4e00\\u5e27'), _zh('\\u8fd9\\u4e00\\u5e27'), _zh('\\u90a3\\u5e27'), _zh('\\u8fd9\\u5e27'), _zh('\\u5f53\\u65f6'), _zh('\\u90a3\\u65f6\\u5019'), _zh('\\u8fd9\\u65f6\\u5019'), 'at that frame', 'at this frame', 'then')
FOLLOWUP_SCENE_PATTERNS = (_zh('\\u90a3\\u4e2a\\u573a\\u666f'), _zh('\\u8fd9\\u4e2a\\u573a\\u666f'), _zh('\\u90a3\\u4e00\\u6bb5'), _zh('\\u8fd9\\u4e00\\u6bb5'), 'that scene', 'this scene')
PLACEHOLDER_FOCUS_PATTERNS = tuple({*FOLLOWUP_ENTITY_PATTERNS, *FOLLOWUP_SCENE_PATTERNS, '\\bthere\\b', '\\bhere\\b', _zh('\\u90a3\\u91cc'), _zh('\\u8fd9\\u91cc')})
GREETING_TERMS = {'hi', 'hello', 'hey', _zh('\\u4f60\\u597d'), _zh('\\u60a8\\u597d'), _zh('\\u55e8'), _zh('\\u54c8\\u55bd')}
IDENTITY_HELP_TERMS = {'help', _zh('\\u4f60\\u662f\\u8c01'), _zh('\\u4f60\\u80fd\\u505a\\u4ec0\\u4e48'), _zh('\\u4f60\\u4f1a\\u505a\\u4ec0\\u4e48'), _zh('\\u4f60\\u53ef\\u4ee5\\u505a\\u4ec0\\u4e48')}
COUNT_TERMS = ('how many', 'count', _zh('\\u591a\\u5c11'), _zh('\\u51e0'))
SCENE_TERMS = ('scene', 'scenes', 'segment', 'segments', _zh('\\u573a\\u666f'), _zh('\\u7247\\u6bb5'), _zh('\\u955c\\u5934'))
FRAME_TERMS = ('frame', 'frames', _zh('\\u5e27'), _zh('\\u753b\\u9762'))
VIDEO_TERMS = ('video', _zh('\\u89c6\\u9891'))
ENTITY_TERMS = ('entity', 'entities', _zh('\\u5b9e\\u4f53'))
PERSON_TERMS = ('person', 'people', 'man', 'woman', 'player', 'players', _zh('\\u4eba'), _zh('\\u4eba\\u7269'), _zh('\\u7403\\u5458'), _zh('\\u7537\\u4eba'), _zh('\\u5973\\u4eba'))
WEARING_TERMS = ('wearing', 'wear', _zh('\\u7a7f'), _zh('\\u8863\\u670d'), _zh('\\u670d\\u88c5'), _zh('\\u7403\\u8863'))
APPEAR_TERMS = ('appear', 'appears', 'appeared', _zh('\\u51fa\\u73b0'))
DISAPPEAR_TERMS = ('disappear', 'disappears', 'disappeared', _zh('\\u6d88\\u5931'))
VIDEO_OVERVIEW_TERMS = ('what is this video about', 'what is the video about', 'what is this about', 'summarize this video', 'video summary', _zh('\\u8fd9\\u4e2a\\u89c6\\u9891\\u662f\\u5173\\u4e8e\\u4ec0\\u4e48\\u7684'), _zh('\\u8fd9\\u4e2a\\u89c6\\u9891\\u8bb2\\u4e86\\u4ec0\\u4e48'), _zh('\\u8fd9\\u4e2a\\u89c6\\u9891\\u662f\\u4ec0\\u4e48'), _zh('\\u89c6\\u9891\\u662f\\u5173\\u4e8e\\u4ec0\\u4e48\\u7684'), _zh('\\u89c6\\u9891\\u8bb2\\u4e86\\u4ec0\\u4e48'), _zh('\\u8fd9\\u4e2a\\u89c6\\u9891\\u91cc\\u6709\\u4ec0\\u4e48'), _zh('\\u89c6\\u9891\\u91cc\\u6709\\u4ec0\\u4e48'))
PERSON_CATEGORIES = {'person', 'people', 'man', 'woman', 'boy', 'girl', 'player'}
COLOR_ALIASES = {'yellow': 'yellow', _zh('\\u9ec4'): 'yellow', _zh('\\u9ec4\\u8272'): 'yellow', 'red': 'red', _zh('\\u7ea2'): 'red', _zh('\\u7ea2\\u8272'): 'red', 'blue': 'blue', _zh('\\u84dd'): 'blue', _zh('\\u84dd\\u8272'): 'blue', 'white': 'white', _zh('\\u767d'): 'white', _zh('\\u767d\\u8272'): 'white', 'black': 'black', _zh('\\u9ed1'): 'black', _zh('\\u9ed1\\u8272'): 'black', 'green': 'green', _zh('\\u7eff'): 'green', _zh('\\u7eff\\u8272'): 'green', 'purple': 'purple', _zh('\\u7d2b'): 'purple', _zh('\\u7d2b\\u8272'): 'purple', 'orange': 'orange', _zh('\\u6a59'): 'orange', _zh('\\u6a59\\u8272'): 'orange'}

@dataclass(frozen=True)
class ChatTurn:
    question: str
    resolved_question: str
    parsed_query_type: str
    extracted_spans: dict[str, str]
    answer_text: str
    used_focus_frames: tuple[int, ...]
    used_evidence_lines: tuple[str, ...]

class GraphChatSession:

    def __init__(self, engine: TemporalGraphQueryEngine, *, answer_mode: str='graph_only', config_path: str | None=None, max_history_turns: int=8) -> None:
        self.engine = engine
        self.answer_mode = answer_mode
        self.config_path = config_path
        self.max_history_turns = max(1, max_history_turns)
        self._turns: list[ChatTurn] = []

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        return tuple(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def ask(self, question: str) -> dict[str, Any]:
        normalized_question = repair_question_text(question)
        resolved_question = self.resolve_question(normalized_question)
        builtin_payload = self._builtin_response(original_question=normalized_question, resolved_question=resolved_question)
        if builtin_payload is not None:
            self._record_turn(question=normalized_question, resolved_question=resolved_question, payload=builtin_payload)
            builtin_payload['chat_context'] = {'resolved_question': resolved_question, 'history_turn_count': len(self._turns)}
            return builtin_payload
        payload = answer_question(self.engine, resolved_question, answer_mode=self.answer_mode, config_path=self.config_path)
        self._record_turn(question=normalized_question, resolved_question=resolved_question, payload=payload)
        payload['chat_context'] = {'resolved_question': resolved_question, 'history_turn_count': len(self._turns)}
        return payload

    def resolve_question(self, question: str) -> str:
        normalized_question = repair_question_text(question)
        if not self._turns:
            return normalized_question
        parsed = parse_question(normalized_question)
        resolved_question = normalized_question
        entity_focus = self._last_entity_focus()
        scene_focus = self._last_scene_focus()
        frame_focus = self._last_focus_frame()
        has_explicit_focus = any((self._is_meaningful_focus(str(value)) for key, value in parsed.extracted_spans.items() if key in {'entity', 'subject', 'object', 'text', 'entity_id', 'scene_id'}))
        has_frame_focus = 'frame_index' in parsed.extracted_spans
        if entity_focus and (not has_explicit_focus):
            resolved_question = self._inject_entity_focus(resolved_question, entity_focus=entity_focus)
        if scene_focus and (not has_explicit_focus):
            resolved_question = self._inject_scene_focus(resolved_question, scene_focus=scene_focus)
        if frame_focus is not None and (not has_frame_focus):
            resolved_question = self._inject_frame_focus(resolved_question, frame_focus=frame_focus)
        if parsed.query_type in {'scene', 'frame'} and frame_focus is not None and (not has_frame_focus) and self._looks_like_followup(question):
            resolved_question = f'{resolved_question} at frame {frame_focus}'
        return repair_question_text(resolved_question)

    def _record_turn(self, *, question: str, resolved_question: str, payload: dict[str, Any]) -> None:
        parsed_query = payload.get('parsed_query', {}) or {}
        llm_answer = payload.get('llm_answer', {}) or {}
        structured_answer = llm_answer.get('structured_answer', {}) or {}
        used_focus_frames = tuple((int(frame_index) for frame_index in structured_answer.get('used_focus_frames', []) if isinstance(frame_index, int)))
        used_evidence_lines = tuple((str(line) for line in structured_answer.get('used_evidence_lines', []) if str(line).strip()))
        self._turns.append(ChatTurn(question=question, resolved_question=resolved_question, parsed_query_type=str(parsed_query.get('query_type', '')), extracted_spans={str(key): str(value) for key, value in dict(parsed_query.get('extracted_spans', {})).items() if str(value).strip()}, answer_text=str(llm_answer.get('answer_text', '')), used_focus_frames=used_focus_frames, used_evidence_lines=used_evidence_lines))
        if len(self._turns) > self.max_history_turns:
            self._turns = self._turns[-self.max_history_turns:]

    def _inject_entity_focus(self, question: str, *, entity_focus: str) -> str:
        updated_question = question
        replaced = False
        for pattern in FOLLOWUP_ENTITY_PATTERNS:
            new_question, count = re.subn(pattern, lambda _match: entity_focus, updated_question, count=1, flags=re.IGNORECASE)
            if count > 0:
                updated_question = new_question
                replaced = True
                break
        if replaced:
            return updated_question
        if self._looks_like_followup(question):
            return f'{entity_focus} {question}'
        return question

    def _inject_frame_focus(self, question: str, *, frame_focus: int) -> str:
        updated_question = question
        replaced = False
        frame_phrase = f"{_zh('\\u7b2c')}{frame_focus}{_zh('\\u5e27')}"
        for pattern in FOLLOWUP_FRAME_PATTERNS:
            new_question, count = re.subn(pattern, lambda _match: frame_phrase, updated_question, count=1, flags=re.IGNORECASE)
            if count > 0:
                updated_question = new_question
                replaced = True
                break
        if replaced:
            return updated_question
        return question

    def _inject_scene_focus(self, question: str, *, scene_focus: str) -> str:
        updated_question = question
        replaced = False
        for pattern in FOLLOWUP_SCENE_PATTERNS:
            new_question, count = re.subn(pattern, lambda _match: scene_focus, updated_question, count=1, flags=re.IGNORECASE)
            if count > 0:
                updated_question = new_question
                replaced = True
                break
        if replaced:
            return updated_question
        return question

    def _last_entity_focus(self) -> str | None:
        preferred_keys = ('entity', 'entity_id', 'subject', 'text', 'object')
        for turn in reversed(self._turns):
            for key in preferred_keys:
                value = turn.extracted_spans.get(key, '').strip()
                if value:
                    return value
        return None

    def _last_scene_focus(self) -> str | None:
        for turn in reversed(self._turns):
            value = turn.extracted_spans.get('scene_id', '').strip()
            if value:
                return value
        return None

    def _last_focus_frame(self) -> int | None:
        for turn in reversed(self._turns):
            if turn.used_focus_frames:
                return turn.used_focus_frames[0]
        return None

    def _looks_like_followup(self, question: str) -> bool:
        normalized = question.strip().lower()
        if len(normalized) <= 10:
            return True
        patterns = (*FOLLOWUP_ENTITY_PATTERNS, *FOLLOWUP_FRAME_PATTERNS, *FOLLOWUP_SCENE_PATTERNS)
        return any((re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns))

    def _is_meaningful_focus(self, value: str) -> bool:
        normalized = value.strip()
        if not normalized:
            return False
        if any((re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in PLACEHOLDER_FOCUS_PATTERNS)):
            return False
        return not any((re.fullmatch(pattern, normalized, flags=re.IGNORECASE) for pattern in PLACEHOLDER_FOCUS_PATTERNS))

    def _builtin_response(self, *, original_question: str, resolved_question: str) -> dict[str, Any] | None:
        if self._is_identity_or_help(original_question):
            answer_text = _zh('\\u6211\\u662f\\u672c\\u5730\\u65f6\\u5e8f\\u56fe\\u95ee\\u7b54\\u52a9\\u624b\\u3002\\u6211\\u4f1a\\u5148\\u4ece temporal graph \\u91cc\\u68c0\\u7d22\\u8bc1\\u636e\\uff0c\\u518d\\u57fa\\u4e8e\\u8fd9\\u4e9b\\u8bc1\\u636e\\u751f\\u6210\\u56de\\u7b54\\u3002\\u4f60\\u53ef\\u4ee5\\u95ee\\u6211\\u5e27\\u3001\\u573a\\u666f\\u3001\\u5b9e\\u4f53\\u3001\\u51fa\\u73b0/\\u6d88\\u5931\\u3001\\u72b6\\u6001\\u53d8\\u5316\\u3001\\u5173\\u7cfb\\u53d8\\u5316\\u548c\\u89c6\\u9891\\u6982\\u89c8\\u7c7b\\u95ee\\u9898\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='chat_capability', reasoning='Handled as a local chat capability/help query.', answer_text=answer_text)
        if self._is_greeting(original_question):
            answer_text = _zh('\\u4f60\\u597d\\uff0c\\u6211\\u4f1a\\u5148\\u68c0\\u7d22\\u65f6\\u5e8f\\u56fe\\u8bc1\\u636e\\uff0c\\u518d\\u57fa\\u4e8e\\u56fe\\u8bc1\\u636e\\u751f\\u6210\\u56de\\u7b54\\u3002\\u4f60\\u53ef\\u4ee5\\u76f4\\u63a5\\u95ee\\u6211\\u5e27\\u3001\\u573a\\u666f\\u3001\\u5b9e\\u4f53\\u3001\\u72b6\\u6001\\u53d8\\u5316\\u3001\\u5173\\u7cfb\\u53d8\\u5316\\u7b49\\u95ee\\u9898\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='chat_greeting', reasoning='Handled as a greeting rather than a graph query.', answer_text=answer_text)
        count_payload = self._count_payload(original_question=original_question, resolved_question=resolved_question)
        if count_payload is not None:
            return count_payload
        video_overview_payload = self._video_overview_payload(original_question=original_question, resolved_question=resolved_question)
        if video_overview_payload is not None:
            return video_overview_payload
        entity_temporal_payload = self._entity_temporal_payload(original_question=original_question, resolved_question=resolved_question)
        if entity_temporal_payload is not None:
            return entity_temporal_payload
        return None

    def _video_overview_payload(self, *, original_question: str, resolved_question: str) -> dict[str, Any] | None:
        normalized = repair_question_text(original_question).strip().lower()
        if not self._contains_any(normalized, VIDEO_OVERVIEW_TERMS):
            return None
        document = self.engine.document
        if document.scene_segments:
            scene = document.scene_segments[0]
            static_labels = ', '.join(scene.representative_static_labels[:4]) or 'n/a'
            dynamic_labels = ', '.join(scene.representative_dynamic_labels[:4]) or 'n/a'
            transition_counts = Counter((event.event_type for event in document.transition_events))
            transition_summary = ', '.join((f'{event_type}:{count}' for event_type, count in transition_counts.most_common(3))) or 'n/a'
            answer_text = _zh('\\u4ece\\u5f53\\u524d\\u56fe\\u8c31\\u770b\\uff0c\\u8fd9\\u4e2a\\u89c6\\u9891\\u4e3b\\u8981\\u662f\\u4e00\\u6bb5\\u56f4\\u7ed5 ') + str(scene.scene_kind_hint or static_labels) + _zh(' \\u7684\\u573a\\u666f\\u3002\\u5f53\\u524d\\u6709 ') + str(len(document.scene_segments)) + _zh(' \\u4e2a scene\\uff0c') + str(document.frame_count) + _zh(' \\u4e2a\\u91c7\\u6837\\u5e27\\u3002\\u9759\\u6001\\u951a\\u70b9\\u4e3b\\u8981\\u5305\\u62ec ') + static_labels + _zh('\\uff0c\\u52a8\\u6001\\u5173\\u6ce8\\u5bf9\\u8c61\\u4e3b\\u8981\\u5305\\u62ec ') + dynamic_labels + _zh('\\u3002\\u4e3b\\u8981\\u4e8b\\u4ef6\\u7c7b\\u578b\\u5305\\u62ec ') + transition_summary + _zh('\\u3002')
            evidence_lines = (f'video={document.video_id} scene_count={len(document.scene_segments)} frame_count={document.frame_count}', f"scene={scene.scene_id} kind={scene.scene_kind_hint or ''}", f'scene={scene.scene_id} static={static_labels}', f'scene={scene.scene_id} dynamic={dynamic_labels}', f'video={document.video_id} transitions={transition_summary}')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='video_overview', reasoning='Handled as a high-level video overview query using scene and transition summaries.', answer_text=answer_text, used_focus_frames=(scene.start_frame, scene.end_frame), used_evidence_lines=evidence_lines)
        category_counts = Counter((entity.category for entity in document.entities))
        top_categories = category_counts.most_common(6)
        if not top_categories:
            answer_text = _zh(f'\\u5f53\\u524d\\u56fe\\u8c31\\uff08{document.video_id}\\uff09\\u8fd8\\u6ca1\\u6709\\u53ef\\u7528\\u7684\\u5b9e\\u4f53\\u7edf\\u8ba1\\u4fe1\\u606f\\uff0c\\u56e0\\u6b64\\u6682\\u65f6\\u65e0\\u6cd5\\u7ed9\\u51fa\\u89c6\\u9891\\u603b\\u89c8\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='video_overview_empty', reasoning='Handled as a video overview query, but the graph contains no entities.', answer_text=answer_text)
        people_count = sum((count for category, count in category_counts.items() if category.lower() in PERSON_CATEGORIES))
        top_category_text = ', '.join((f'{category}({count})' for category, count in top_categories))
        answer_text = _zh('\\u5f53\\u524d\\u8fd9\\u4e2a\\u56fe\\u8c31\\uff08') + document.video_id + _zh('\\uff09\\u6682\\u65f6\\u6ca1\\u6709 scene \\u7ea7\\u603b\\u7ed3\\uff0c\\u6240\\u4ee5\\u6211\\u53ea\\u80fd\\u57fa\\u4e8e\\u5b9e\\u4f53\\u5206\\u5e03\\u505a\\u6982\\u89c8\\u3002\\u89c6\\u9891\\u5171\\u5305\\u542b ') + str(document.frame_count) + _zh(' \\u4e2a\\u91c7\\u6837\\u5e27\\uff0c\\u8bc6\\u522b\\u5230 ') + str(len(document.entities)) + _zh(' \\u4e2a\\u5b9e\\u4f53\\u8282\\u70b9\\u3002\\u9ad8\\u9891\\u7c7b\\u522b\\u5305\\u62ec ') + top_category_text + _zh('\\u3002')
        if people_count > 0:
            answer_text += _zh('\\u5176\\u4e2d\\u4eba\\u7269\\u7c7b\\u5b9e\\u4f53\\u7ea6\\u6709 ') + str(people_count) + _zh(' \\u4e2a\\u3002')
        else:
            answer_text += _zh('\\u5f53\\u524d\\u5e76\\u6ca1\\u6709\\u68c0\\u51fa\\u5230\\u4eba\\u7269\\u7c7b\\u5b9e\\u4f53\\u3002')
        evidence_lines = tuple((f'category={category} count={count}' for category, count in top_categories))
        return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='video_overview', reasoning='Handled as a high-level video overview query using entity category statistics.', answer_text=answer_text, used_evidence_lines=evidence_lines)

    def _entity_temporal_payload(self, *, original_question: str, resolved_question: str) -> dict[str, Any] | None:
        normalized = repair_question_text(resolved_question).strip().lower()
        is_appear_query = self._contains_any(normalized, APPEAR_TERMS)
        is_disappear_query = self._contains_any(normalized, DISAPPEAR_TERMS)
        if not is_appear_query and (not is_disappear_query):
            return None
        parsed = parse_question(resolved_question)
        if parsed.query_type != 'entity':
            return None
        entity_id = str(parsed.filters.get('entity_id') or '').strip() or None
        text = str(parsed.filters.get('text') or '').strip() or None
        label = str(parsed.filters.get('label') or '').strip() or None
        if not any((entity_id, text, label)):
            return None
        resolution = self.engine.resolve_entities(entity_id=entity_id, label=label, text=text, limit=5)
        if not resolution.entity_ids:
            answer_text = _zh('\\u56fe\\u8c31\\u4e2d\\u6ca1\\u6709\\u627e\\u5230\\u4e0e\\u8fd9\\u4e2a\\u5b9e\\u4f53\\u76f8\\u5339\\u914d\\u7684\\u8bc1\\u636e\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='entity_temporal_no_match', reasoning='Handled as an entity temporal query, but no entity matched.', answer_text=answer_text, extracted_spans={'entity': text or label or entity_id or ''})
        if len(resolution.entity_ids) > 1:
            preview_lines = []
            for current_entity_id in resolution.entity_ids[:5]:
                entity = self.engine.entities_by_id[current_entity_id]
                preview_lines.append(f'{entity.entity_id} category={entity.category} frames {entity.first_seen_frame}->{entity.last_seen_frame}')
            answer_text = _zh('\\u56fe\\u8c31\\u4e2d\\u6709\\u591a\\u4e2a\\u5b9e\\u4f53\\u90fd\\u53ef\\u80fd\\u5339\\u914d\\u8fd9\\u4e2a\\u63cf\\u8ff0\\uff0c\\u76ee\\u524d\\u65e0\\u6cd5\\u552f\\u4e00\\u5b9a\\u4f4d\\u3002\\u4f60\\u53ef\\u4ee5\\u518d\\u5177\\u4f53\\u4e00\\u70b9\\uff0c\\u6bd4\\u5982\\u52a0\\u4e0a\\u989c\\u8272\\u3001\\u4f4d\\u7f6e\\u6216\\u5176\\u4ed6\\u5173\\u7cfb\\u63cf\\u8ff0\\u3002')
            focus_frames = tuple(sorted({frame for current_entity_id in resolution.entity_ids[:5] for frame in (self.engine.entities_by_id[current_entity_id].first_seen_frame, self.engine.entities_by_id[current_entity_id].last_seen_frame)})[:8])
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='entity_temporal_ambiguous', reasoning='Handled as an entity temporal query, but multiple entities matched.', answer_text=answer_text, extracted_spans={'entity': text or label or entity_id or ''}, used_focus_frames=focus_frames, used_evidence_lines=tuple(preview_lines))
        matched_entity_id = resolution.entity_ids[0]
        entity = self.engine.entities_by_id[matched_entity_id]
        if is_disappear_query:
            answer_text = _zh(f'\\u5b9e\\u4f53 {entity.entity_id} \\u5728\\u56fe\\u8c31\\u4e2d\\u6700\\u540e\\u4e00\\u6b21\\u51fa\\u73b0\\u4e8e\\u7b2c {entity.last_seen_frame} \\u5e27\\uff0c\\u65f6\\u95f4\\u6233\\u7ea6\\u4e3a {entity.last_seen_ts:.3f} \\u79d2\\u3002')
            query_type = 'entity_disappear'
            focus_frames = (entity.last_seen_frame,)
        elif is_appear_query:
            answer_text = _zh(f'\\u5b9e\\u4f53 {entity.entity_id} \\u5728\\u56fe\\u8c31\\u4e2d\\u9996\\u6b21\\u51fa\\u73b0\\u4e8e\\u7b2c {entity.first_seen_frame} \\u5e27\\uff0c\\u65f6\\u95f4\\u6233\\u7ea6\\u4e3a {entity.first_seen_ts:.3f} \\u79d2\\u3002')
            query_type = 'entity_appear'
            focus_frames = (entity.first_seen_frame,)
        else:
            return None
        evidence_lines = (f'{entity.entity_id} category={entity.category} frames {entity.first_seen_frame}->{entity.last_seen_frame}',)
        return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type=query_type, reasoning='Handled as an entity appearance/disappearance query.', answer_text=answer_text, extracted_spans={'entity': text or label or entity_id or entity.entity_id}, used_focus_frames=focus_frames, used_evidence_lines=evidence_lines)

    def _count_payload(self, *, original_question: str, resolved_question: str) -> dict[str, Any] | None:
        normalized = repair_question_text(original_question).strip().lower()
        if not self._contains_any(normalized, COUNT_TERMS):
            return None
        if self._contains_any(normalized, SCENE_TERMS):
            count = len(self.engine.document.scene_segments)
            answer_text = _zh(f'\\u5f53\\u524d\\u56fe\\u8c31\\u4e2d\\u5171\\u6709 {count} \\u4e2a\\u573a\\u666f\\u6bb5\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_scene', reasoning='Handled as a scene count query.', answer_text=answer_text, used_focus_frames=self._scene_focus_frames())
        if self._contains_any(normalized, FRAME_TERMS) or (self._contains_any(normalized, VIDEO_TERMS) and (not self._contains_any(normalized, SCENE_TERMS + PERSON_TERMS + ENTITY_TERMS))):
            count = int(self.engine.document.frame_count)
            answer_text = _zh(f'\\u5f53\\u524d\\u56fe\\u8c31\\u4e2d\\u5171\\u6709 {count} \\u4e2a\\u91c7\\u6837\\u5e27\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_frame', reasoning='Handled as a frame count query.', answer_text=answer_text)
        if self._contains_any(normalized, PERSON_TERMS) and self._contains_any(normalized, WEARING_TERMS):
            return self._count_people_by_state(original_question=original_question, resolved_question=resolved_question)
        if self._contains_any(normalized, PERSON_TERMS):
            return self._count_people_entities(original_question=original_question, resolved_question=resolved_question)
        if self._contains_any(normalized, ENTITY_TERMS):
            count = len(self.engine.document.entities)
            answer_text = _zh(f'\\u5f53\\u524d\\u56fe\\u8c31\\u4e2d\\u5171\\u6709 {count} \\u4e2a\\u5b9e\\u4f53\\u8282\\u70b9\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_entity', reasoning='Handled as an entity count query.', answer_text=answer_text)
        answer_text = _zh('\\u6211\\u8bc6\\u522b\\u5230\\u4f60\\u5728\\u95ee\\u201c\\u591a\\u5c11/\\u8ba1\\u6570\\u201d\\u7c7b\\u95ee\\u9898\\uff0c\\u4f46\\u76ee\\u6807\\u8fd8\\u4e0d\\u591f\\u660e\\u786e\\u3002\\u4f60\\u53ef\\u4ee5\\u76f4\\u63a5\\u95ee\\uff1a\\u201c\\u8fd9\\u4e2a\\u89c6\\u9891\\u6709\\u591a\\u5c11\\u5e27\\u201d\\uff0c\\u201c\\u6709\\u591a\\u5c11\\u4e2a\\u573a\\u666f\\u201d\\uff0c\\u6216\\u201c\\u6709\\u591a\\u5c11\\u4e2a\\u4eba\\u201d\\u3002')
        return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_unsupported', reasoning='Detected a count intent, but the count target is ambiguous.', answer_text=answer_text)

    def _count_people_entities(self, *, original_question: str, resolved_question: str) -> dict[str, Any]:
        matched_entities = [entity for entity in self.engine.document.entities if entity.category.lower() in PERSON_CATEGORIES]
        count = len(matched_entities)
        if count == 0:
            preview_categories = ', '.join(sorted({entity.category for entity in self.engine.document.entities[:8]}))
            answer_text = _zh(f'\\u5f53\\u524d\\u52a0\\u8f7d\\u7684\\u56fe\\u8c31\\uff08{self.engine.document.video_id}\\uff09\\u4e2d\\u6ca1\\u6709\\u4eba\\u7269\\u7c7b\\u5b9e\\u4f53\\u8bc1\\u636e\\uff0c\\u56e0\\u6b64\\u7ed3\\u679c\\u4e3a 0\\u3002\\u5f53\\u524d\\u4e3b\\u8981\\u5b9e\\u4f53\\u7c7b\\u578b\\u5305\\u62ec\\uff1a{preview_categories}\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_person_entity', reasoning='Handled as a person entity count query, but the current graph contains no person-like entities.', answer_text=answer_text)
        answer_text = _zh(f'\\u5f53\\u524d\\u56fe\\u8c31\\u4e2d\\u5171\\u8bc6\\u522b\\u5230 {count} \\u4e2a\\u4eba\\u7269\\u5b9e\\u4f53\\u3002')
        evidence_lines = [f'{entity.entity_id} category={entity.category} frames {entity.first_seen_frame}->{entity.last_seen_frame}' for entity in matched_entities[:6]]
        focus_frames = tuple(sorted({frame for entity in matched_entities[:6] for frame in (entity.first_seen_frame, entity.last_seen_frame)})[:8])
        return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_person_entity', reasoning='Handled as a person entity count query.', answer_text=answer_text, used_focus_frames=focus_frames, used_evidence_lines=tuple(evidence_lines))

    def _count_people_by_state(self, *, original_question: str, resolved_question: str) -> dict[str, Any]:
        normalized = repair_question_text(original_question).strip().lower()
        color_token = self._extract_color_token(normalized)
        if color_token is None:
            answer_text = _zh('\\u6211\\u8bc6\\u522b\\u5230\\u4f60\\u5728\\u95ee\\u201c\\u67d0\\u79cd\\u7a7f\\u7740\\u7684\\u4eba\\u6709\\u591a\\u5c11\\u201d\\uff0c\\u4f46\\u989c\\u8272\\u6216\\u8863\\u7740\\u72b6\\u6001\\u8fd8\\u4e0d\\u591f\\u5177\\u4f53\\u3002\\u4f60\\u53ef\\u4ee5\\u7ee7\\u7eed\\u95ee\\uff1a\\u201c\\u6709\\u591a\\u5c11\\u4e2a\\u7a7f\\u9ec4\\u8272\\u8863\\u670d\\u7684\\u4eba\\u201d\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_state_unsupported', reasoning='Detected a clothing count query without a supported color token.', answer_text=answer_text)
        matched_episodes: list[StateEpisode] = []
        matched_entity_ids: set[str] = set()
        focus_frames: list[int] = []
        for episode in self.engine.document.state_episodes:
            entity = self.engine.entities_by_id.get(episode.entity_id)
            if entity is None or entity.category.lower() not in PERSON_CATEGORIES:
                continue
            state_text = episode.attr_value.lower()
            if color_token not in state_text:
                continue
            matched_episodes.append(episode)
            matched_entity_ids.add(episode.entity_id)
            focus_frames.extend([episode.start_frame, episode.end_frame])
        if not matched_entity_ids:
            person_count = sum((1 for entity in self.engine.document.entities if entity.category.lower() in PERSON_CATEGORIES))
            if person_count == 0:
                answer_text = _zh(f'\\u5f53\\u524d\\u52a0\\u8f7d\\u7684\\u56fe\\u8c31\\uff08{self.engine.document.video_id}\\uff09\\u4e2d\\u6ca1\\u6709\\u4eba\\u7269\\u7c7b\\u5b9e\\u4f53\\uff0c\\u56e0\\u6b64\\u4e5f\\u65e0\\u6cd5\\u5728\\u56fe\\u8c31\\u4e2d\\u7edf\\u8ba1\\u201c{color_token}\\u8863\\u670d\\u7684\\u4eba\\u201d\\u3002')
            else:
                answer_text = _zh(f'\\u56fe\\u8c31\\u4e2d\\u6ca1\\u6709\\u627e\\u5230\\u201c{color_token}\\u201d\\u76f8\\u5173\\u7684\\u4eba\\u7269\\u7a7f\\u7740\\u72b6\\u6001\\u8bc1\\u636e\\u3002\\u8fd9\\u901a\\u5e38\\u8868\\u793a\\u4e0a\\u6e38\\u5c5e\\u6027\\u62bd\\u53d6\\u6ca1\\u6709\\u7a33\\u5b9a\\u62ff\\u5230\\u989c\\u8272\\u4fe1\\u606f\\uff0c\\u4e0d\\u7b49\\u4e8e\\u89c6\\u9891\\u91cc\\u4e00\\u5b9a\\u6ca1\\u6709\\u8fd9\\u79cd\\u989c\\u8272\\u8863\\u670d\\u3002')
            return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_people_by_state', reasoning='Handled as a clothing count query, but the graph has no matching evidence.', answer_text=answer_text)
        count = len(matched_entity_ids)
        answer_text = _zh(f'\\u56fe\\u8c31\\u4e2d\\u53ef\\u4ee5\\u786e\\u8ba4\\u6709 {count} \\u4e2a\\u4eba\\u7269\\u5b9e\\u4f53\\u51fa\\u73b0\\u4e86\\u201c{color_token}\\u201d\\u76f8\\u5173\\u7684\\u7a7f\\u7740\\u72b6\\u6001\\u3002')
        evidence_lines = [f"{episode.entity_id} state='{episode.attr_value}' frames {episode.start_frame}->{episode.end_frame}" for episode in matched_episodes[:6]]
        return self._simple_payload(question=original_question, resolved_question=resolved_question, query_type='count_people_by_state', reasoning='Handled as a person clothing count query.', answer_text=answer_text, used_focus_frames=tuple(sorted(set(focus_frames))[:8]), used_evidence_lines=tuple(evidence_lines))

    def _extract_color_token(self, normalized_question: str) -> str | None:
        for source, target in COLOR_ALIASES.items():
            if source.lower() in normalized_question:
                return target
        return None

    def _scene_focus_frames(self) -> tuple[int, ...]:
        frames: list[int] = []
        for scene in self.engine.document.scene_segments[:4]:
            frames.extend([scene.start_frame, scene.end_frame])
        return tuple(sorted(set(frames))[:8])

    def _is_greeting(self, question: str) -> bool:
        normalized = repair_question_text(question).strip().lower()
        return normalized in GREETING_TERMS

    def _is_identity_or_help(self, question: str) -> bool:
        normalized = repair_question_text(question).strip().lower()
        return normalized in IDENTITY_HELP_TERMS

    def _contains_any(self, question: str, terms: tuple[str, ...]) -> bool:
        normalized = repair_question_text(question).strip().lower()
        return any((term.lower() in normalized for term in terms))

    def _simple_payload(self, *, question: str, resolved_question: str, query_type: str, reasoning: str, answer_text: str, extracted_spans: dict[str, str] | None=None, used_focus_frames: tuple[int, ...]=(), used_evidence_lines: tuple[str, ...]=()) -> dict[str, Any]:
        return {'question': resolved_question, 'parsed_query': {'query_type': query_type, 'filters': {}, 'confidence': 0.99, 'reasoning': reasoning, 'extracted_spans': extracted_spans or {}}, 'graph_answer': {'query_type': query_type, 'matched_by': 'chat_builtin', 'filters': {}, 'summary': answer_text, 'matches': [], 'retrieval_bundle': {'video_id': self.engine.document.video_id, 'query_type': query_type, 'match_count': 0, 'focus_frames': [{'frame_index': frame_index} for frame_index in used_focus_frames], 'scene_summaries': [], 'evidence_digest': []}, 'llm_ready_bundle': {'video_id': self.engine.document.video_id, 'query_type': query_type, 'answer_brief': answer_text, 'evidence_lines': list(used_evidence_lines), 'focus_frame_lines': [f'frame={frame_index}' for frame_index in used_focus_frames], 'scene_lines': [], 'match_highlights': [], 'uncertainty_notes': [], 'llm_context_text': answer_text, 'structured_graph_context': {'video_id': self.engine.document.video_id, 'query_type': query_type, 'answer_brief': answer_text, 'evidence_lines': list(used_evidence_lines), 'focus_frames': list(used_focus_frames)}}, 'answer_brief': answer_text}, 'llm_answer': {'mode': 'chat_builtin', 'graph_rag_status': 'chat_builtin', 'provider': 'local_chat_session', 'model': None, 'question': question, 'answer_text': answer_text, 'graph_rag_metadata': {'contract_version': 'graph_rag_answer_v1', 'query_type': query_type, 'match_count': 0, 'focus_frame_count': len(used_focus_frames), 'scene_count': 0, 'evidence_line_count': len(used_evidence_lines)}, 'structured_answer': {'answer_text': answer_text, 'used_focus_frames': list(used_focus_frames), 'used_evidence_lines': list(used_evidence_lines), 'uncertainty': []}}, 'chat_context': {'resolved_question': resolved_question, 'history_turn_count': len(self._turns)}}
