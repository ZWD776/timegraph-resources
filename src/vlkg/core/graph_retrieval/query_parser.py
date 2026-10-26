from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedQuery:
    query_type: str
    filters: dict[str, Any]
    confidence: float
    reasoning: str
    extracted_spans: dict[str, str] = field(default_factory=dict)


ENTITY_ID_PATTERN = re.compile(r"[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+:\d{6}")
TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9_\-]+:[A-Za-z0-9_\-]+:\d{6}|[A-Za-z0-9_\-]+|[\u4e00-\u9fff]+"
)
QUOTED_TEXT_PATTERN = re.compile(
    r'"([^"]+)"|\'([^\']+)\'|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019|'
    r'\u300c([^\u300d]+)\u300d|\u300e([^\u300f]+)\u300f'
)
FRAME_INDEX_PATTERN = re.compile(
    r"(?:frame|frames)\s*(\d+)|第?\s*(\d+)\s*帧",
    re.IGNORECASE,
)


def _zh(value: str) -> str:
    return value.encode("ascii").decode("unicode_escape")


RELATION_PREDICATE_ALIASES: dict[str, tuple[str, ...]] = {
    "HOLDING": (
        "holding",
        "holds",
        "hold",
        _zh("\\u62ff\\u7740"),
        _zh("\\u62ff\\u4f4f"),
        _zh("\\u62b1\\u7740"),
        _zh("\\u6293\\u7740"),
    ),
    "NEAR": (
        "near",
        _zh("\\u9644\\u8fd1"),
        _zh("\\u9760\\u8fd1"),
        _zh("\\u65c1\\u8fb9"),
    ),
    "NEXT_TO": (
        "next to",
        "beside",
        _zh("\\u6328\\u7740"),
        _zh("\\u7d27\\u90bb"),
    ),
    "GUARDING": (
        "guarding",
        _zh("\\u76ef\\u9632"),
        _zh("\\u9632\\u5b88"),
    ),
    "PASSING_TO": (
        "passing to",
        "pass to",
        _zh("\\u4f20\\u7ed9"),
        _zh("\\u4f20\\u5411"),
    ),
    "LOOKING_AT": (
        "looking at",
        "look at",
        _zh("\\u770b\\u7740"),
        _zh("\\u770b\\u5411"),
    ),
    "MOVING_TOWARD": (
        "moving toward",
        "move toward",
        _zh("\\u671d\\u5411"),
        _zh("\\u79fb\\u52a8\\u5230"),
    ),
    "WATCHING": (
        "watching",
        "watch",
        _zh("\\u6ce8\\u89c6"),
    ),
    "RECEIVING_FROM": (
        "receiving from",
        "receive from",
        _zh("\\u63a5\\u5230"),
        _zh("\\u63a5\\u81ea\\u6765\\u81ea"),
    ),
    "BEING_CHASED_BY": (
        "being chased by",
        "chased by",
        _zh("\\u88ab\\u8ffd\\u9010"),
        _zh("\\u88ab\\u8ffd\\u7740"),
    ),
    "BEING_PUSHED_BY": (
        "being pushed by",
        "pushed by",
        _zh("\\u88ab\\u63a8\\u7740"),
        _zh("\\u88ab\\u63a8\\u52a8"),
    ),
    "SHOOTING": (
        "shooting",
        "shoot",
        _zh("\\u6295\\u7bee"),
        _zh("\\u5c04\\u95e8"),
    ),
}

STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "wearing": (
        "wearing",
        _zh("\\u7a7f\\u7740"),
        _zh("\\u8eab\\u7a7f"),
    ),
    "holding": (
        "holding",
        _zh("\\u62ff\\u7740"),
        _zh("\\u62ff\\u4f4f"),
        _zh("\\u62b1\\u7740"),
    ),
    "dribbling": (
        "dribbling",
        _zh("\\u8fd0\\u7403"),
    ),
    "shooting": (
        "shooting",
        _zh("\\u6295\\u7bee"),
        _zh("\\u5c04\\u95e8"),
    ),
    "guarding": (
        "guarding",
        _zh("\\u9632\\u5b88"),
        _zh("\\u76ef\\u9632"),
    ),
    "looking": (
        "looking",
        _zh("\\u770b\\u7740"),
        _zh("\\u6ce8\\u89c6"),
    ),
    "running": (
        "running",
        _zh("\\u5954\\u8dd1"),
        _zh("\\u8dd1\\u52a8"),
    ),
    "standing": (
        "standing",
        _zh("\\u7ad9\\u7acb"),
        _zh("\\u7ad9\\u7740"),
    ),
}

RELATION_HINT_TERMS = (
    "relation",
    "relationship",
    "predicate",
    _zh("\\u5173\\u7cfb"),
    _zh("\\u5173\\u8054"),
)
STATE_HINT_TERMS = (
    "state",
    "status",
    _zh("\\u72b6\\u6001"),
    _zh("\\u5c5e\\u6027"),
)
ENTITY_HINT_TERMS = (
    "entity",
    "entities",
    "appear",
    "appears",
    "appeared",
    "disappear",
    "disappears",
    _zh("\\u51fa\\u73b0"),
    _zh("\\u6d88\\u5931"),
    _zh("\\u5b9e\\u4f53"),
)
TEMPORAL_HINT_TERMS = (
    "when",
    "what time",
    _zh("\\u4f55\\u65f6"),
    _zh("\\u4ec0\\u4e48\\u65f6\\u5019"),
    _zh("\\u5728\\u54ea\\u4e00\\u5e27"),
    _zh("\\u54ea\\u4e00\\u5e27"),
)
SCENE_HINT_TERMS = (
    "scene",
    "scenes",
    "segment",
    "segments",
    _zh("\\u573a\\u666f"),
    _zh("\\u7247\\u6bb5"),
)
FRAME_HINT_TERMS = (
    "frame",
    "frames",
    "snapshot",
    "moment",
    "active",
    "happen",
    "happens",
    "happened",
    "occur",
    "occurs",
    "occurred",
    "what happens",
    "what happened",
    _zh("\\u5e27"),
    _zh("\\u8fd9\\u4e00\\u5e27"),
    _zh("\\u5f53\\u524d\\u5e27"),
    _zh("\\u53d1\\u751f\\u4e86\\u4ec0\\u4e48"),
    _zh("\\u6709\\u4ec0\\u4e48"),
)
RELATION_CONNECTORS = (
    "and",
    "with",
    "to",
    "toward",
    "versus",
    "vs",
    _zh("\\u548c"),
    _zh("\\u4e0e"),
    _zh("\\u8ddf"),
    _zh("\\u540c"),
    _zh("\\u53ca"),
    _zh("\\u5bf9"),
)
GENERIC_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "this",
    "that",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "time",
    "frame",
    "frames",
    "timeline",
    "history",
    "show",
    "tell",
    "me",
    "happen",
    "happens",
    "happened",
    "contain",
    "contains",
    "occur",
    "occurs",
    "occurred",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "into",
    "during",
    "please",
    "entity",
    "entities",
    "state",
    "status",
    "relation",
    "relationship",
    "predicate",
    _zh("\\u51fa\\u73b0"),
    _zh("\\u6d88\\u5931"),
    _zh("\\u4ec0\\u4e48"),
    _zh("\\u4f55\\u65f6"),
    _zh("\\u4ec0\\u4e48\\u65f6\\u5019"),
    _zh("\\u54ea\\u4e00\\u5e27"),
    _zh("\\u5b9e\\u4f53"),
    _zh("\\u72b6\\u6001"),
    _zh("\\u5173\\u7cfb"),
    _zh("\\u5173\\u8054"),
    _zh("\\u662f\\u5426"),
    _zh("\\u54ea\\u4e2a"),
    _zh("\\u54ea\\u4e9b"),
    _zh("\\u7b2c"),
    _zh("\\u8fd9"),
    _zh("\\u5f53\\u524d"),
    _zh("\\u8be5"),
    _zh("\\u672c"),
}


def parse_question(question: str) -> ParsedQuery:
    question = repair_question_text(question)
    normalized = _normalize_text(question)
    quoted_spans = _extract_quoted_spans(question)
    entity_ids = ENTITY_ID_PATTERN.findall(question)
    candidate_entities = _candidate_entities(question)

    relation_predicate = _match_relation_predicate(normalized)
    state_text = _match_state_text(normalized)
    relation_hint = bool(relation_predicate) or _contains_any(normalized, RELATION_HINT_TERMS)
    state_hint = bool(state_text) or _contains_any(normalized, STATE_HINT_TERMS)
    entity_hint = _contains_any(normalized, ENTITY_HINT_TERMS)
    temporal_hint = _contains_any(normalized, TEMPORAL_HINT_TERMS)
    scene_hint = _contains_any(normalized, SCENE_HINT_TERMS)
    frame_hint = _contains_any(normalized, FRAME_HINT_TERMS)
    frame_index = _extract_frame_index(question)

    relation_pair = _resolve_relation_pair(
        quoted_spans=quoted_spans,
        entity_ids=entity_ids,
        candidate_entities=candidate_entities,
        normalized_question=normalized,
        relation_predicate=relation_predicate,
        relation_hint=relation_hint,
    )
    if relation_pair is not None:
        subject_value, object_value = relation_pair
        subject_key = (
            "subject_entity_id"
            if ENTITY_ID_PATTERN.fullmatch(subject_value)
            else "subject_text"
        )
        object_key = (
            "object_entity_id"
            if ENTITY_ID_PATTERN.fullmatch(object_value)
            else "object_text"
        )
        return ParsedQuery(
            query_type="relation",
            filters={
                subject_key: subject_value,
                object_key: object_value,
                "predicate": relation_predicate,
                "limit": 20,
            },
            confidence=0.93 if relation_predicate else 0.82,
            reasoning="Detected a relation query with subject, object, and predicate.",
            extracted_spans={
                "subject": subject_value,
                "object": object_value,
                "predicate": relation_predicate or "",
            },
        )

    state_entity = _resolve_state_entity(
        quoted_spans=quoted_spans,
        entity_ids=entity_ids,
        candidate_entities=candidate_entities,
    )
    explicit_state = _resolve_state_value(
        quoted_spans=quoted_spans,
        state_text=state_text,
    )
    if state_hint or (explicit_state and state_entity):
        filters: dict[str, Any] = {"limit": 20}
        if state_entity:
            if ENTITY_ID_PATTERN.fullmatch(state_entity):
                filters["entity_id"] = state_entity
            else:
                filters["text"] = state_entity
        if explicit_state:
            filters["state_text"] = explicit_state
        return ParsedQuery(
            query_type="state",
            filters=filters,
            confidence=0.90 if state_entity and explicit_state else 0.76,
            reasoning="Detected a state query with entity and state cues.",
            extracted_spans={
                "entity": state_entity or "",
                "state_text": explicit_state or "",
            },
        )

    if scene_hint:
        scene_value = _resolve_scene_focus(
            quoted_spans=quoted_spans,
            entity_ids=entity_ids,
            candidate_entities=candidate_entities,
        )
        if not _is_specific_entity_focus(
            scene_value,
            disallow_terms=(*SCENE_HINT_TERMS, *FRAME_HINT_TERMS),
        ):
            scene_value = ""
        filters: dict[str, Any] = {"limit": 20}
        extracted_spans: dict[str, str] = {}
        if frame_index is not None and not (
            scene_value
            and (
                ":scene:" in scene_value
                or ENTITY_ID_PATTERN.fullmatch(scene_value)
            )
        ):
            filters["frame_index"] = frame_index
            extracted_spans["frame_index"] = str(frame_index)
        elif scene_value:
            if ":scene:" in scene_value:
                filters["scene_id"] = scene_value
                extracted_spans["scene_id"] = scene_value
            elif ENTITY_ID_PATTERN.fullmatch(scene_value):
                filters["entity_id"] = scene_value
                extracted_spans["entity_id"] = scene_value
            else:
                filters["text"] = scene_value
                extracted_spans["text"] = scene_value
        return ParsedQuery(
            query_type="scene",
            filters=filters,
            confidence=0.88 if scene_value or frame_index is not None else 0.72,
            reasoning="Detected a scene container query.",
            extracted_spans=extracted_spans,
        )

    if frame_index is not None and (frame_hint or temporal_hint or not entity_hint):
        entity_value = _resolve_entity_focus(
            quoted_spans=quoted_spans,
            entity_ids=entity_ids,
            candidate_entities=candidate_entities,
        )
        if not _is_specific_entity_focus(entity_value, disallow_terms=FRAME_HINT_TERMS):
            entity_value = ""
        filters = {
            "frame_index": frame_index,
            "limit": 20,
        }
        extracted_spans = {
            "frame_index": str(frame_index),
        }
        if entity_value:
            if ENTITY_ID_PATTERN.fullmatch(entity_value):
                filters["entity_id"] = entity_value
                extracted_spans["entity_id"] = entity_value
            else:
                filters["text"] = entity_value
                extracted_spans["text"] = entity_value
        return ParsedQuery(
            query_type="frame",
            filters=filters,
            confidence=0.86 if entity_value else 0.81,
            reasoning="Detected a frame-level snapshot query.",
            extracted_spans=extracted_spans,
        )

    entity_value = _resolve_entity_focus(
        quoted_spans=quoted_spans,
        entity_ids=entity_ids,
        candidate_entities=candidate_entities,
    )
    filters: dict[str, Any] = {"limit": 20}
    if entity_value:
        if ENTITY_ID_PATTERN.fullmatch(entity_value):
            filters["entity_id"] = entity_value
        else:
            filters["text"] = entity_value
    confidence = 0.80 if entity_value and (entity_hint or temporal_hint) else 0.62
    return ParsedQuery(
        query_type="entity",
        filters=filters,
        confidence=confidence,
        reasoning="Fell back to an entity timeline query.",
        extracted_spans={"entity": entity_value or ""},
    )


def repair_question_text(question: str | None) -> str:
    if question is None:
        return ""

    original = str(question).strip()
    if not original:
        return ""

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: str) -> None:
        stripped = value.strip()
        if stripped and stripped not in seen:
            candidates.append(stripped)
            seen.add(stripped)

    add_candidate(original)
    for source_encoding in ("latin-1", "cp1252", "gbk"):
        try:
            add_candidate(original.encode(source_encoding).decode("utf-8"))
        except UnicodeError:
            continue

    original_score = _question_text_score(original)
    best_candidate = original
    best_score = original_score
    for candidate in candidates[1:]:
        candidate_score = _question_text_score(candidate)
        if candidate_score > best_score:
            best_candidate = candidate
            best_score = candidate_score
    return best_candidate


def _match_relation_predicate(normalized_question: str) -> str | None:
    for predicate, aliases in RELATION_PREDICATE_ALIASES.items():
        if _contains_any(normalized_question, aliases):
            return predicate
    uppercase_match = re.search(r"\b[A-Z_]{3,}\b", normalized_question)
    if uppercase_match is not None:
        return uppercase_match.group(0)
    return None


def _match_state_text(normalized_question: str) -> str | None:
    for state_text, aliases in STATE_ALIASES.items():
        if _contains_any(normalized_question, aliases):
            return state_text
    return None


def _extract_quoted_spans(question: str) -> list[str]:
    spans: list[str] = []
    for match in QUOTED_TEXT_PATTERN.finditer(question):
        value = next((item for item in match.groups() if item), "")
        value = value.strip()
        if value:
            spans.append(value)
    return spans


def _extract_frame_index(question: str) -> int | None:
    match = FRAME_INDEX_PATTERN.search(question)
    if match is None:
        return None
    for group in match.groups():
        if group is not None:
            return int(group)
    return None


def _candidate_entities(question: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_PATTERN.findall(question):
        normalized_token = token.strip().lower()
        if not normalized_token:
            continue
        if ENTITY_ID_PATTERN.fullmatch(token):
            if token not in seen:
                candidates.append(token)
                seen.add(token)
            continue
        if not _is_entity_like_token(normalized_token):
            continue
        if normalized_token not in seen:
            candidates.append(normalized_token)
            seen.add(normalized_token)
    return candidates


def _is_entity_like_token(token: str) -> bool:
    if token.isdigit():
        return False
    if len(token) == 1 and not re.fullmatch(r"[\u4e00-\u9fff]", token):
        return False
    if token in GENERIC_STOPWORDS:
        return False
    if token in _all_keyword_terms():
        return False
    return True


def _all_keyword_terms() -> set[str]:
    keywords = set()
    for alias_group in RELATION_PREDICATE_ALIASES.values():
        keywords.update(_split_alias_terms(alias_group))
    for alias_group in STATE_ALIASES.values():
        keywords.update(_split_alias_terms(alias_group))
    keywords.update(term.lower() for term in RELATION_HINT_TERMS)
    keywords.update(term.lower() for term in STATE_HINT_TERMS)
    keywords.update(term.lower() for term in ENTITY_HINT_TERMS)
    keywords.update(term.lower() for term in TEMPORAL_HINT_TERMS)
    keywords.update(term.lower() for term in SCENE_HINT_TERMS)
    keywords.update(term.lower() for term in FRAME_HINT_TERMS)
    keywords.update(term.lower() for term in RELATION_CONNECTORS)
    return keywords


def _split_alias_terms(aliases: tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for alias in aliases:
        normalized = _normalize_text(alias)
        terms.update(part for part in normalized.split(" ") if part)
    return terms


def _resolve_relation_pair(
    *,
    quoted_spans: list[str],
    entity_ids: list[str],
    candidate_entities: list[str],
    normalized_question: str,
    relation_predicate: str | None,
    relation_hint: bool,
) -> tuple[str, str] | None:
    if len(entity_ids) >= 2:
        return entity_ids[0], entity_ids[1]
    if len(quoted_spans) >= 2:
        return quoted_spans[0], quoted_spans[1]
    if not (relation_hint or relation_predicate):
        return None

    ordered_candidates = [
        value for value in candidate_entities if not ENTITY_ID_PATTERN.fullmatch(value)
    ]
    if len(ordered_candidates) < 2:
        return None

    connector_pair = _relation_pair_by_connector(
        normalized_question=normalized_question,
        candidate_entities=ordered_candidates,
    )
    if connector_pair is not None:
        return connector_pair

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for value in ordered_candidates:
        if value in seen:
            continue
        unique_candidates.append(value)
        seen.add(value)
        if len(unique_candidates) == 2:
            break
    if len(unique_candidates) >= 2:
        return unique_candidates[0], unique_candidates[1]
    return None


def _relation_pair_by_connector(
    *,
    normalized_question: str,
    candidate_entities: list[str],
) -> tuple[str, str] | None:
    spans: list[tuple[int, str]] = []
    for value in candidate_entities:
        match = re.search(
            rf"(?<![a-z0-9_\-]){re.escape(value)}(?![a-z0-9_\-])",
            normalized_question,
        )
        if match is not None:
            spans.append((match.start(), value))
    spans.sort(key=lambda item: item[0])
    if len(spans) < 2:
        return None

    for index in range(len(spans) - 1):
        left_end = spans[index][0] + len(spans[index][1])
        right_start = spans[index + 1][0]
        gap = normalized_question[left_end:right_start].strip()
        if not gap:
            continue
        gap_terms = set(gap.split(" "))
        if any(connector in gap_terms or connector in gap for connector in RELATION_CONNECTORS):
            return spans[index][1], spans[index + 1][1]
    return None


def _resolve_state_entity(
    *,
    quoted_spans: list[str],
    entity_ids: list[str],
    candidate_entities: list[str],
) -> str:
    if entity_ids:
        return entity_ids[0]
    if quoted_spans:
        return quoted_spans[0]
    return candidate_entities[0] if candidate_entities else ""


def _resolve_state_value(
    *,
    quoted_spans: list[str],
    state_text: str | None,
) -> str:
    if len(quoted_spans) >= 2:
        return quoted_spans[1]
    return state_text or ""


def _resolve_entity_focus(
    *,
    quoted_spans: list[str],
    entity_ids: list[str],
    candidate_entities: list[str],
) -> str:
    if entity_ids:
        return entity_ids[0]
    if quoted_spans:
        return quoted_spans[0]
    return candidate_entities[0] if candidate_entities else ""


def _is_specific_entity_focus(
    value: str,
    *,
    disallow_terms: tuple[str, ...] = (),
) -> bool:
    normalized = _normalize_text(value)
    if not normalized:
        return False
    if normalized in GENERIC_STOPWORDS:
        return False
    if disallow_terms and _contains_any(normalized, disallow_terms):
        return False
    return True


def _resolve_scene_focus(
    *,
    quoted_spans: list[str],
    entity_ids: list[str],
    candidate_entities: list[str],
) -> str:
    ordered_values = [*quoted_spans, *entity_ids, *candidate_entities]
    for value in ordered_values:
        normalized = value.strip()
        if ":scene:" in normalized:
            return normalized
    for value in ordered_values:
        normalized = value.strip()
        if normalized:
            return normalized
    return ""


def _contains_any(normalized_question: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        normalized_term = _normalize_text(term)
        if not normalized_term:
            continue
        if re.search(r"[\u4e00-\u9fff]", normalized_term) and normalized_term in normalized_question:
            return True
        if re.search(
            rf"(?<![a-z0-9_\-]){re.escape(normalized_term)}(?![a-z0-9_\-])",
            normalized_question,
        ):
            return True
    return False


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    normalized = re.sub(r"[\r\n\t]+", " ", normalized)
    normalized = re.sub(
        r"[,\.\!\?;:\(\)\[\]\{\}\"'`“”‘’，。！？、；：（）【】《》]+",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _question_text_score(question: str) -> int:
    normalized = _normalize_text(question)
    score = 0
    if _match_relation_predicate(normalized) is not None:
        score += 6
    if _match_state_text(normalized) is not None:
        score += 4
    if _contains_any(normalized, RELATION_HINT_TERMS):
        score += 2
    if _contains_any(normalized, STATE_HINT_TERMS):
        score += 2
    if _contains_any(normalized, ENTITY_HINT_TERMS):
        score += 2
    if _contains_any(normalized, TEMPORAL_HINT_TERMS):
        score += 2
    if _contains_any(normalized, SCENE_HINT_TERMS):
        score += 2
    score -= sum(question.count(marker) for marker in ("å", "ä", "æ", "ç", "é", "â", "鍜", "浣", "鏄", "鍏", "崇", "郴"))
    return score


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).strip().lower()
    normalized = re.sub(r"[\r\n\t]+", " ", normalized)
    normalized = re.sub(
        r"""[,.!?;:()\[\]{}"'`“”‘’，。！？；：（）【】《》、]+""",
        " ",
        normalized,
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()
