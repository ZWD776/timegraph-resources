from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from vlkg.core.config.domain_lexicon import (
    DomainLexicon,
    get_domain_lexicon,
)


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
REFERENCE_EVENT_PATTERN = re.compile(r"$^")

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow",
}

ATTRIBUTE_QUERY_WORDS = {
    "appearance",
    "color",
    "colored",
    "colour",
    "colors",
    "look",
    "looks",
    "wearing",
    "wore",
}

ATTRIBUTE_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "appearance": ("appearance", "look", "looks", "wearing", "wore"),
    "color": ("color", "colour", "colored", "colors"),
}

EVENT_ALIASES: dict[str, tuple[str, ...]] = {
    "enter": ("enter", "entered", "entering"),
    "leave": ("leave", "left", "exited", "exit"),
    "cross": ("cross", "crossed", "crossing"),
    "get_off": ("get", "off", "got"),
    "turn": ("turn", "turned"),
    "stop": ("stop", "stopped", "stopping"),
    "open": ("open", "opened", "opening"),
}


@dataclass(frozen=True)
class QuestionPlan:
    question: str
    answer_type: str = "text"
    query_family: str = "general"
    target_entities: tuple[str, ...] = ()
    target_aliases: tuple[str, ...] = ()
    attributes: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    temporal_constraints: tuple[str, ...] = ()
    spatial_constraints: tuple[str, ...] = ()
    current_window_required: bool = False
    ranking_policy: tuple[str, ...] = ()
    retrieval_strategy: str = "hybrid_local"
    preferred_source_kinds: tuple[str, ...] = ()
    answer_rendering: str | None = None
    risk_notes: tuple[str, ...] = ()
    normalized_query_terms: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compile_question(
    question: str,
    *,
    answer_type: str | None = None,
    category: str | None = None,
    domain_profile: str | None = None,
) -> QuestionPlan:
    lexicon = get_domain_lexicon(domain_profile)
    text = _normalize_text(question)
    tokens = tuple(TOKEN_PATTERN.findall(text))
    token_set = set(tokens)
    inferred_answer_type = (answer_type or _infer_answer_type(text, token_set)).strip().lower()
    target_entities = _extract_targets(text, lexicon=lexicon)
    attributes = _extract_attributes(text, token_set)
    events = _extract_events(text)
    temporal_constraints = _temporal_constraints(text, token_set)
    spatial_constraints = _spatial_constraints(text, token_set)
    current_window_required = _current_window_required(text, token_set)
    ranking_policy = _ranking_policy(text, token_set, inferred_answer_type)
    query_family = _query_family(
        inferred_answer_type,
        token_set,
        events,
        temporal_constraints,
        current_window_required,
    )
    target_aliases = _target_aliases(target_entities, lexicon=lexicon)
    retrieval_strategy = _retrieval_strategy(
        query_family,
        inferred_answer_type,
        current_window_required,
        ranking_policy,
    )
    preferred_source_kinds = _preferred_source_kinds(query_family, inferred_answer_type)
    answer_rendering = _answer_rendering(inferred_answer_type)
    risk_notes = _risk_notes(
        text,
        inferred_answer_type,
        target_entities,
        attributes,
        events,
        current_window_required,
    )
    normalized_query_terms = _normalized_query_terms(
        target_entities,
        target_aliases,
        attributes,
        events,
    )
    return QuestionPlan(
        question=question,
        answer_type=inferred_answer_type,
        query_family=query_family,
        target_entities=target_entities,
        target_aliases=target_aliases,
        attributes=attributes,
        events=events,
        temporal_constraints=temporal_constraints,
        spatial_constraints=spatial_constraints,
        current_window_required=current_window_required,
        ranking_policy=ranking_policy,
        retrieval_strategy=retrieval_strategy,
        preferred_source_kinds=preferred_source_kinds,
        answer_rendering=answer_rendering,
        risk_notes=risk_notes,
        normalized_query_terms=normalized_query_terms,
        metadata={
            "category": category or "",
            "compiler": "rule_v1",
            "domain_profile": lexicon.name,
        },
    )


def _normalize_text(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _infer_answer_type(text: str, token_set: set[str]) -> str:
    if text.startswith("what color") or " color " in f" {text} ":
        return "text"
    if text.startswith("did ") or text.startswith("was ") or text.startswith("are "):
        return "binary"
    if text.startswith("where "):
        return "position"
    if text.startswith("when ") or " when " in text:
        return "time"
    if text.startswith("how long") or {"duration", "long"} & token_set:
        return "duration"
    return "text"


def _extract_targets(text: str, *, lexicon: DomainLexicon) -> tuple[str, ...]:
    targets: list[str] = []
    constraint_spans = _constraint_clause_spans(text)
    for phrase, target in lexicon.phrase_targets:
        if (
            _contains_phrase(text, phrase)
            and not _phrase_only_occurs_in_constraint(text, phrase, constraint_spans)
            and target not in targets
        ):
            targets.append(target)
    if ("person" in text or "people" in text) and "person" not in targets:
        targets.append("person")
    return tuple(targets)


def _contains_phrase(text: str, phrase: str) -> bool:
    parts = [re.escape(part) for part in TOKEN_PATTERN.findall(phrase.lower())]
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _constraint_clause_spans(text: str) -> tuple[tuple[int, int], ...]:
    pattern = re.compile(
        r"\b(?:near|nearby|around|beside|next\s+to|close\s+to|in\s+front\s+of|behind|"
        r"after|before|during|while|since|until|between)\b[^,?.;]*"
    )
    return tuple((match.start(), match.end()) for match in pattern.finditer(text))


def _phrase_only_occurs_in_constraint(
    text: str,
    phrase: str,
    constraint_spans: tuple[tuple[int, int], ...],
) -> bool:
    parts = [re.escape(part) for part in TOKEN_PATTERN.findall(phrase.lower())]
    if not parts:
        return False
    pattern = re.compile(r"(?<![a-z0-9])" + r"\s+".join(parts) + r"(?![a-z0-9])")
    matches = tuple(pattern.finditer(text))
    if not matches:
        return False
    return all(
        any(start <= match.start() and match.end() <= end for start, end in constraint_spans)
        for match in matches
    )


def _extract_attributes(text: str, token_set: set[str]) -> tuple[str, ...]:
    attributes: list[str] = []
    for color in sorted(COLOR_WORDS & token_set):
        attributes.append(color)
    if "what color" in text or bool({"color", "colour", "colored", "colors"} & token_set):
        attributes.append("color")
    if bool({"appearance", "wearing", "wore"} & token_set) or "look like" in text:
        attributes.append("appearance")
    return tuple(dict.fromkeys(attributes))


def _target_aliases(
    targets: tuple[str, ...],
    *,
    lexicon: DomainLexicon,
) -> tuple[str, ...]:
    aliases: list[str] = []
    for target in targets:
        for alias in lexicon.entity_aliases.get(target, (target,)):
            if alias not in aliases:
                aliases.append(alias)
    return tuple(aliases)


def _extract_events(text: str) -> tuple[str, ...]:
    events: list[str] = []
    checks = (
        ("enter", ("enter", "entered", "entering")),
        ("leave", ("leave", "left", "leaving", "exited", "exit")),
        ("cross", ("cross", "crossed", "crossing")),
        ("get_off", ("get off", "got off")),
        ("turn", ("turn", "turned")),
        ("stop", ("stopped", "stop")),
        ("open", ("open", "opened", "opening")),
    )
    for event, phrases in checks:
        if any(phrase in text for phrase in phrases) and event not in events:
            events.append(event)
    return tuple(events)


def _temporal_constraints(text: str, token_set: set[str]) -> tuple[str, ...]:
    constraints: list[str] = []
    if "first" in token_set:
        constraints.append("earliest")
    if "right now" in text or "current" in token_set:
        constraints.append("current")
    if "after" in token_set:
        constraints.append("after_event")
    if "before" in token_set:
        constraints.append("before_event")
    if "since" in token_set or text.startswith("how long have"):
        constraints.append("since_event_to_current")
    if text.startswith("how long were") or text.startswith("how long was"):
        constraints.append("event_interval")
    if text.startswith("when "):
        constraints.append("event_time")
    return tuple(constraints)


def _spatial_constraints(text: str, token_set: set[str]) -> tuple[str, ...]:
    constraints: list[str] = []
    if "closest" in token_set:
        constraints.append("closest")
    if "nearest" in token_set:
        constraints.append("nearest")
    if "in front of" in text:
        constraints.append("in_front_of_landmark")
    if "near" in token_set:
        constraints.append("near_reference_event")
    return tuple(constraints)


def _current_window_required(text: str, token_set: set[str]) -> bool:
    return "right now" in text or "current" in token_set or "currently" in token_set


def _ranking_policy(
    text: str,
    token_set: set[str],
    answer_type: str,
) -> tuple[str, ...]:
    policies: list[str] = []
    if "closest" in token_set or "nearest" in token_set:
        policies.append("distance_or_recency_proxy")
    if "first" in token_set:
        policies.append("earliest_event")
    if _current_window_required(text, token_set):
        policies.append("question_end_window_first")
    if answer_type == "position":
        policies.append("target_event_time_then_pose_render")
    if policies:
        return tuple(policies)
    return ("semantic_relevance",)


def _query_family(
    answer_type: str,
    token_set: set[str],
    events: tuple[str, ...],
    temporal_constraints: tuple[str, ...],
    current_window_required: bool,
) -> str:
    if current_window_required:
        return "current_state"
    if answer_type == "duration":
        return "duration"
    if answer_type == "position":
        return "position_target"
    if COLOR_WORDS & token_set or ATTRIBUTE_QUERY_WORDS & token_set:
        return "attribute"
    if {"side", "direction", "driving", "walking", "turn"} & token_set:
        return "ego_motion"
    if answer_type == "time" or events or temporal_constraints:
        return "temporal_event"
    if answer_type == "binary":
        return "existence"
    return "descriptive"


def _retrieval_strategy(
    query_family: str,
    answer_type: str,
    current_window_required: bool,
    ranking_policy: tuple[str, ...],
) -> str:
    if current_window_required:
        return "question_window_scene_first"
    if query_family in {"temporal_event", "duration"}:
        return "temporal_axis_then_target_filter"
    if answer_type == "position":
        return "target_entity_scene_then_event_time"
    if query_family == "attribute":
        return "target_entity_attribute_binding"
    if query_family == "ego_motion":
        return "ego_event_or_scene_transition"
    if "distance_or_recency_proxy" in ranking_policy:
        return "target_candidates_ranked_by_temporal_or_spatial_proxy"
    return "hybrid_local"


def _preferred_source_kinds(query_family: str, answer_type: str) -> tuple[str, ...]:
    if query_family in {"temporal_event", "duration", "ego_motion", "current_state"}:
        return ("temporal_composition", "transition_event", "scene", "episode")
    if answer_type == "position":
        return ("entity", "stable_identity", "scene", "episode")
    if query_family == "attribute":
        return ("episode", "state_value", "entity", "stable_identity")
    if query_family == "existence":
        return ("entity", "stable_identity", "scene", "episode")
    return ("scene", "entity", "episode")


def _answer_rendering(answer_type: str) -> str | None:
    if answer_type == "position":
        return "ego_pose_at_retrieved_event_time"
    if answer_type == "time":
        return "minutes_before_question_end_time"
    if answer_type == "duration":
        return "duration_minutes"
    if answer_type == "binary":
        return "yes_no"
    return "natural_language"


def _risk_notes(
    text: str,
    answer_type: str,
    target_entities: tuple[str, ...],
    attributes: tuple[str, ...],
    events: tuple[str, ...],
    current_window_required: bool,
) -> tuple[str, ...]:
    notes: list[str] = []
    if answer_type in {"position", "binary", "time", "duration"} and not target_entities and not events:
        notes.append("no_specific_target_or_event_detected")
    if attributes and target_entities:
        notes.append("attribute_must_bind_to_target_entity")
    if current_window_required:
        notes.append("must_not_answer_from_anywhere_in_video")
    if "closest" in text or "nearest" in text:
        notes.append("closest_requires_spatial_pose_or_proxy_ranking")
    token_set = set(TOKEN_PATTERN.findall(text))
    if "side" in token_set or "direction" in token_set:
        notes.append("ego_motion_may_need_pose_or_heading_signal")
    if "in front of" in text:
        notes.append("landmark_relation_may_be_missing_from_hsgg")
    return tuple(notes)


def _normalized_query_terms(
    target_entities: tuple[str, ...],
    target_aliases: tuple[str, ...],
    attributes: tuple[str, ...],
    events: tuple[str, ...],
) -> tuple[str, ...]:
    terms: list[str] = []
    for group in (target_entities, target_aliases, attributes, events):
        for item in group:
            for token in TOKEN_PATTERN.findall(item.lower()):
                if token not in terms:
                    terms.append(token)
    for event in events:
        for alias in EVENT_ALIASES.get(event, ()):
            for token in TOKEN_PATTERN.findall(alias.lower()):
                if token not in terms:
                    terms.append(token)
    return tuple(terms)
