from __future__ import annotations
import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from vlkg.core.ir.query_annotations import synonym_expansion_enabled, synonym_expansion_metadata
from vlkg.core.temporal.pathing import resolve_repo_input_path
SYNONYM_LEXICON_CONTRACT = 'navqa_synonym_l0_l1_v1'
SYNONYM_ROUTE_IR_QA_ID = 'IR-QA-F-synonym-l0-l1-v1'
DEFAULT_SYNONYM_LEXICON_PATH = Path('configs/datasets/navqa/synonyms.json')
ALLOWED_LEVELS = {'L0', 'L1'}
ALLOWED_ROLES = {'target', 'constraint', 'attribute'}
ALLOWED_SOURCES = {'general_language', 'hsgg_cg_observed'}
NORMALIZED_IR_ALIAS_FIELDS = ('target_entity_aliases', 'attribute_dimension_aliases', 'event_constraint_aliases', 'spatial_constraint_aliases', 'hard_constraint_aliases')
_MAX_ALIASES_PER_FIELD = {'target_entities': 8, 'attribute_dimension': 6, 'event_constraints': 6, 'spatial_constraints': 6, 'hard_constraints': 6}

@dataclass(frozen=True)
class SynonymEntry:
    canonical: str
    aliases: tuple[str, ...]
    level: str
    roles: tuple[str, ...]
    source: str
    source_evidence: tuple[str, ...]
    uses_gold_answer: bool = False

    @property
    def terms(self) -> tuple[str, ...]:
        return _ordered_unique([self.canonical, *self.aliases])

@dataclass(frozen=True)
class SynonymLexicon:
    contract: str
    path: Path
    entries: tuple[SynonymEntry, ...]

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(sorted({entry.level for entry in self.entries}))

    def entries_for_role(self, role: str) -> tuple[SynonymEntry, ...]:
        normalized = _normalize_role(role)
        return tuple((entry for entry in self.entries if normalized in entry.roles))

def load_synonym_lexicon(path: str | Path=DEFAULT_SYNONYM_LEXICON_PATH) -> SynonymLexicon:
    resolved = resolve_repo_input_path(path)
    with resolved.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return synonym_lexicon_from_payload(payload, path=resolved)

def synonym_lexicon_from_payload(payload: Any, *, path: str | Path=DEFAULT_SYNONYM_LEXICON_PATH) -> SynonymLexicon:
    if not isinstance(payload, dict):
        raise ValueError('synonym lexicon must be a JSON object')
    contract = str(payload.get('contract') or '').strip()
    if contract != SYNONYM_LEXICON_CONTRACT:
        raise ValueError(f'synonym lexicon contract must be {SYNONYM_LEXICON_CONTRACT!r}')
    raw_entries = payload.get('entries')
    if not isinstance(raw_entries, list):
        raise ValueError('synonym lexicon entries must be a list')
    entries: list[SynonymEntry] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_entries, start=1):
        entry = _parse_entry(raw, index=index)
        key = _normalize_text(entry.canonical)
        if key in seen:
            raise ValueError(f'duplicate synonym canonical at entry {index}: {entry.canonical}')
        seen.add(key)
        entries.append(entry)
    return SynonymLexicon(contract=contract, path=Path(path), entries=tuple(entries))

def expand_question_ir_with_synonyms(question_ir: dict[str, Any] | None, lexicon: SynonymLexicon | None, *, question: str='') -> dict[str, Any] | None:
    if not isinstance(question_ir, dict) or lexicon is None:
        return question_ir
    expanded = copy.deepcopy(question_ir)
    expansions: list[dict[str, Any]] = []
    target_sources = _clean_text_list(expanded.get('target_entities'))
    if not target_sources:
        target_state = str(expanded.get('target_state') or '').strip()
        if target_state:
            target_sources = [target_state]
    target_aliases, target_records = _aliases_for_sources(target_sources, role='target', field='target_entities', lexicon=lexicon)
    _merge_alias_field(expanded, 'target_entity_aliases', target_aliases)
    expansions.extend(target_records)
    attribute_sources = _clean_text_list(expanded.get('attribute_dimension'))
    attribute_aliases, attribute_records = _aliases_for_sources(attribute_sources, role='attribute', field='attribute_dimension', lexicon=lexicon)
    _merge_alias_field(expanded, 'attribute_dimension_aliases', attribute_aliases)
    expansions.extend(attribute_records)
    for source_field, alias_field in (('event_constraints', 'event_constraint_aliases'), ('spatial_constraints', 'spatial_constraint_aliases'), ('hard_constraints', 'hard_constraint_aliases')):
        aliases, records = _aliases_for_sources(_iter_ir_text_values(expanded.get(source_field)), role='constraint', field=source_field, lexicon=lexicon)
        _merge_alias_field(expanded, alias_field, aliases)
        expansions.extend(records)
    expansion_metadata = {'enabled': True, 'contract': SYNONYM_LEXICON_CONTRACT, 'normalization_scope': ['target_entities', 'attribute_dimension', 'event_constraints', 'spatial_constraints', 'hard_constraints'], 'role_preserving': True, 'support_expansion_enabled': False, 'lexicon_path': str(lexicon.path), 'levels': list(lexicon.levels), 'uses_gold_answers': False, 'uses_dataset_or_domain_wordlist': True, 'question': str(question or ''), 'expanded_terms': expansions, 'alias_fields': {key: list(expanded.get(key) or []) for key in NORMALIZED_IR_ALIAS_FIELDS if expanded.get(key)}}
    previous = expanded.get('synonym_expansion')
    if isinstance(previous, dict):
        expansion_metadata['previous'] = previous
    expanded['synonym_expansion'] = expansion_metadata
    return expanded

def _parse_entry(raw: Any, *, index: int) -> SynonymEntry:
    if not isinstance(raw, dict):
        raise ValueError(f'synonym entry {index} must be an object')
    if raw.get('uses_gold_answer') is True:
        raise ValueError(f'synonym entry {index} uses gold answers')
    canonical = str(raw.get('canonical') or '').strip()
    if not canonical:
        raise ValueError(f'synonym entry {index} missing canonical')
    aliases = _clean_text_list(raw.get('aliases'))
    if not aliases:
        raise ValueError(f'synonym entry {index} missing aliases')
    level = str(raw.get('level') or '').strip()
    if level not in ALLOWED_LEVELS:
        raise ValueError(f'synonym entry {index} level must be L0 or L1')
    roles = tuple((_normalize_role(value) for value in _clean_text_list(raw.get('roles'))))
    if not roles or any((role not in ALLOWED_ROLES for role in roles)):
        raise ValueError(f'synonym entry {index} roles must be target/constraint/attribute')
    source = str(raw.get('source') or '').strip()
    if source not in ALLOWED_SOURCES:
        raise ValueError(f'synonym entry {index} source must be general_language or hsgg_cg_observed')
    source_evidence = tuple(_clean_text_list(raw.get('source_evidence')))
    if not source_evidence:
        raise ValueError(f'synonym entry {index} missing source_evidence')
    return SynonymEntry(canonical=canonical, aliases=tuple(aliases), level=level, roles=tuple(_ordered_unique(roles)), source=source, source_evidence=source_evidence, uses_gold_answer=False)

def _aliases_for_sources(sources: list[str], *, role: str, field: str, lexicon: SynonymLexicon) -> tuple[list[str], list[dict[str, Any]]]:
    aliases: list[str] = []
    records: list[dict[str, Any]] = []
    entries = lexicon.entries_for_role(role)
    max_aliases = _MAX_ALIASES_PER_FIELD.get(field, 6)
    for source_text in sources:
        source = str(source_text or '').strip()
        if not source:
            continue
        for entry, match_terms in _best_entry_matches_for_source(source, entries):
            generated = _entry_aliases_for_source(source, entry, match_terms=match_terms)
            for alias in generated:
                if _normalize_text(alias) == _normalize_text(source):
                    continue
                if len(aliases) >= max_aliases:
                    break
                if alias not in aliases:
                    aliases.append(alias)
                records.append({'role': _normalize_role(role), 'field': field, 'source_text': source, 'alias': alias, 'canonical': entry.canonical, 'level': entry.level, 'source': entry.source, 'source_evidence': list(entry.source_evidence), 'uses_gold_answer': False})
            if len(aliases) >= max_aliases:
                break
    return (aliases, records)

def _best_entry_matches_for_source(source: str, entries: tuple[SynonymEntry, ...]) -> list[tuple[SynonymEntry, tuple[str, ...]]]:
    normalized_source = _normalize_text(source)
    matches: list[tuple[int, int, SynonymEntry, tuple[str, ...]]] = []
    for entry in entries:
        matched_terms: list[str] = []
        best_word_count = 0
        best_char_count = 0
        for term in entry.terms:
            normalized_term = _normalize_text(term)
            if not normalized_term:
                continue
            if normalized_source == normalized_term or _contains_phrase(normalized_source, normalized_term):
                word_count = len(normalized_term.split())
                char_count = len(normalized_term)
                if (word_count, char_count) > (best_word_count, best_char_count):
                    matched_terms = [term]
                    best_word_count = word_count
                    best_char_count = char_count
                elif (word_count, char_count) == (best_word_count, best_char_count):
                    matched_terms.append(term)
        if matched_terms:
            matches.append((best_word_count, best_char_count, entry, tuple(_ordered_unique(matched_terms))))
    if not matches:
        return []
    max_word_count = max((item[0] for item in matches))
    max_char_count = max((item[1] for item in matches if item[0] == max_word_count))
    return [(entry, matched_terms) for word_count, char_count, entry, matched_terms in matches if word_count == max_word_count and char_count == max_char_count]

def _entry_aliases_for_source(source: str, entry: SynonymEntry, *, match_terms: tuple[str, ...]) -> list[str]:
    normalized_source = _normalize_text(source)
    generated: list[str] = []
    for term in match_terms:
        normalized_term = _normalize_text(term)
        if not normalized_term:
            continue
        if normalized_source == normalized_term:
            generated.extend(entry.terms)
            continue
        if _contains_phrase(normalized_source, normalized_term):
            for alias in entry.terms:
                if _normalize_text(alias) == normalized_term:
                    continue
                replacement = _replace_phrase(source, term, alias)
                if replacement:
                    generated.append(replacement)
                generated.append(alias)
    return _ordered_unique(generated)

def _merge_alias_field(payload: dict[str, Any], key: str, values: list[str]) -> None:
    merged = _ordered_unique([*_clean_text_list(payload.get(key)), *values])
    if merged:
        payload[key] = merged

def _iter_ir_text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        values: list[str] = []
        for key in ('text', 'target', 'target_state', 'relation', 'operation', 'must_bind_to'):
            raw = value.get(key)
            if isinstance(raw, list):
                values.extend((str(item).strip() for item in raw if str(item).strip()))
            elif str(raw or '').strip():
                values.append(str(raw).strip())
        nested = value.get('target_entities')
        if isinstance(nested, list):
            values.extend((str(item).strip() for item in nested if str(item).strip()))
        return values
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_iter_ir_text_values(item))
        return values
    return [str(value).strip()] if str(value or '').strip() else []

def _clean_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []

def _normalize_role(value: Any) -> str:
    text = str(value or '').strip().lower().replace('-', '_')
    if text in {'primary_target', 'target_entity'}:
        return 'target'
    if text in {'event', 'spatial', 'hard', 'support'}:
        return 'constraint'
    if text in {'target_attribute', 'attribute_dimension'}:
        return 'attribute'
    return text

def _normalize_text(value: Any) -> str:
    text = str(value or '').lower()
    text = re.sub('[-_/]+', ' ', text)
    text = re.sub('[^a-z0-9]+', ' ', text)
    return re.sub('\\s+', ' ', text).strip()

def _contains_phrase(source: str, phrase: str) -> bool:
    if not source or not phrase:
        return False
    return re.search(f'(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])', source) is not None

def _replace_phrase(source: str, old: str, new: str) -> str:
    normalized_old = _normalize_text(old)
    if not normalized_old:
        return ''
    pattern = re.compile(f'(?<![a-z0-9]){re.escape(normalized_old)}(?![a-z0-9])', re.I)
    normalized_source = _normalize_text(source)
    replaced = pattern.sub(str(new).strip(), normalized_source)
    return re.sub('\\s+', ' ', replaced).strip()

def _ordered_unique(values: list[str] | tuple[str, ...]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or '').strip()
        key = _normalize_text(text)
        if not text or not key or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered
