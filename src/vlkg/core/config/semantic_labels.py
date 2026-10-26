from __future__ import annotations

import re


_TOKEN_PATTERN = re.compile(r"[^a-z0-9]+")

LABEL_ALIASES: dict[str, str] = {}
SEMANTIC_GROUPS: dict[str, str] = {}
SALIENT_CANONICAL_LABELS: set[str] = set()
SALIENT_SEMANTIC_GROUPS: set[str] = set()


def normalize_label_text(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower().replace("-", " ")
    normalized = " ".join(token for token in _TOKEN_PATTERN.split(normalized) if token)
    return normalized


def canonical_label(value: str | None) -> str:
    return normalize_label_text(value)


def semantic_group(value: str | None) -> str:
    return canonical_label(value)


def label_alias_terms(value: str | None) -> tuple[str, ...]:
    canonical = canonical_label(value)
    return (canonical,) if canonical else ()


def is_salient_label(value: str | None) -> bool:
    return False
