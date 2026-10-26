from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


GENERIC_DOMAIN_PROFILE = "generic"
DEFAULT_DOMAIN_PROFILE = GENERIC_DOMAIN_PROFILE


@dataclass(frozen=True)
class DomainLexicon:
    name: str
    entity_aliases: Mapping[str, tuple[str, ...]]
    phrase_targets: tuple[tuple[str, str], ...]
    core_term_aliases: Mapping[str, tuple[str, ...]]
    core_phrase_aliases: Mapping[str, tuple[str, ...]]


GENERIC_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {}
GENERIC_PHRASE_TARGETS: tuple[tuple[str, str], ...] = ()
GENERIC_CORE_TERM_ALIASES: dict[str, tuple[str, ...]] = {}
GENERIC_CORE_PHRASE_ALIASES: dict[str, tuple[str, ...]] = {}


_PROFILES: dict[str, DomainLexicon] = {
    GENERIC_DOMAIN_PROFILE: DomainLexicon(
        name=GENERIC_DOMAIN_PROFILE,
        entity_aliases=MappingProxyType(GENERIC_ENTITY_ALIASES),
        phrase_targets=GENERIC_PHRASE_TARGETS,
        core_term_aliases=MappingProxyType(GENERIC_CORE_TERM_ALIASES),
        core_phrase_aliases=MappingProxyType(GENERIC_CORE_PHRASE_ALIASES),
    ),
}


def available_domain_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


def resolve_domain_profile(profile: str | None) -> str:
    normalized = str(profile or DEFAULT_DOMAIN_PROFILE).strip().lower().replace("-", "_")
    aliases = {
        "default": DEFAULT_DOMAIN_PROFILE,
        "generic_only": GENERIC_DOMAIN_PROFILE,
        "generic-only": GENERIC_DOMAIN_PROFILE,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _PROFILES:
        valid = ", ".join(available_domain_profiles())
        raise ValueError(f"Unknown domain profile {profile!r}; expected one of: {valid}")
    return normalized


def get_domain_lexicon(profile: str | None = None) -> DomainLexicon:
    return _PROFILES[resolve_domain_profile(profile)]
