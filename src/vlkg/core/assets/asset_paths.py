"""Boundary-aware path translation for explicit, dry-run migration maps."""
from __future__ import annotations

from pathlib import Path, PurePosixPath


def _normalize(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def _relative(value: str) -> str:
    text = _normalize(value)
    if not text or text.startswith('/') or ':' in text or '..' in PurePosixPath(text).parts:
        raise ValueError(f"mapping requires a repository-relative path: {value}")
    return text


def validate_mapping(mapping: dict[str, str]) -> None:
    sources, targets = [], []
    for source, target in mapping.items():
        sources.append(_relative(source).casefold())
        targets.append(_relative(target).casefold())
    if len(set(sources)) != len(sources):
        raise ValueError("duplicate source mapping")
    for i, target in enumerate(targets):
        for other in targets[i + 1:]:
            if target == other or target.startswith(other + '/') or other.startswith(target + '/'):
                raise ValueError("target roots overlap; give each asset a distinct destination")


def translate_path(value: str, mapping: dict[str, str], root: Path) -> str:
    """Translate one decoded path, not arbitrary prose or serialized JSON text.

    Case-insensitive comparisons support Windows provenance. Longer source roots
    win, and an underscore is never treated as a directory boundary.
    """
    validate_mapping(mapping)
    normalized = _normalize(value)
    root_text = _normalize(str(root.resolve()))
    absolute = normalized.casefold().startswith(root_text.casefold() + '/')
    relative = normalized[len(root_text) + 1:] if absolute else normalized
    for source, target in sorted(mapping.items(), key=lambda p: len(p[0]), reverse=True):
        src = _relative(source)
        if relative.casefold() == src.casefold() or relative.casefold().startswith(src.casefold() + '/'):
            translated = _relative(target) + relative[len(src):]
            return root_text + '/' + translated if absolute else translated
    return value
