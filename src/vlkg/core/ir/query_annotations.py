"""Inspect caller-supplied synonym annotations without loading a dataset lexicon."""
from __future__ import annotations
from typing import Any

def synonym_expansion_enabled(question_ir: dict[str, Any] | None) -> bool:
    if not isinstance(question_ir, dict):
        return False
    metadata = question_ir.get('synonym_expansion')
    return isinstance(metadata, dict) and metadata.get('enabled') is True

def synonym_expansion_metadata(question_ir: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(question_ir, dict):
        return {}
    metadata = question_ir.get('synonym_expansion')
    return dict(metadata) if isinstance(metadata, dict) else {}
