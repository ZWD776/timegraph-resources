from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceSourceKind = Literal[
    "scene_anchor",
    "entity_identity",
    "entity_state",
    "interaction",
]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_kind: EvidenceSourceKind
    source_record_id: str
    summary: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_text: str | None = None
    answer_brief: str | None = None
    items: list[EvidenceItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryQueryResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend_name: str
    evidence_bundle: EvidenceBundle
    raw_backend_payload: dict[str, Any] = Field(default_factory=dict)
