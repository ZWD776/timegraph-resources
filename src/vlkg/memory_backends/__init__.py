"""Backend-neutral memory contracts for structured graph memory."""

from vlkg.core.memory.base import MemoryBackend
from vlkg.memory_backends.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceSourceKind,
    MemoryQueryResult,
)
from vlkg.core.memory.types import (
    EntityIdentityRecord,
    EntityStateRecord,
    InteractionRecord,
    MemoryQuery,
    MemoryRecordBatch,
    SceneAnchorRecord,
)

__all__ = [
    "EntityIdentityRecord",
    "EntityStateRecord",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceSourceKind",
    "InteractionRecord",
    "MemoryBackend",
    "MemoryQuery",
    "MemoryQueryResult",
    "MemoryRecordBatch",
    "SceneAnchorRecord",
]
