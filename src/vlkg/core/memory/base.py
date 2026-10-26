from __future__ import annotations

from abc import ABC, abstractmethod

from vlkg.memory_backends.evidence import EvidenceBundle, MemoryQueryResult
from vlkg.core.memory.types import MemoryQuery, MemoryRecordBatch


class MemoryBackend(ABC):
    backend_name = "memory_backend"

    @abstractmethod
    async def ingest_batch(self, batch: MemoryRecordBatch) -> None:
        """Persist a normalized batch into the backend."""

    @abstractmethod
    async def retrieve_temporal_context(self, query: MemoryQuery) -> MemoryQueryResult:
        """Retrieve time-scoped evidence for a query."""

    @abstractmethod
    async def retrieve_entity_context(self, query: MemoryQuery) -> MemoryQueryResult:
        """Retrieve entity-scoped evidence for a query."""

    async def build_evidence_bundle(self, query: MemoryQuery) -> EvidenceBundle:
        result = await self.retrieve_temporal_context(query)
        return result.evidence_bundle

    async def close(self) -> None:
        """Release backend resources when needed."""
