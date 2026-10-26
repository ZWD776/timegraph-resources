from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Sequence, TextIO

import numpy as np

from vlkg.core.graph_retrieval.evidence_demand import EVIDENCE_UNIT_OBJECT_CROP, visual_query_intents


OBJECT_VISUAL_INDEX_CONTRACT = "object_visual_graph_index.v1"
OBJECT_VISUAL_SEARCH_CONTRACT = "object_visual_graph_search.v1"
_STREAM_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ObservationBinding:
    observation_id: str
    frame_name: str
    frame_index: int
    object_index: int
    source_entity_id: str
    canonical_entity_id: str
    scene_id: str
    image_path: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphObservationBindingIndex:
    """Exact frame/object binding from HSGG detections into canonical entities."""

    def __init__(
        self,
        *,
        observations: dict[tuple[int, int], dict[str, Any]],
        canonical_entity_by_source: dict[str, str],
    ) -> None:
        self._observations = observations
        self._canonical_entity_by_source = canonical_entity_by_source

    @classmethod
    def from_graphs(
        cls,
        *,
        temporal_graph_path: str | Path,
        canonical_graph_path: str | Path,
    ) -> "GraphObservationBindingIndex":
        observations: dict[tuple[int, int], dict[str, Any]] = {}
        with Path(temporal_graph_path).open("r", encoding="utf-8") as handle:
            for row in iter_json_array_objects(handle, "observations"):
                frame_index = _optional_int(row.get("frame_index"))
                object_index = _optional_int(row.get("object_index"))
                source_entity_id = _clean_text(row.get("entity_id"))
                if frame_index is None or object_index is None or not source_entity_id:
                    continue
                key = (frame_index, object_index)
                if key in observations:
                    observations[key] = {"ambiguous": True}
                    continue
                observations[key] = row

        canonical_entity_by_source: dict[str, str] = {}
        with Path(canonical_graph_path).open("r", encoding="utf-8") as handle:
            for row in iter_json_array_objects(handle, "entity_nodes"):
                canonical_entity_id = _clean_text(row.get("entity_id"))
                if not canonical_entity_id:
                    continue
                attributes = row.get("attributes")
                attributes = attributes if isinstance(attributes, dict) else {}
                source_ids = [
                    *_text_values(row.get("source_entity_ids")),
                    *_text_values(attributes.get("source_entity_ids")),
                    *_text_values(attributes.get("member_entity_ids")),
                ]
                for source_entity_id in source_ids:
                    previous = canonical_entity_by_source.get(source_entity_id)
                    if previous is not None and previous != canonical_entity_id:
                        canonical_entity_by_source[source_entity_id] = ""
                    elif previous is None:
                        canonical_entity_by_source[source_entity_id] = canonical_entity_id
        return cls(
            observations=observations,
            canonical_entity_by_source=canonical_entity_by_source,
        )

    @property
    def observation_count(self) -> int:
        return sum(1 for row in self._observations.values() if not row.get("ambiguous"))

    @property
    def canonical_source_count(self) -> int:
        return sum(1 for value in self._canonical_entity_by_source.values() if value)

    def bind(self, *, frame_index: int, object_index: int) -> ObservationBinding | None:
        row = self._observations.get((int(frame_index), int(object_index)))
        if not isinstance(row, dict) or row.get("ambiguous"):
            return None
        source_entity_id = _clean_text(row.get("entity_id"))
        canonical_entity_id = self._canonical_entity_by_source.get(source_entity_id, "")
        if not source_entity_id or not canonical_entity_id:
            return None
        return ObservationBinding(
            observation_id=_clean_text(row.get("observation_id")),
            frame_name=_clean_text(row.get("frame_name")),
            frame_index=int(frame_index),
            object_index=int(object_index),
            source_entity_id=source_entity_id,
            canonical_entity_id=canonical_entity_id,
            scene_id=_clean_text(row.get("scene_id")),
            image_path=_clean_text(row.get("image_path")),
            label=_clean_text(row.get("label")),
        )


class ObjectVisualSearchIndex:
    """Read-only SigLIP object search whose hits are already graph-bound."""

    def __init__(
        self,
        *,
        index_dir: str | Path,
        index_name: str = "openeqa_object_siglip_flat_ip",
        text_encoder: Callable[[Sequence[str]], Sequence[np.ndarray]] | None = None,
    ) -> None:
        import faiss

        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.index_path = self.index_dir / f"{index_name}.faiss"
        self.metadata_path = self.index_dir / f"{index_name}.metadata.sqlite"
        self.manifest_path = self.index_dir / f"{index_name}.manifest.json"
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if not self.metadata_path.is_file():
            raise FileNotFoundError(self.metadata_path)
        self.index = faiss.read_index(str(self.index_path))
        self.connection = sqlite3.connect(self.metadata_path)
        self.connection.row_factory = sqlite3.Row
        self._faiss = faiss
        self._text_encoder = text_encoder
        metadata_count = int(
            self.connection.execute("SELECT COUNT(*) FROM object_metadata").fetchone()[0]
        )
        if metadata_count != int(self.index.ntotal):
            raise ValueError(
                f"FAISS/metadata mismatch: index={self.index.ntotal}, metadata={metadata_count}"
            )

    def close(self) -> None:
        self.connection.close()

    def search_evidence_demand(
        self,
        *,
        episode_history: str,
        evidence_demand: dict[str, Any],
        top_k: int = 8,
        search_k: int = 1000,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        intents = [
            row
            for row in visual_query_intents(evidence_demand)
            if row.get("evidence_unit") == EVIDENCE_UNIT_OBJECT_CROP
        ]
        rendered = [f"a photo of {row['text']}" for row in intents]
        vectors = self._encode_texts(rendered)
        groups: list[dict[str, Any]] = []
        seen: set[int] = set()
        for intent, query_text, vector in zip(intents, rendered, vectors, strict=True):
            hits = self.search_vector(
                episode_history=episode_history,
                vector=vector,
                top_k=top_k,
                search_k=search_k,
                min_score=min_score,
            )
            unique_hits = []
            for hit in hits:
                faiss_id = int(hit["faiss_id"])
                if faiss_id in seen:
                    continue
                seen.add(faiss_id)
                unique_hits.append(hit)
            if unique_hits:
                groups.append(
                    {
                        "query_intent": intent,
                        "rendered_query": query_text,
                        "hits": unique_hits,
                    }
                )
        return {
            "contract": OBJECT_VISUAL_SEARCH_CONTRACT,
            "available": bool(groups),
            "episode_history": episode_history,
            "uses_gold_answer": False,
            "exact_graph_binding_required": True,
            "query_intent_count": len(intents),
            "groups": groups,
        }

    def search_vector(
        self,
        *,
        episode_history: str,
        vector: np.ndarray,
        top_k: int,
        search_k: int,
        min_score: float,
    ) -> list[dict[str, Any]]:
        query = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self._faiss.normalize_L2(query)
        resolved_search_k = min(int(self.index.ntotal), max(int(top_k), int(search_k)))
        scores, ids = self.index.search(query, resolved_search_k)
        candidate_ids = [int(value) for value in ids[0] if int(value) >= 0]
        metadata = self._metadata_by_ids(candidate_ids)
        hits: list[dict[str, Any]] = []
        for score, faiss_id in zip(scores[0], ids[0], strict=True):
            faiss_id = int(faiss_id)
            if faiss_id < 0 or float(score) < float(min_score):
                continue
            row = metadata.get(faiss_id)
            if row is None or row.get("episode_history") != episode_history:
                continue
            row["similarity"] = float(score)
            row["rank"] = len(hits) + 1
            hits.append(row)
            if len(hits) >= int(top_k):
                break
        return hits

    def _encode_texts(self, texts: Sequence[str]) -> Sequence[np.ndarray]:
        if not texts:
            return []
        if self._text_encoder is None:
            from vlkg.core.indexing.vision import VisionFeatureExtractor

            extractor = VisionFeatureExtractor(local_files_only=True)
            self._text_encoder = extractor.encode_text_batch
        return self._text_encoder(texts)

    def _metadata_by_ids(self, ids: Sequence[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self.connection.execute(
            f"SELECT faiss_id, raw_json FROM object_metadata WHERE faiss_id IN ({placeholders})",
            [int(value) for value in ids],
        ).fetchall()
        return {
            int(row["faiss_id"]): json.loads(str(row["raw_json"]))
            for row in rows
        }


def iter_json_array_objects(handle: TextIO, property_name: str) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    marker = f'"{property_name}"'
    buffer = ""
    found_property = False
    in_array = False
    finished = False
    while not finished:
        chunk = handle.read(_STREAM_CHUNK_SIZE)
        if not chunk:
            finished = True
        buffer += chunk
        while True:
            if not found_property:
                index = buffer.find(marker)
                if index < 0:
                    buffer = buffer[-max(0, len(marker) - 1) :]
                    break
                buffer = buffer[index + len(marker) :]
                found_property = True
            if not in_array:
                index = buffer.find("[")
                if index < 0:
                    if finished:
                        raise ValueError(f"JSON array not found: {property_name}")
                    break
                buffer = buffer[index + 1 :]
                in_array = True
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return
            if not buffer:
                break
            try:
                item, end_index = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if finished:
                    raise
                break
            if isinstance(item, dict):
                yield item
            buffer = buffer[end_index:]


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _text_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _clean_text(value)
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        return [text for item in value for text in _text_values(item)]
    return []


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())
