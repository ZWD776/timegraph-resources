from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from vlkg.core.graph_retrieval.canonical_record_vector_recall import (
    CanonicalRecordVectorIndex,
    canonical_vector_query_text,
    record_semantic_text,
    vector_cache_manifest_path,
)
from vlkg.core.graph_retrieval.canonical_retrieval_types import (
    CanonicalProjection,
    GraphRecordView,
)
from vlkg.core.memory.types import MemoryQuery


class _SemanticProbeEncoder:
    def encode(
        self,
        sentences: Sequence[str],
        **_: object,
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for sentence in sentences:
            text = sentence.lower()
            cooling = any(
                phrase in text
                for phrase in ("air conditioner", "cool down", "cooling appliance")
            )
            rows.append([1.0, 0.0] if cooling else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def _record(
    record_id: str,
    *,
    label: str,
    raw_visual_attribute: str = "",
) -> GraphRecordView:
    return GraphRecordView(
        source_kind="entity_attribute" if raw_visual_attribute else "entity",
        source_record_id=record_id,
        summary=f"Entity internal:{record_id} is {label}",
        metadata={
            "entity_id": f"hm3d-v0__episode:{record_id}",
            "label": label,
            "canonical_label": label,
            "attr_key": "description",
            "attr_value": raw_visual_attribute,
            "raw_visual_attribute": raw_visual_attribute,
        },
        index_text=f"{label} {raw_visual_attribute}".strip(),
        tokens=frozenset(label.lower().split()),
        video_id="episode",
        scene_ids=(),
        entity_ids=(record_id,),
        start_ts=None,
        end_ts=None,
        priority=5,
    )


def test_vector_query_uses_semantics_but_excludes_route_and_evidence_templates() -> None:
    query = MemoryQuery(
        query_text="expanded query ; visual ; kind ; role",
        metadata={
            "raw_question": "What could cool down the room?",
            "question_ir": {
                "target_entities": ["cooling appliance"],
                "attribute_dimension": "function",
                "spatial_constraints": [{"relation": "on", "target": "wall"}],
                "evidence_requirements": ["visual", "kind", "role"],
                "graph_information_needs": ["attribute"],
                "question_route": {"route_id": "world_knowledge"},
                "category": "world knowledge",
            },
        },
    )

    text = canonical_vector_query_text(query)

    assert "What could cool down the room?" in text
    assert "cooling appliance" in text
    assert "spatial: on; wall" in text
    assert "world_knowledge" not in text
    assert "evidence" not in text
    assert "graph information" not in text
    assert "visual" not in text
    assert "kind" not in text
    assert "role" not in text


def test_record_semantic_text_uses_atomic_fields_without_internal_ids() -> None:
    record = _record(
        "entity:hm3d-v0__episode:air-conditioner-123",
        label="air conditioner",
        raw_visual_attribute="white wall-mounted cooling unit",
    )

    text = record_semantic_text(record)

    assert "entity air conditioner" in text
    assert "white wall-mounted cooling unit" in text
    assert "hm3d-v0__" not in text
    assert "air-conditioner-123" not in text


def test_vector_index_recalls_semantic_record_without_token_overlap(
    tmp_path: Path,
) -> None:
    projection = CanonicalProjection()
    cooling = _record("cooling", label="air conditioner")
    vase = _record("vase", label="decorative vase")
    projection.add_record(cooling)
    projection.add_record(vase)
    cache_path = tmp_path / "canonical_vectors.npz"
    encoder = _SemanticProbeEncoder()

    index = CanonicalRecordVectorIndex.build_or_load(
        projection,
        cache_path=cache_path,
        model_name="semantic-probe",
        encoder=encoder,
    )
    hits = index.search(
        "What could cool down this room?",
        top_k=1,
        min_similarity=0.0,
    )

    assert hits[0].record_id == "cooling"
    assert not ({"cool", "down"} & set(cooling.tokens))
    assert cache_path.exists()
    manifest = json.loads(vector_cache_manifest_path(cache_path).read_text())
    assert manifest["uses_images_or_crops"] is False
    assert manifest["uses_visual_faiss"] is False
    assert manifest["uses_gold_answers"] is False

    loaded = CanonicalRecordVectorIndex.build_or_load(
        projection,
        cache_path=cache_path,
        model_name="semantic-probe",
        encoder=encoder,
    )
    assert loaded.cache_status == "loaded"
    assert loaded.search("cool down", top_k=1, min_similarity=0.0)[0].record_id == (
        "cooling"
    )
