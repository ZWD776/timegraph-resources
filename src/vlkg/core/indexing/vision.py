from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from vlkg.core.models.temporal_graph import BoundingBox
from vlkg.core.assets.source_paths import resolve_data_path


class EmbeddingCache:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.db_path)
        self._stats = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                image_path TEXT NOT NULL,
                bbox_key TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (image_path, bbox_key)
            )
            """
        )
        self._connection.commit()

    @staticmethod
    def bbox_to_key(bbox: BoundingBox) -> str:
        return (
            f"{bbox.x1:.3f},{bbox.y1:.3f},{bbox.x2:.3f},{bbox.y2:.3f}"
        )

    def get(self, image_path: str | Path, bbox: BoundingBox) -> np.ndarray | None:
        self._stats["requests"] += 1
        normalized_path = str(Path(image_path))
        bbox_key = self.bbox_to_key(bbox)
        cursor = self._connection.execute(
            """
            SELECT embedding, dim
            FROM embedding_cache
            WHERE image_path = ? AND bbox_key = ?
            """,
            (normalized_path, bbox_key),
        )
        row = cursor.fetchone()
        if row is None:
            self._stats["misses"] += 1
            return None

        embedding = np.frombuffer(row[0], dtype=np.float32).copy()
        if embedding.size != row[1]:
            self._stats["misses"] += 1
            return None
        self._stats["hits"] += 1
        return embedding

    def get_many(
        self,
        items: Sequence[tuple[str | Path, BoundingBox]],
    ) -> dict[tuple[str, str], np.ndarray]:
        hits: dict[tuple[str, str], np.ndarray] = {}
        for image_path, bbox in items:
            embedding = self.get(image_path=image_path, bbox=bbox)
            if embedding is None:
                continue
            normalized_path = str(Path(image_path))
            hits[(normalized_path, self.bbox_to_key(bbox))] = embedding
        return hits

    def set_many(
        self,
        items: Sequence[tuple[str | Path, BoundingBox, np.ndarray]],
    ) -> None:
        if not items:
            return

        self._stats["writes"] += len(items)
        records = [
            (
                str(Path(image_path)),
                self.bbox_to_key(bbox),
                np.asarray(embedding, dtype=np.float32).tobytes(),
                int(np.asarray(embedding).shape[-1]),
            )
            for image_path, bbox, embedding in items
        ]
        self._connection.executemany(
            """
            INSERT INTO embedding_cache (image_path, bbox_key, embedding, dim)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(image_path, bbox_key)
            DO UPDATE SET embedding = excluded.embedding, dim = excluded.dim
            """,
            records,
        )
        self._connection.commit()

    def get_stats(self) -> dict[str, int | float]:
        requests = int(self._stats["requests"])
        hits = int(self._stats["hits"])
        misses = int(self._stats["misses"])
        writes = int(self._stats["writes"])
        hit_rate = hits / requests if requests else 0.0
        return {
            "requests": requests,
            "hits": hits,
            "misses": misses,
            "writes": writes,
            "hit_rate": hit_rate,
        }

    def reset_stats(self) -> None:
        self._stats = {
            "requests": 0,
            "hits": 0,
            "misses": 0,
            "writes": 0,
        }

    def close(self) -> None:
        if getattr(self, "_connection", None) is not None:
            self._connection.close()
            self._connection = None

    def __del__(self) -> None:
        self.close()


class VisionFeatureExtractor:
    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_files_only = local_files_only
        self._processor = None
        self._model = None

    def extract_embedding(
        self,
        image_path: str | Path,
        bbox: BoundingBox,
    ) -> np.ndarray:
        return self.encode_batch([(image_path, bbox)])[0]

    def encode_batch(
        self,
        items: Sequence[tuple[str | Path, BoundingBox]],
    ) -> list[np.ndarray]:
        if not items:
            return []

        self._ensure_loaded()
        crops = [
            self._crop_image(image_path=image_path, bbox=bbox)
            for image_path, bbox in items
        ]
        inputs = self._processor(images=crops, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            if hasattr(self._model, "get_image_features"):
                raw_output = self._model.get_image_features(**inputs)
            else:
                raw_output = self._model(**inputs)
            embedding = self._unwrap_embedding(raw_output)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)

        return [
            row.detach().cpu().numpy().astype(np.float32)
            for row in embedding
        ]

    def encode_text_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []

        self._ensure_loaded()
        inputs = self._processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            if hasattr(self._model, "get_text_features"):
                raw_output = self._model.get_text_features(**inputs)
            else:
                raw_output = self._model.text_model(**inputs)
            embedding = self._unwrap_text_embedding(raw_output)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)

        return [
            row.detach().cpu().numpy().astype(np.float32)
            for row in embedding
        ]

    def _unwrap_embedding(self, raw_output: object) -> torch.Tensor:
        if isinstance(raw_output, torch.Tensor):
            return raw_output
        if hasattr(raw_output, "image_embeds") and raw_output.image_embeds is not None:
            return raw_output.image_embeds
        if hasattr(raw_output, "pooler_output") and raw_output.pooler_output is not None:
            return raw_output.pooler_output
        if hasattr(raw_output, "last_hidden_state") and raw_output.last_hidden_state is not None:
            # Fallback: average pool token embeddings if only hidden states are available.
            return raw_output.last_hidden_state.mean(dim=1)
        raise TypeError(
            "Unsupported SigLIP output type. Expected Tensor or model output with "
            f"image_embeds/pooler_output/last_hidden_state, got: {type(raw_output)!r}"
        )

    def _unwrap_text_embedding(self, raw_output: object) -> torch.Tensor:
        if isinstance(raw_output, torch.Tensor):
            return raw_output
        if hasattr(raw_output, "text_embeds") and raw_output.text_embeds is not None:
            return raw_output.text_embeds
        if hasattr(raw_output, "pooler_output") and raw_output.pooler_output is not None:
            return raw_output.pooler_output
        if hasattr(raw_output, "last_hidden_state") and raw_output.last_hidden_state is not None:
            return raw_output.last_hidden_state.mean(dim=1)
        raise TypeError(
            "Unsupported SigLIP text output type. Expected Tensor or model output with "
            f"text_embeds/pooler_output/last_hidden_state, got: {type(raw_output)!r}"
        )

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        self._model.to(self.device)

    def _crop_image(
        self,
        *,
        image_path: str | Path,
        bbox: BoundingBox,
    ) -> Image.Image:
        image = Image.open(resolve_data_path(image_path)).convert("RGB")
        width, height = image.size

        x1 = max(0, min(int(round(bbox.x1)), width - 1))
        y1 = max(0, min(int(round(bbox.y1)), height - 1))
        x2 = max(x1 + 1, min(int(round(bbox.x2)), width))
        y2 = max(y1 + 1, min(int(round(bbox.y2)), height))

        return image.crop((x1, y1, x2, y2))
