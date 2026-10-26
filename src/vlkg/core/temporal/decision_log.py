from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class DecisionLogRecord:
    phase: str
    provider: str
    reason: str
    video_id: str
    frame_name: str
    frame_index: int
    object_index: int
    current_label: str
    decision_path: str
    matched_entity_id: str | None
    confidence: float | None
    candidate_entity_ids: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DecisionLogger(Protocol):
    def log(self, record: DecisionLogRecord) -> None:
        ...


class NullDecisionLogger:
    def log(self, record: DecisionLogRecord) -> None:
        return None


class JSONLDecisionLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: DecisionLogRecord) -> None:
        payload = asdict(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
