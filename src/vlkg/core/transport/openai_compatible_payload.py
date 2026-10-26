from __future__ import annotations

from typing import Any


def apply_reasoning_options(
    payload: dict[str, Any],
    *,
    reasoning_effort: str | None,
    thinking_enabled: bool | None,
) -> None:
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    if thinking_enabled is None:
        return
    payload["thinking"] = {
        "type": "enabled" if thinking_enabled else "disabled",
    }
