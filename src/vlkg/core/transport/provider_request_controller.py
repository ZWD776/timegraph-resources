from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar

import openai


T = TypeVar("T")


class ProviderRequestController:
    """Serialize provider requests and back off on rate limits."""

    def __init__(
        self,
        *,
        min_interval_seconds: float = 2.0,
        retry_seconds: float = 30.0,
        max_retries: int = 10,
    ):
        self.min_interval_seconds = max(0.0, min_interval_seconds)
        self.retry_seconds = max(1.0, retry_seconds)
        self.max_retries = max(0, max_retries)
        self._lock = asyncio.Lock()
        self._last_request_finished_at = 0.0

    async def run(self, operation: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            attempts = 0
            while True:
                delay = self.min_interval_seconds - (
                    time.monotonic() - self._last_request_finished_at
                )
                if delay > 0:
                    await asyncio.sleep(delay)

                try:
                    result = await operation()
                    self._last_request_finished_at = time.monotonic()
                    return result
                except openai.RateLimitError:
                    self._last_request_finished_at = time.monotonic()
                    attempts += 1
                    if attempts > self.max_retries:
                        raise
                    await asyncio.sleep(self.retry_seconds)
