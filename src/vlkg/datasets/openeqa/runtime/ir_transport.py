from __future__ import annotations
import json
import os
import re
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

OPENAI_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36'

_API_KEY_POOL_LOCK = threading.Lock()

_API_KEY_POOL_INDEX = 0

def _openeqa_post_payload(payload: dict[str, Any], *, api_key: str, config: Any) -> dict[str, Any]:
    """Use the OpenEQA-only transport without changing shared dataset code."""
    global _API_KEY_POOL_INDEX
    pool_var = str(config.api_key_env_var)
    if not pool_var.endswith('_KEYS'):
        pool_var = f'{pool_var}S'
    raw_pool = os.getenv(pool_var, '')
    pool: list[str] = []
    if raw_pool:
        try:
            parsed_pool = json.loads(raw_pool)
        except (TypeError, ValueError):
            parsed_pool = None
        if isinstance(parsed_pool, list):
            pool.extend((str(item).strip() for item in parsed_pool))
        else:
            pool.extend((part.strip() for part in re.split('[,;\\r\\n]+', raw_pool)))
    pool = list(dict.fromkeys((item for item in pool if item)))
    if not pool:
        pool = [api_key] if api_key else []
    if pool:
        with _API_KEY_POOL_LOCK:
            api_key = pool[_API_KEY_POOL_INDEX % len(pool)]
            _API_KEY_POOL_INDEX += 1
    endpoint = urljoin(config.api_base_url.rstrip('/') + '/', config.endpoint_path.lstrip('/'))
    last_error: Exception | None = None
    for attempt in range(4):
        request = Request(endpoint, data=json.dumps(payload).encode('utf-8'), headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json', 'User-Agent': OPENAI_BROWSER_USER_AGENT}, method='POST')
        try:
            with urlopen(request, timeout=config.timeout_sec) as response:
                raw_response = json.loads(response.read().decode('utf-8'))
            if not isinstance(raw_response, dict):
                raise json.JSONDecodeError('Response JSON is not an object', str(raw_response), 0)
            return raw_response
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 502, 503, 504} or attempt >= 3:
                raise
            exc.close()
        except (TimeoutError, URLError, OSError) as exc:
            last_error = exc
            if attempt >= 3:
                raise
        time.sleep(min(5 * 2 ** attempt, 30))
    raise RuntimeError(f'OpenEQA IR request failed after retries: {last_error}')
