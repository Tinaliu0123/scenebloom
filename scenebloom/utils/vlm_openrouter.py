"""OpenRouter VLM calls shared by every VLM stage: image blocks, HTTP retries, JSON extraction."""

from __future__ import annotations

import base64
import logging
import os
import time
import re
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Default VLM.
DEFAULT_MODEL = "qwen/qwen3-vl-8b-instruct"
TIMEOUT_S = 120

__all__ = [
    "OPENROUTER_URL",
    "DEFAULT_MODEL",
    "image_block",
    "call_openrouter",
    "extract_first_json_object",
]


def image_block(path: Path) -> dict[str, Any] | None:
    """Base64 data-URL ``image_url`` block of an image; None (logged) if the file is missing."""
    path = Path(path)
    if not path.exists():
        log.warning("vlm_openrouter: image not found, skipping: %s", path)
        return None
    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else f"image/{ext or 'jpeg'}"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


#: Only transient failures are retried (429, 5xx, connection and read timeouts).
RETRY_STATUSES = {429, 500, 502, 503, 504}
RETRY_BACKOFF_S = (3.0, 8.0, 20.0, 45.0)
MAX_RETRIES = len(RETRY_BACKOFF_S)


def _post_with_retry(headers: dict[str, str], payload: dict[str, Any], timeout_s: int):
    """POST and return a 2xx response, retrying transient failures."""
    attempts = MAX_RETRIES + 1
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout_s)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            reason = f"{type(exc).__name__}"
        else:
            if resp.status_code not in RETRY_STATUSES:
                resp.raise_for_status()
                return resp
            reason = f"HTTP {resp.status_code}"
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                last_exc = exc
        if i + 1 >= attempts:
            break
        delay = RETRY_BACKOFF_S[min(i, len(RETRY_BACKOFF_S) - 1)]
        log.warning("vlm_openrouter: %s; retry %d/%d in %.0fs", reason, i + 1, attempts - 1, delay)
        time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def call_openrouter(
    messages: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> str:
    """POST ``messages`` to OpenRouter (needs ``OPENROUTER_API_KEY``) and return the reply text."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    log.info("vlm_openrouter: POST OpenAI-compatible endpoint model=%s", model)
    body = _post_with_retry(headers, payload, TIMEOUT_S).json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise ValueError(f"Unexpected OpenRouter response shape: {body}") from e


def extract_first_json_object(text: str) -> str:
    """First balanced ``{ ... }`` of ``text``, skipping code fences and braces in strings."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("unbalanced JSON braces in response")
