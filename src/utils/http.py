"""HTTP with retry/backoff. Open-Meteo returns 429 under load routinely."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

log = logging.getLogger(__name__)

RETRY_STATUS = {429, 500, 502, 503, 504}


class NonRetryableHTTPError(RuntimeError):
    """A 4xx that will fail identically on every retry."""


def get_json(url: str, params: dict[str, Any], *, timeout: int = 180,
             retries: int = 5, backoff: float = 2.0) -> Any:
    """GET returning parsed JSON, retrying on rate limits and transient 5xx.

    Note for the Airflow port: inside a DAG task this sleeping loop should be
    replaced by task-level `retries` + `retry_exponential_backoff` (spec gotcha
    15.22). It lives here so `src/` stays runnable outside Airflow.
    """
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}", response=r)
            if 400 <= r.status_code < 500:
                # Deterministic; retrying only wastes the API budget.
                raise NonRetryableHTTPError(f"GET {url} -> {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            return r.json()
        except NonRetryableHTTPError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last = exc
            if attempt == retries:
                break
            delay = backoff ** attempt + random.uniform(0, 1)
            log.warning("GET %s failed (%s); retry %d/%d in %.1fs",
                        url, exc, attempt + 1, retries, delay)
            time.sleep(delay)
    raise RuntimeError(f"GET {url} failed after {retries} retries") from last


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "<REDACTED>")
    return text


def get_text(url: str, *, params: dict[str, Any] | None = None, timeout: int = 180,
             retries: int = 4, backoff: float = 2.0,
             secrets: tuple[str, ...] = ()) -> str:
    """GET returning the raw body.

    `secrets` are scrubbed from every log line and exception message. FIRMS puts
    the MAP_KEY in the URL *path*, so without this an ordinary 400 writes the
    key into logs, tracebacks and notebook output -- and from there into commits.
    """
    safe_url = _redact(url, secrets)
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{r.status_code}")
            if 400 <= r.status_code < 500:
                # Client errors are deterministic; retrying just burns the quota.
                raise NonRetryableHTTPError(
                    f"GET {safe_url} -> {r.status_code}: "
                    f"{_redact(r.text[:300], secrets)}")
            r.raise_for_status()
            return r.text
        except NonRetryableHTTPError:
            raise
        except requests.RequestException as exc:
            last = exc
            if attempt == retries:
                break
            log.warning("GET %s failed (%s); retry %d/%d",
                        safe_url, _redact(str(exc), secrets), attempt + 1, retries)
            time.sleep(backoff ** attempt + random.uniform(0, 1))
    raise RuntimeError(f"GET {safe_url} failed after {retries} retries") from last
