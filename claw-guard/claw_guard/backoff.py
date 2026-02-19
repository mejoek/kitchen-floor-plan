"""Exponential backoff with full jitter for 429 and transient error handling.

Section 2.10 of the architecture doc.

Strategy:
  1. Parse ``x-ratelimit-reset-tokens`` and ``x-ratelimit-reset-requests``
     headers from Gemini 429 responses.
  2. If headers present, wait for specified duration + random jitter.
  3. If headers missing, use exponential backoff with full jitter:
       sleep = random(0, base * 2^attempt)
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, Optional

from .config import (
    BACKOFF_BASE_DELAY,
    BACKOFF_MULTIPLIER,
    BACKOFF_MAX_DELAY,
    BACKOFF_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def parse_rate_limit_headers(headers: Dict[str, str]) -> Optional[float]:
    """Extract wait time from Gemini rate-limit response headers.

    Looks for:
      - ``x-ratelimit-reset-tokens``  (e.g. "15s", "1m30s")
      - ``x-ratelimit-reset-requests`` (e.g. "10s")

    Returns the longer of the two durations in seconds, or None if
    neither header is present.

    TODO: Verify exact header names and value format from live Gemini 429
          responses during Phase 1 instrumentation.  The header names and
          value format below are based on common Google API patterns but
          may differ for Gemini specifically.
    """
    token_reset = _parse_duration(headers.get("x-ratelimit-reset-tokens", ""))
    request_reset = _parse_duration(headers.get("x-ratelimit-reset-requests", ""))

    if token_reset is None and request_reset is None:
        return None

    return max(token_reset or 0.0, request_reset or 0.0)


def _parse_duration(value: str) -> Optional[float]:
    """Parse a duration string like '15s', '1m30s', '500ms' into seconds.

    TODO: Verify exact format from Gemini API.  This handles common patterns.
    """
    if not value or not value.strip():
        return None

    value = value.strip().lower()
    total = 0.0
    current = ""

    for ch in value:
        if ch.isdigit() or ch == ".":
            current += ch
        elif ch == "m" and current:
            # Check if next char is 's' (for 'ms')
            # Simple approach: 'm' at end or followed by digits means minutes
            total += float(current) * 60.0
            current = ""
        elif ch == "s" and current:
            total += float(current)
            current = ""
        elif ch == "h" and current:
            total += float(current) * 3600.0
            current = ""

    # If there's leftover numeric, assume seconds
    if current:
        try:
            total += float(current)
        except ValueError:
            pass

    return total if total > 0 else None


def calculate_backoff_delay(attempt: int) -> float:
    """Calculate backoff delay with full jitter.

    Full jitter: sleep = random(0, base * multiplier^attempt)
    Capped at BACKOFF_MAX_DELAY.
    """
    max_sleep = min(
        BACKOFF_BASE_DELAY * (BACKOFF_MULTIPLIER ** attempt),
        BACKOFF_MAX_DELAY,
    )
    return random.uniform(0, max_sleep)


async def backoff_sleep(
    attempt: int,
    response_headers: Optional[Dict[str, str]] = None,
) -> float:
    """Sleep for the appropriate backoff duration.

    If rate-limit headers are available, uses those + jitter.
    Otherwise uses exponential backoff with full jitter.

    Returns the actual sleep duration in seconds.
    """
    header_wait = None
    if response_headers:
        header_wait = parse_rate_limit_headers(response_headers)

    if header_wait is not None:
        # Use header-specified wait + small jitter
        jitter = random.uniform(0.05, 0.2)
        sleep_time = header_wait + jitter
        logger.info(
            "Backoff (header-based): attempt=%d, header_wait=%.2fs, "
            "jitter=%.3fs, total=%.2fs",
            attempt, header_wait, jitter, sleep_time,
        )
    else:
        sleep_time = calculate_backoff_delay(attempt)
        logger.info(
            "Backoff (exponential): attempt=%d, sleep=%.2fs",
            attempt, sleep_time,
        )

    await asyncio.sleep(sleep_time)
    return sleep_time


def should_retry(attempt: int, status_code: int) -> bool:
    """Determine whether to retry based on attempt count and HTTP status.

    Retries on:
      - 429 (rate limit)
      - 500 (internal server error)
      - 502 (bad gateway)
      - 503 (service unavailable)
      - 504 (gateway timeout)
    """
    if attempt >= BACKOFF_MAX_RETRIES:
        return False
    return status_code in (429, 500, 502, 503, 504)
