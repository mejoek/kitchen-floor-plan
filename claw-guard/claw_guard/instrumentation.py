"""Instrumentation, logging, and metrics for Claw Guard.

Section 2.1 of the architecture doc.

Captures:
  - Response header logging (x-ratelimit-*, x-goog-*)
  - Token ratio calibration (actual tokens vs char count)
  - Cache hit logging (cached_content_token_count)
  - Reservation delta logging (reserved vs actual)
  - Agent identification
  - Shadow mode (would-throttle / would-fallback events)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Metrics captured for a single request/response cycle."""

    timestamp: float = field(default_factory=time.time)
    request_id: str = ""
    agent_id: str = ""

    # Model routing
    model_requested: str = ""
    model_served: str = ""
    fallback_depth: int = 0

    # Token accounting
    estimated_prompt_tokens: int = 0
    max_output_tokens: int = 0
    total_reserved: int = 0
    actual_prompt_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    actual_total_tokens: Optional[int] = None
    cached_content_token_count: Optional[int] = None
    credit_back: int = 0

    # Calibration
    prompt_char_count: int = 0
    actual_chars_per_token: Optional[float] = None

    # Rate limit headers from Gemini response
    rate_limit_headers: Dict[str, str] = field(default_factory=dict)

    # Bucket state at time of request
    bucket_tpm_available: Optional[float] = None
    bucket_rpm_available: Optional[float] = None

    # Shadow mode
    would_throttle: bool = False
    would_fallback: bool = False
    would_fallback_to: Optional[str] = None
    would_queue: bool = False

    # Timing
    upstream_latency_ms: Optional[float] = None

    # Errors
    status_code: Optional[int] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for k, v in self.__dict__.items():
            if v is not None and v != "" and v != {} and v != 0:
                d[k] = v
        return d


class MetricsCollector:
    """Collects and persists request metrics."""

    def __init__(self, metrics_file: str = "claw_guard_metrics.jsonl") -> None:
        self._metrics_file = Path(metrics_file)
        self._buffer: List[RequestMetrics] = []
        self._shadow_events: List[Dict[str, Any]] = []

        # Counters
        self.total_requests = 0
        self.total_throttled = 0
        self.total_fallbacks = 0
        self.total_queued = 0
        self.total_429s = 0
        self.total_errors = 0

    def record(self, metrics: RequestMetrics) -> None:
        """Record a completed request's metrics."""
        self.total_requests += 1

        if metrics.would_throttle:
            self.total_throttled += 1
        if metrics.would_fallback:
            self.total_fallbacks += 1
        if metrics.would_queue:
            self.total_queued += 1
        if metrics.status_code == 429:
            self.total_429s += 1
        if metrics.error:
            self.total_errors += 1

        self._buffer.append(metrics)

        # Write to file
        self._flush_one(metrics)

        # Log notable events
        self._log_notable(metrics)

    def record_shadow_event(
        self,
        event_type: str,
        request_id: str,
        details: Dict[str, Any],
    ) -> None:
        """Record a shadow-mode event (would-throttle, would-fallback, etc.)."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "request_id": request_id,
            **details,
        }
        self._shadow_events.append(event)
        logger.warning(
            "SHADOW EVENT [%s]: request=%s %s",
            event_type, request_id, json.dumps(details),
        )

        # Also write to metrics file
        try:
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.error("Failed to write shadow event: %s", e)

    def extract_rate_limit_headers(
        self, headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Extract x-ratelimit-* and x-goog-* headers from a response.

        TODO: Verify exact header names from live Gemini responses.
              Common patterns:
              - x-ratelimit-limit-requests
              - x-ratelimit-limit-tokens
              - x-ratelimit-remaining-requests
              - x-ratelimit-remaining-tokens
              - x-ratelimit-reset-requests
              - x-ratelimit-reset-tokens
              - x-goog-api-response-latency
        """
        relevant = {}
        for key, value in headers.items():
            lower_key = key.lower()
            if lower_key.startswith("x-ratelimit-") or lower_key.startswith("x-goog-"):
                relevant[lower_key] = value
        return relevant

    def extract_usage_metadata(
        self, response_body: Dict[str, Any]
    ) -> Dict[str, Optional[int]]:
        """Extract token usage from Gemini response usageMetadata.

        TODO: Verify exact field structure from live Gemini responses.
              Expected structure:
              {
                "usageMetadata": {
                  "promptTokenCount": 100,
                  "candidatesTokenCount": 50,
                  "totalTokenCount": 150,
                  "cachedContentTokenCount": 80
                }
              }
        """
        usage = response_body.get("usageMetadata", {})
        return {
            "prompt_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
            "cached_tokens": usage.get("cachedContentTokenCount"),
        }

    def calibrate_token_ratio(
        self,
        char_count: int,
        actual_tokens: Optional[int],
    ) -> Optional[float]:
        """Calculate actual chars-per-token ratio for calibration.

        Used to validate/refine the 4.0 chars/token heuristic.
        """
        if actual_tokens and actual_tokens > 0 and char_count > 0:
            ratio = char_count / actual_tokens
            logger.debug(
                "Token ratio calibration: %d chars / %d tokens = %.2f chars/token",
                char_count, actual_tokens, ratio,
            )
            return ratio
        return None

    def _flush_one(self, metrics: RequestMetrics) -> None:
        """Append a single metrics record to the JSONL file."""
        try:
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")
        except OSError as e:
            logger.error("Failed to write metrics: %s", e)

    def _log_notable(self, metrics: RequestMetrics) -> None:
        """Log notable events at appropriate levels."""
        if metrics.would_throttle:
            logger.warning(
                "Would-Throttle: request=%s model=%s",
                metrics.request_id, metrics.model_served,
            )
        if metrics.would_fallback:
            logger.warning(
                "Would-Fallback: request=%s from=%s to=%s",
                metrics.request_id, metrics.model_requested,
                metrics.would_fallback_to,
            )
        if metrics.status_code == 429:
            logger.error(
                "Upstream 429: request=%s model=%s headers=%s",
                metrics.request_id, metrics.model_served,
                json.dumps(metrics.rate_limit_headers),
            )
        if metrics.cached_content_token_count and metrics.cached_content_token_count > 0:
            logger.info(
                "Cache hit: request=%s cached_tokens=%d model=%s",
                metrics.request_id, metrics.cached_content_token_count,
                metrics.model_served,
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "total_throttled": self.total_throttled,
            "total_fallbacks": self.total_fallbacks,
            "total_queued": self.total_queued,
            "total_429s": self.total_429s,
            "total_errors": self.total_errors,
            "shadow_events": len(self._shadow_events),
            "recent_shadow_events": [
                e for e in self._shadow_events[-10:]  # last 10
            ],
        }
