"""Configuration for Claw Guard proxy.

Hardcoded baseline limits from AI Studio console data.
Effective quotas include a 7.5% safety buffer.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Operating modes
# ---------------------------------------------------------------------------

class OperatingMode(enum.Enum):
    """Proxy operating modes — controls whether guardrails block or only log."""

    PASSTHROUGH = "passthrough"  # Phase 1: forward everything, log only
    SHADOW = "shadow"            # Phase 2: calculate throttle state, log would-block, never block
    ACTIVE = "active"            # Phase 3: hard-blocking, fallback, queue


# ---------------------------------------------------------------------------
# Model identifiers (as they appear in OpenClaw provider routing)
# ---------------------------------------------------------------------------

# Canonical model IDs used throughout Claw Guard.
# These MUST match whatever string OpenClaw puts in the request path or body.
# TODO: Verify exact model strings from live OpenClaw traffic during Phase 1.
MODEL_GEMINI_3_PRO = "gemini-3-pro"
MODEL_GEMINI_25_PRO = "gemini-2.5-pro"
MODEL_GEMINI_2_FLASH = "gemini-2-flash"
MODEL_GEMINI_3_FLASH = "gemini-3-flash"
MODEL_GEMINI_25_FLASH = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Per-model raw and effective limits
# ---------------------------------------------------------------------------

SAFETY_BUFFER = 0.075  # 7.5 %


@dataclass(frozen=True)
class ModelLimits:
    """Rate limits for a single Gemini model."""

    model_id: str
    raw_tpm: int           # tokens per minute (Google's published limit)
    raw_rpm: int           # requests per minute
    raw_rpd: Optional[int] # requests per day (None = unlimited)

    @property
    def effective_tpm(self) -> int:
        return int(self.raw_tpm * (1 - SAFETY_BUFFER))

    @property
    def effective_rpm(self) -> int:
        return int(self.raw_rpm * (1 - SAFETY_BUFFER))

    @property
    def effective_rpd(self) -> Optional[int]:
        if self.raw_rpd is None:
            return None
        return int(self.raw_rpd * (1 - SAFETY_BUFFER))


# Baseline limits from the architecture doc (Section 2.2)
MODEL_LIMITS: Dict[str, ModelLimits] = {
    MODEL_GEMINI_3_PRO: ModelLimits(
        model_id=MODEL_GEMINI_3_PRO,
        raw_tpm=1_000_000,
        raw_rpm=25,
        raw_rpd=250,
    ),
    MODEL_GEMINI_25_PRO: ModelLimits(
        model_id=MODEL_GEMINI_25_PRO,
        raw_tpm=2_000_000,
        raw_rpm=150,
        raw_rpd=1_000,
    ),
    MODEL_GEMINI_2_FLASH: ModelLimits(
        model_id=MODEL_GEMINI_2_FLASH,
        raw_tpm=4_000_000,
        raw_rpm=2_000,
        raw_rpd=None,
    ),
    MODEL_GEMINI_3_FLASH: ModelLimits(
        model_id=MODEL_GEMINI_3_FLASH,
        raw_tpm=1_000_000,
        raw_rpm=1_000,
        raw_rpd=10_000,
    ),
    MODEL_GEMINI_25_FLASH: ModelLimits(
        model_id=MODEL_GEMINI_25_FLASH,
        raw_tpm=1_000_000,
        raw_rpm=1_000,
        raw_rpd=10_000,
    ),
}


# ---------------------------------------------------------------------------
# Fallback chain (Section 2.4)
# ---------------------------------------------------------------------------

FALLBACK_CHAIN: List[str] = [
    MODEL_GEMINI_3_PRO,   # primary
    MODEL_GEMINI_25_PRO,  # fallback 1
    MODEL_GEMINI_2_FLASH, # fallback 2 (last resort)
]


# ---------------------------------------------------------------------------
# max_output_tokens defaults (Section 2.5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutputTokenDefaults:
    normal: int
    fallback: int


# "Pro" vs "Flash" classification
MAX_OUTPUT_TOKENS: Dict[str, OutputTokenDefaults] = {
    MODEL_GEMINI_3_PRO:   OutputTokenDefaults(normal=2048, fallback=1024),
    MODEL_GEMINI_25_PRO:  OutputTokenDefaults(normal=2048, fallback=1024),
    MODEL_GEMINI_2_FLASH: OutputTokenDefaults(normal=512,  fallback=256),
    MODEL_GEMINI_3_FLASH: OutputTokenDefaults(normal=512,  fallback=256),
    MODEL_GEMINI_25_FLASH: OutputTokenDefaults(normal=512, fallback=256),
}


# ---------------------------------------------------------------------------
# Recovery / anti-flap (Section 2.4)
# ---------------------------------------------------------------------------

RECOVERY_THRESHOLD_FRACTION = 0.25  # promote back when primary bucket >= 25 %
ANTI_FLAP_SECONDS = 30.0            # suppress fallback for 30 s after promotion


# ---------------------------------------------------------------------------
# Queue management (Section 2.8)
# ---------------------------------------------------------------------------

QUEUE_MAX_DEPTH = 50
QUEUE_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Reservation accounting (Section 2.9)
# ---------------------------------------------------------------------------

RESERVATION_EXPIRY_SECONDS = 90.0  # release reservation if no response in 90 s
CHARS_PER_TOKEN = 4.0              # heuristic for prompt estimation


# ---------------------------------------------------------------------------
# Batch API (Section 2.7)
# ---------------------------------------------------------------------------

BATCH_ACCUMULATION_SECONDS = 300   # 5 minutes
BATCH_ACCUMULATION_MAX_REQUESTS = 100


# ---------------------------------------------------------------------------
# Explicit caching (Section 2.6)
# ---------------------------------------------------------------------------

CACHE_DEFAULT_TTL_SECONDS = 3600       # 60 minutes
CACHE_STABLE_TTL_SECONDS = 21600       # 6 hours (system prompts, tool defs)

# Minimum cached tokens per model family
CACHE_MIN_TOKENS: Dict[str, int] = {
    MODEL_GEMINI_25_FLASH: 1024,
    MODEL_GEMINI_25_PRO:   2048,
    MODEL_GEMINI_2_FLASH:  4096,
    MODEL_GEMINI_3_PRO:    2048,   # TBD — placeholder, verify in Phase 1
    MODEL_GEMINI_3_FLASH:  1024,   # TBD — placeholder, verify in Phase 1
}


# ---------------------------------------------------------------------------
# Backoff (Section 2.10)
# ---------------------------------------------------------------------------

BACKOFF_BASE_DELAY = 1.0    # seconds
BACKOFF_MULTIPLIER = 2.0
BACKOFF_MAX_DELAY = 60.0    # seconds
BACKOFF_MAX_RETRIES = 6


# ---------------------------------------------------------------------------
# Load reduction (Section 2.4 — active load reduction)
# ---------------------------------------------------------------------------

CONTEXT_PRUNE_MAX_TURNS = 10  # trim to last N turns during fallback


# ---------------------------------------------------------------------------
# Proxy networking
# ---------------------------------------------------------------------------

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 18800
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1"

# Xeon agent identifier
# TODO: Confirm how OpenClaw tags the originating agent in request metadata.
#       Possible locations: custom header, query param, or body field.
XEON_AGENT_ID = "xeon"


# ---------------------------------------------------------------------------
# Aggregate config object
# ---------------------------------------------------------------------------

@dataclass
class ClawGuardConfig:
    """Top-level runtime configuration."""

    mode: OperatingMode = OperatingMode.PASSTHROUGH
    host: str = PROXY_HOST
    port: int = PROXY_PORT
    gemini_base_url: str = GEMINI_BASE_URL
    api_key: str = ""  # set at startup from env or arg
    log_file: str = "claw_guard.log"
    metrics_file: str = "claw_guard_metrics.jsonl"
    shadow_mode_hours: float = 72.0
