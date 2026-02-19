"""Sticky model fallback chain with recovery and anti-flap guard.

Section 2.4 of the architecture doc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    FALLBACK_CHAIN,
    RECOVERY_THRESHOLD_FRACTION,
    ANTI_FLAP_SECONDS,
    MODEL_LIMITS,
)
from .token_bucket import BucketRegistry

logger = logging.getLogger(__name__)


@dataclass
class FallbackState:
    """Tracks current lock position, promotion history, and anti-flap timing."""

    # Index into FALLBACK_CHAIN (0 = primary)
    locked_index: int = 0
    last_promotion_time: float = 0.0  # monotonic timestamp of last climb-back

    @property
    def current_model(self) -> str:
        return FALLBACK_CHAIN[self.locked_index]

    @property
    def fallback_depth(self) -> int:
        return self.locked_index

    @property
    def is_on_primary(self) -> bool:
        return self.locked_index == 0

    @property
    def in_anti_flap_window(self) -> bool:
        if self.last_promotion_time == 0.0:
            return False
        return (time.monotonic() - self.last_promotion_time) < ANTI_FLAP_SECONDS


class FallbackManager:
    """Manages the sticky fallback chain for Xeon's real-time traffic."""

    def __init__(self, bucket_registry: BucketRegistry) -> None:
        self._buckets = bucket_registry
        self._state = FallbackState()

    @property
    def state(self) -> FallbackState:
        return self._state

    @property
    def current_model(self) -> str:
        return self._state.current_model

    @property
    def fallback_depth(self) -> int:
        return self._state.fallback_depth

    def check_recovery(self) -> bool:
        """Check if the primary model has recovered enough to promote back.

        Called on every incoming Xeon request while in fallback.
        Returns True if promotion happened.
        """
        if self._state.is_on_primary:
            return False  # already on primary

        primary_model = FALLBACK_CHAIN[0]
        primary_limits = MODEL_LIMITS.get(primary_model)
        if primary_limits is None:
            return False

        pair = self._buckets.get(primary_model)
        if pair is None:
            return False

        threshold_tpm = primary_limits.effective_tpm * RECOVERY_THRESHOLD_FRACTION
        if (
            pair.tpm_bucket.available >= threshold_tpm
            and pair.rpm_bucket.available >= 1.0
        ):
            old_model = self._state.current_model
            self._state.locked_index = 0
            self._state.last_promotion_time = time.monotonic()
            logger.info(
                "RECOVERY: promoted back to primary %s from %s "
                "(tpm_available=%.0f >= threshold=%.0f)",
                primary_model, old_model,
                pair.tpm_bucket.available, threshold_tpm,
            )
            return True

        return False

    def select_model(self, estimated_tokens: float) -> Optional[str]:
        """Select the best available model for the request.

        Implements the sticky fallback logic:
        1. If in anti-flap window, use current locked model (or queue if no capacity).
        2. If in fallback, check recovery first.
        3. Check current model's capacity.
        4. If no capacity, walk down the chain and lock.
        5. If chain exhausted, return None (caller should queue).
        """
        # Step 1: Anti-flap guard
        if self._state.in_anti_flap_window:
            pair = self._buckets.get(self._state.current_model)
            if pair and pair.has_capacity(estimated_tokens):
                return self._state.current_model
            # In anti-flap window but no capacity → caller should queue
            logger.debug(
                "Anti-flap active, no capacity on %s — request should queue",
                self._state.current_model,
            )
            return None

        # Step 2: Recovery check (only when in fallback)
        if not self._state.is_on_primary:
            self.check_recovery()

        # Step 3: Check current locked model
        pair = self._buckets.get(self._state.current_model)
        if pair and pair.has_capacity(estimated_tokens):
            return self._state.current_model

        # Step 4: Walk the fallback chain from current position downward
        for i in range(self._state.locked_index + 1, len(FALLBACK_CHAIN)):
            candidate = FALLBACK_CHAIN[i]
            candidate_pair = self._buckets.get(candidate)
            if candidate_pair and candidate_pair.has_capacity(estimated_tokens):
                old_model = self._state.current_model
                self._state.locked_index = i
                logger.warning(
                    "FALLBACK: %s → %s (depth %d → %d)",
                    old_model, candidate,
                    i - 1 if i > 0 else 0, i,
                )
                return candidate

        # Step 5: Chain exhausted
        logger.error("Fallback chain fully exhausted — no model has capacity")
        return None

    def is_in_fallback(self) -> bool:
        """Whether Xeon is currently locked to a non-primary model."""
        return not self._state.is_on_primary

    def snapshot(self) -> dict:
        return {
            "current_model": self._state.current_model,
            "fallback_depth": self._state.fallback_depth,
            "is_on_primary": self._state.is_on_primary,
            "in_anti_flap_window": self._state.in_anti_flap_window,
            "last_promotion_age_seconds": (
                round(time.monotonic() - self._state.last_promotion_time, 1)
                if self._state.last_promotion_time > 0 else None
            ),
        }
