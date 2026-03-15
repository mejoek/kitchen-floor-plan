"""Isolated token-bucket rate limiter for TPM and RPM per model.

Each model in the fallback chain gets its own independent bucket pair.
Buckets refill at a constant rate (effective_limit / 60 per second).

Section 2.9 of the architecture doc.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import MODEL_LIMITS, FALLBACK_CHAIN, ModelLimits


@dataclass
class Bucket:
    """A single token bucket with linear refill."""

    capacity: float        # max tokens the bucket can hold
    refill_per_second: float  # tokens added per second
    tokens: float = 0.0    # current level
    last_refill: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        # Start full
        self.tokens = self.capacity

    def _refill(self) -> None:
        """Add tokens accrued since last refill."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
            self.last_refill = now

    def try_consume(self, amount: float) -> bool:
        """Attempt to consume *amount* tokens.  Returns True on success."""
        with self._lock:
            self._refill()
            if self.tokens >= amount:
                self.tokens -= amount
                return True
            return False

    def credit(self, amount: float) -> None:
        """Return unused tokens (credit-back after response)."""
        with self._lock:
            self._refill()
            self.tokens = min(self.capacity, self.tokens + amount)

    def force_consume(self, amount: float) -> None:
        """Consume tokens without checking — used when we've already verified capacity."""
        with self._lock:
            self._refill()
            self.tokens = max(0.0, self.tokens - amount)

    @property
    def available(self) -> float:
        """Current available tokens (non-locking snapshot, may be slightly stale)."""
        with self._lock:
            self._refill()
            return self.tokens

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently consumed (0.0 = empty, 1.0 = full bucket)."""
        with self._lock:
            self._refill()
            return self.tokens / self.capacity if self.capacity > 0 else 0.0


@dataclass
class ModelBucketPair:
    """TPM and RPM buckets for a single model."""

    model_id: str
    tpm_bucket: Bucket
    rpm_bucket: Bucket

    def has_capacity(self, estimated_tokens: float) -> bool:
        """Check if both TPM and RPM have room for the request.

        Does NOT consume — call :meth:`reserve` to actually deduct.
        """
        return (
            self.tpm_bucket.available >= estimated_tokens
            and self.rpm_bucket.available >= 1.0
        )

    def reserve(self, estimated_tokens: float) -> bool:
        """Atomically reserve TPM tokens + 1 RPM slot.

        Returns False if either bucket lacks capacity.
        """
        # We check-and-consume in sequence.  A race is possible between the
        # two buckets but acceptable for a single-host proxy.
        if not self.tpm_bucket.try_consume(estimated_tokens):
            return False
        if not self.rpm_bucket.try_consume(1.0):
            # Roll back TPM
            self.tpm_bucket.credit(estimated_tokens)
            return False
        return True

    def credit_back(self, unused_tokens: float) -> None:
        """Return over-reserved TPM tokens after actual usage is known."""
        if unused_tokens > 0:
            self.tpm_bucket.credit(unused_tokens)

    @property
    def tpm_utilization(self) -> float:
        return self.tpm_bucket.utilization

    @property
    def rpm_utilization(self) -> float:
        return self.rpm_bucket.utilization


class BucketRegistry:
    """Registry of per-model bucket pairs for the entire fallback chain."""

    def __init__(self) -> None:
        self._buckets: Dict[str, ModelBucketPair] = {}
        self._init_buckets()

    def _init_buckets(self) -> None:
        for model_id in FALLBACK_CHAIN:
            limits = MODEL_LIMITS.get(model_id)
            if limits is None:
                continue
            tpm = Bucket(
                capacity=float(limits.effective_tpm),
                refill_per_second=limits.effective_tpm / 60.0,
            )
            rpm = Bucket(
                capacity=float(limits.effective_rpm),
                refill_per_second=limits.effective_rpm / 60.0,
            )
            self._buckets[model_id] = ModelBucketPair(
                model_id=model_id,
                tpm_bucket=tpm,
                rpm_bucket=rpm,
            )

    def get(self, model_id: str) -> Optional[ModelBucketPair]:
        return self._buckets.get(model_id)

    def all_pairs(self) -> Dict[str, ModelBucketPair]:
        return dict(self._buckets)

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a JSON-serialisable snapshot of all bucket levels."""
        return {
            mid: {
                "tpm_available": pair.tpm_bucket.available,
                "tpm_capacity": pair.tpm_bucket.capacity,
                "tpm_utilization": pair.tpm_utilization,
                "rpm_available": pair.rpm_bucket.available,
                "rpm_capacity": pair.rpm_bucket.capacity,
                "rpm_utilization": pair.rpm_utilization,
            }
            for mid, pair in self._buckets.items()
        }
