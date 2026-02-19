"""Reservation accounting — reserve tokens on send, credit-back on response.

Tracks in-flight reservations and automatically expires them after 90 seconds
(Section 2.9 leak protection).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from .config import RESERVATION_EXPIRY_SECONDS, CHARS_PER_TOKEN
from .token_bucket import BucketRegistry

logger = logging.getLogger(__name__)


@dataclass
class Reservation:
    """A single in-flight token reservation."""

    reservation_id: str
    model_id: str
    estimated_prompt_tokens: int
    max_output_tokens: int
    total_reserved: int  # estimated_prompt_tokens + max_output_tokens
    created_at: float = field(default_factory=time.monotonic)
    settled: bool = False  # True once settle() or release() has been called

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def expired(self) -> bool:
        return self.age_seconds > RESERVATION_EXPIRY_SECONDS


class ReservationLedger:
    """Manages in-flight token reservations with automatic expiry."""

    def __init__(self, bucket_registry: BucketRegistry) -> None:
        self._buckets = bucket_registry
        self._reservations: Dict[str, Reservation] = {}
        self._expiry_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the background expiry sweeper."""
        self._expiry_task = asyncio.create_task(self._sweep_expired())

    async def stop(self) -> None:
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass

    def estimate_prompt_tokens(self, prompt_text: str) -> int:
        """Estimate token count from raw character count using the 4.0 chars/token heuristic.

        TODO: Replace with a proper tokenizer or refine heuristic based on
              Phase 1 calibration data (Section 2.1 ratio calibration).
        """
        return max(1, int(len(prompt_text) / CHARS_PER_TOKEN))

    def create_reservation(
        self,
        model_id: str,
        estimated_prompt_tokens: int,
        max_output_tokens: int,
    ) -> Optional[Reservation]:
        """Create a reservation and deduct from the model's buckets.

        Returns the Reservation on success, None if the bucket lacks capacity.
        """
        total = estimated_prompt_tokens + max_output_tokens
        pair = self._buckets.get(model_id)
        if pair is None:
            logger.warning("No bucket pair for model %s", model_id)
            return None

        if not pair.reserve(float(total)):
            return None

        reservation_id = uuid.uuid4().hex[:12]
        r = Reservation(
            reservation_id=reservation_id,
            model_id=model_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
            max_output_tokens=max_output_tokens,
            total_reserved=total,
        )
        self._reservations[reservation_id] = r
        logger.debug(
            "Reservation %s: model=%s reserved=%d (prompt=%d + output=%d)",
            reservation_id, model_id, total,
            estimated_prompt_tokens, max_output_tokens,
        )
        return r

    def settle(self, reservation_id: str, actual_total_tokens: int) -> int:
        """Settle a reservation after receiving the response.

        Returns the number of tokens credited back (>= 0).
        Guards against double-settlement (e.g. if the sweeper already released).
        """
        r = self._reservations.pop(reservation_id, None)
        if r is None:
            logger.warning("Settlement for unknown reservation %s (likely already expired)", reservation_id)
            return 0
        if r.settled:
            logger.warning("Reservation %s already settled — ignoring duplicate", reservation_id)
            return 0
        r.settled = True

        credit = max(0, r.total_reserved - actual_total_tokens)
        if credit > 0:
            pair = self._buckets.get(r.model_id)
            if pair:
                pair.credit_back(float(credit))
                # Notify queue that capacity is available
                self._buckets.notify_capacity(r.model_id)
                logger.debug(
                    "Reservation %s settled: reserved=%d actual=%d credit=%d",
                    reservation_id, r.total_reserved, actual_total_tokens, credit,
                )
        return credit

    def release(self, reservation_id: str) -> None:
        """Release a reservation entirely (e.g. on timeout / error).

        Guards against double-release via the settled flag.
        """
        r = self._reservations.pop(reservation_id, None)
        if r is None:
            return
        if r.settled:
            logger.debug("Reservation %s already settled — skipping release", reservation_id)
            return
        r.settled = True

        pair = self._buckets.get(r.model_id)
        if pair:
            pair.credit_back(float(r.total_reserved))
            # Notify queue that capacity is available
            self._buckets.notify_capacity(r.model_id)
            logger.info(
                "Reservation %s released (expired/error): returned %d tokens to %s",
                reservation_id, r.total_reserved, r.model_id,
            )

    @property
    def in_flight_count(self) -> int:
        return len(self._reservations)

    def snapshot(self) -> Dict[str, dict]:
        return {
            rid: {
                "model_id": r.model_id,
                "total_reserved": r.total_reserved,
                "age_seconds": round(r.age_seconds, 1),
            }
            for rid, r in self._reservations.items()
        }

    async def _sweep_expired(self) -> None:
        """Periodically release expired reservations (leak protection)."""
        while True:
            await asyncio.sleep(10.0)  # check every 10 s
            expired_ids = [
                rid for rid, r in self._reservations.items()
                if r.expired and not r.settled
            ]
            for rid in expired_ids:
                logger.warning("Reservation %s expired after %.0f s — releasing",
                               rid, RESERVATION_EXPIRY_SECONDS)
                self.release(rid)
