"""Queue management for Xeon requests when the entire fallback chain is exhausted.

Section 2.8 of the architecture doc.

- Max 50 pending requests
- 30-second timeout per queued request
- FIFO ordering
- Overflow immediately rejected with 429
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import QUEUE_MAX_DEPTH, QUEUE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


@dataclass
class QueuedRequest:
    """A request waiting for bucket capacity."""

    request_id: str
    estimated_tokens: float
    enqueued_at: float = field(default_factory=time.monotonic)
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    assigned_model: Optional[str] = None  # set when capacity becomes available
    timed_out: bool = False

    @property
    def wait_seconds(self) -> float:
        return time.monotonic() - self.enqueued_at


class RequestQueue:
    """FIFO queue for Xeon requests that cannot be served immediately."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(
            maxsize=QUEUE_MAX_DEPTH
        )
        self._pending: list[QueuedRequest] = []
        self._drain_task: Optional[asyncio.Task] = None

    @property
    def depth(self) -> int:
        return len(self._pending)

    @property
    def is_full(self) -> bool:
        return self.depth >= QUEUE_MAX_DEPTH

    async def start(self) -> None:
        """Start the background drain loop."""
        self._drain_task = asyncio.create_task(self._drain_loop())

    async def stop(self) -> None:
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
        # Wake all waiters so they can exit
        for qr in self._pending:
            qr.timed_out = True
            qr.ready_event.set()

    async def enqueue(self, request_id: str, estimated_tokens: float) -> QueuedRequest:
        """Add a request to the queue.

        Raises ``QueueFullError`` if the queue is at capacity.
        """
        if self.is_full:
            raise QueueFullError(
                depth=self.depth,
                estimated_wait=self._estimated_wait(),
            )

        qr = QueuedRequest(
            request_id=request_id,
            estimated_tokens=estimated_tokens,
        )
        self._pending.append(qr)
        logger.info(
            "Queued request %s (depth=%d, est_tokens=%.0f)",
            request_id, self.depth, estimated_tokens,
        )
        return qr

    async def wait_for_capacity(self, qr: QueuedRequest) -> bool:
        """Block until capacity is available or timeout.

        Returns True if a model was assigned, False on timeout.
        The caller checks ``qr.assigned_model`` for the selected model.
        """
        try:
            await asyncio.wait_for(
                qr.ready_event.wait(),
                timeout=QUEUE_TIMEOUT_SECONDS,
            )
            return qr.assigned_model is not None
        except asyncio.TimeoutError:
            qr.timed_out = True
            self._remove(qr)
            logger.warning(
                "Queued request %s timed out after %.1f s",
                qr.request_id, qr.wait_seconds,
            )
            return False

    def notify_capacity(self, model_id: str, available_tokens: float) -> Optional[QueuedRequest]:
        """Called when bucket capacity becomes available.

        Assigns the first eligible pending request and wakes it up.
        Returns the assigned request or None.
        """
        for qr in self._pending:
            if qr.timed_out or qr.ready_event.is_set():
                continue
            if qr.estimated_tokens <= available_tokens:
                qr.assigned_model = model_id
                qr.ready_event.set()
                self._remove(qr)
                logger.info(
                    "Dequeued request %s → model %s (waited %.1f s)",
                    qr.request_id, model_id, qr.wait_seconds,
                )
                return qr
        return None

    def _remove(self, qr: QueuedRequest) -> None:
        try:
            self._pending.remove(qr)
        except ValueError:
            pass

    def _estimated_wait(self) -> float:
        """Rough estimate of wait time for a new request at the back of the queue."""
        if not self._pending:
            return 0.0
        oldest = self._pending[0]
        remaining = max(0.0, QUEUE_TIMEOUT_SECONDS - oldest.wait_seconds)
        return remaining

    async def _drain_loop(self) -> None:
        """Periodically clean up timed-out entries.

        The main timeout is handled in ``wait_for_capacity``, but this
        sweeps entries that were abandoned without awaiting.
        """
        while True:
            await asyncio.sleep(5.0)
            now = time.monotonic()
            expired = [
                qr for qr in self._pending
                if (now - qr.enqueued_at) > QUEUE_TIMEOUT_SECONDS
                and not qr.ready_event.is_set()
            ]
            for qr in expired:
                qr.timed_out = True
                qr.ready_event.set()
                self._remove(qr)

    def snapshot(self) -> dict:
        return {
            "depth": self.depth,
            "is_full": self.is_full,
            "pending": [
                {
                    "request_id": qr.request_id,
                    "wait_seconds": round(qr.wait_seconds, 1),
                    "estimated_tokens": qr.estimated_tokens,
                }
                for qr in self._pending
            ],
        }


class QueueFullError(Exception):
    """Raised when the queue is at max capacity."""

    def __init__(self, depth: int, estimated_wait: float) -> None:
        self.depth = depth
        self.estimated_wait = estimated_wait
        super().__init__(
            f"Queue full (depth={depth}), estimated wait {estimated_wait:.1f}s"
        )
