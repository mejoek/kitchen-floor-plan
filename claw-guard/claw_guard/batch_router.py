"""Batch API routing for sub-agent traffic.

All agents except Xeon submit work through the Gemini Batch API.
Requests are micro-batched: accumulated for up to 5 minutes or 100 requests,
whichever comes first, then submitted as a single batch job.

Section 2.7 of the architecture doc.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import (
    BATCH_ACCUMULATION_SECONDS,
    BATCH_ACCUMULATION_MAX_REQUESTS,
    GEMINI_BASE_URL,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingBatchRequest:
    """A single request waiting to be included in a batch submission."""

    request_id: str
    agent_id: str
    model_id: str
    body: Dict[str, Any]
    enqueued_at: float = field(default_factory=time.monotonic)
    result_future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


@dataclass
class BatchJob:
    """Tracks a submitted batch job."""

    job_id: str
    batch_id: str  # our internal batch ID
    model_id: str
    request_count: int
    submitted_at: float = field(default_factory=time.monotonic)
    status: str = "pending"  # pending, running, completed, failed
    results: Optional[List[Dict[str, Any]]] = None


class BatchRouter:
    """Accumulates sub-agent requests and submits them to the Gemini Batch API."""

    def __init__(self, gemini_base_url: str = GEMINI_BASE_URL, api_key: str = "") -> None:
        self._base_url = gemini_base_url
        self._api_key = api_key
        self._accumulator: List[PendingBatchRequest] = []
        self._accumulator_start: Optional[float] = None
        self._jobs: Dict[str, BatchJob] = {}
        self._flush_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background flush and poll tasks."""
        self._flush_task = asyncio.create_task(self._flush_loop())
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        for task in [self._flush_task, self._poll_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # Fail all pending futures
        for req in self._accumulator:
            if not req.result_future.done():
                req.result_future.set_exception(
                    RuntimeError("BatchRouter shutting down")
                )

    async def submit(
        self,
        agent_id: str,
        model_id: str,
        body: Dict[str, Any],
    ) -> asyncio.Future:
        """Accept a sub-agent request for batch processing.

        Returns a Future that resolves when the batch result is available.
        """
        request_id = uuid.uuid4().hex[:12]
        pending = PendingBatchRequest(
            request_id=request_id,
            agent_id=agent_id,
            model_id=model_id,
            body=body,
        )
        self._accumulator.append(pending)
        if self._accumulator_start is None:
            self._accumulator_start = time.monotonic()

        logger.debug(
            "Batch request %s from agent %s queued (accumulator size: %d)",
            request_id, agent_id, len(self._accumulator),
        )

        # Check if we should flush immediately (count threshold)
        if len(self._accumulator) >= BATCH_ACCUMULATION_MAX_REQUESTS:
            await self._flush()

        return pending.result_future

    async def _flush_loop(self) -> None:
        """Periodically flush the accumulator based on time threshold."""
        while True:
            await asyncio.sleep(10.0)  # check every 10 seconds
            if self._accumulator and self._accumulator_start is not None:
                elapsed = time.monotonic() - self._accumulator_start
                if elapsed >= BATCH_ACCUMULATION_SECONDS:
                    await self._flush()

    async def _flush(self) -> None:
        """Submit accumulated requests as a batch job."""
        if not self._accumulator:
            return

        batch = self._accumulator[:]
        self._accumulator.clear()
        self._accumulator_start = None

        batch_id = uuid.uuid4().hex[:12]

        # Group by model for separate batch submissions
        by_model: Dict[str, List[PendingBatchRequest]] = {}
        for req in batch:
            by_model.setdefault(req.model_id, []).append(req)

        for model_id, requests in by_model.items():
            await self._submit_batch(batch_id, model_id, requests)

    async def _submit_batch(
        self,
        batch_id: str,
        model_id: str,
        requests: List[PendingBatchRequest],
    ) -> None:
        """Submit a batch of requests to the Gemini Batch API.

        TODO: Implement actual Batch API integration.

        The Gemini Batch API workflow:
        1. Create a JSONL file with individual requests
        2. Upload to GCS or pass inline
        3. Call batchPredict / createBatchJob endpoint
        4. Poll for completion
        5. Parse results and resolve futures

        PLACEHOLDER: The exact API endpoint and request format depend on
        the Gemini Batch API, which uses:
          POST /v1/models/{model}:batchGenerateContent
        or the newer batch prediction job API.

        For now, we log the submission and fail the futures with a
        "not yet implemented" error so callers get a clear signal.
        """
        logger.info(
            "BATCH SUBMIT [%s]: model=%s, requests=%d",
            batch_id, model_id, len(requests),
        )

        job = BatchJob(
            job_id=f"placeholder-{batch_id}-{model_id}",
            batch_id=batch_id,
            model_id=model_id,
            request_count=len(requests),
        )
        self._jobs[job.job_id] = job

        # --- PLACEHOLDER ---
        # In production, this would:
        # 1. Format requests as JSONL:
        #    {"request": {"model": "models/{model_id}", "body": {...}}}
        # 2. Submit via HTTP:
        #    POST {base_url}/models/{model_id}:batchGenerateContent
        #    or create a batch job with input/output GCS URIs
        # 3. Store job reference for polling
        #
        # For now, resolve futures with a placeholder response.
        for req in requests:
            if not req.result_future.done():
                req.result_future.set_result({
                    "_claw_guard_batch": True,
                    "_batch_id": batch_id,
                    "_status": "placeholder",
                    "_message": (
                        "Batch API integration not yet implemented. "
                        "Request was queued for batch processing."
                    ),
                    "request_id": req.request_id,
                    "agent_id": req.agent_id,
                    "model_id": model_id,
                })
        # --- END PLACEHOLDER ---

    async def _poll_loop(self) -> None:
        """Poll for completed batch jobs and resolve pending futures.

        TODO: Implement actual polling against the Gemini Batch API.

        The poll loop would:
        1. List active batch jobs
        2. Check status via GET /v1/batchJobs/{jobId}
        3. On completion, download results
        4. Match results to pending futures and resolve them
        """
        while True:
            await asyncio.sleep(30.0)
            # Placeholder: just log active jobs
            active = [j for j in self._jobs.values() if j.status in ("pending", "running")]
            if active:
                logger.debug("Active batch jobs: %d", len(active))

    def snapshot(self) -> dict:
        return {
            "accumulator_size": len(self._accumulator),
            "accumulator_age_seconds": (
                round(time.monotonic() - self._accumulator_start, 1)
                if self._accumulator_start else None
            ),
            "active_jobs": sum(
                1 for j in self._jobs.values()
                if j.status in ("pending", "running")
            ),
            "total_jobs": len(self._jobs),
        }
