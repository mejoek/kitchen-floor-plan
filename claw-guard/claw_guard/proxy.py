"""Core HTTP proxy handler — the central orchestrator of Claw Guard.

Implements the complete request lifecycle from the architecture doc:
  1. Receive request from OpenClaw
  2. Identify agent (Xeon → real-time, others → batch)
  3. Apply fallback/recovery logic
  4. Enforce max_output_tokens and load reduction
  5. Check/create explicit caches
  6. Reserve tokens
  7. Forward to Gemini
  8. Tag response, credit-back, log metrics
  9. Handle 429s with backoff
  10. Handle timeouts with reservation release
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple

import aiohttp
from aiohttp import web

from .backoff import backoff_sleep, should_retry
from .batch_router import BatchRouter
from .cache_manager import CacheManager
from .config import (
    CHARS_PER_TOKEN,
    FALLBACK_CHAIN,
    RESERVATION_EXPIRY_SECONDS,
    XEON_AGENT_ID,
    ClawGuardConfig,
    OperatingMode,
)
from .fallback import FallbackManager
from .instrumentation import MetricsCollector, RequestMetrics
from .load_reduction import (
    apply_load_reduction,
    enforce_max_output_tokens,
    get_max_output_tokens,
)
from .queue_manager import QueueFullError, RequestQueue
from .reservation import ReservationLedger
from .token_bucket import BucketRegistry

logger = logging.getLogger(__name__)


class ClawGuardProxy:
    """The main proxy that sits between OpenClaw and the Gemini API."""

    def __init__(self, config: ClawGuardConfig) -> None:
        self.config = config

        # Core components
        self.buckets = BucketRegistry()
        self.ledger = ReservationLedger(self.buckets)
        self.fallback = FallbackManager(self.buckets)
        self.queue = RequestQueue()
        self.batch_router = BatchRouter(
            gemini_base_url=config.gemini_base_url,
            api_key=config.api_key,
        )
        self.cache_manager = CacheManager(
            api_key=config.api_key,
            gemini_base_url=config.gemini_base_url,
        )
        self.metrics = MetricsCollector(metrics_file=config.metrics_file)

        # HTTP client session (created on start)
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Initialize async resources."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=RESERVATION_EXPIRY_SECONDS,
            )
        )
        await self.ledger.start()
        await self.queue.start()
        await self.batch_router.start()
        logger.info(
            "Claw Guard started: mode=%s, upstream=%s",
            self.config.mode.value, self.config.gemini_base_url,
        )

    async def stop(self) -> None:
        """Shut down async resources."""
        await self.ledger.stop()
        await self.queue.stop()
        await self.batch_router.stop()
        if self._session:
            await self._session.close()
        logger.info("Claw Guard stopped")

    # ------------------------------------------------------------------
    # aiohttp request handler
    # ------------------------------------------------------------------

    async def handle_request(self, request: web.Request) -> web.Response:
        """Main entry point for all proxied requests."""
        request_id = uuid.uuid4().hex[:12]
        start_time = time.monotonic()

        # Read body
        try:
            raw_body = await request.read()
            body = json.loads(raw_body) if raw_body else {}
        except (json.JSONDecodeError, Exception) as e:
            logger.error("Failed to parse request body: %s", e)
            # In passthrough mode, forward anyway
            if self.config.mode == OperatingMode.PASSTHROUGH:
                return await self._passthrough(request, raw_body or b"")
            return web.json_response(
                {"error": f"Invalid request body: {e}"},
                status=400,
            )

        # Identify agent
        agent_id = self._identify_agent(request, body)
        model_requested = self._extract_model_from_path(request.path)

        logger.debug(
            "Request %s: agent=%s model=%s path=%s",
            request_id, agent_id, model_requested, request.path,
        )

        # --- PASSTHROUGH MODE ---
        if self.config.mode == OperatingMode.PASSTHROUGH:
            response = await self._passthrough(request, raw_body or b"")
            # Still log metrics in passthrough
            await self._log_passthrough_metrics(
                request_id, agent_id, model_requested,
                response, start_time, body,
            )
            return response

        # --- BATCH ROUTING (non-Xeon agents) ---
        if agent_id != XEON_AGENT_ID:
            return await self._handle_batch_request(
                request_id, agent_id, model_requested, body,
            )

        # --- XEON REAL-TIME PATH ---
        return await self._handle_xeon_request(
            request, request_id, agent_id, model_requested,
            body, raw_body or b"", start_time,
        )

    # ------------------------------------------------------------------
    # Xeon real-time path (the core guardrail logic)
    # ------------------------------------------------------------------

    async def _handle_xeon_request(
        self,
        request: web.Request,
        request_id: str,
        agent_id: str,
        model_requested: str,
        body: Dict[str, Any],
        raw_body: bytes,
        start_time: float,
    ) -> web.Response:
        """Handle a Xeon real-time request through the full guardrail pipeline."""

        is_in_fallback = self.fallback.is_in_fallback()

        # Estimate prompt tokens
        prompt_text = self._extract_prompt_text(body)
        estimated_prompt_tokens = max(1, int(len(prompt_text) / CHARS_PER_TOKEN))
        max_output = get_max_output_tokens(
            model_requested, is_in_fallback=is_in_fallback,
        )
        estimated_total = estimated_prompt_tokens + max_output

        # --- SHADOW MODE: simulate but don't block ---
        if self.config.mode == OperatingMode.SHADOW:
            return await self._handle_shadow_mode(
                request, request_id, agent_id, model_requested,
                body, raw_body, start_time,
                estimated_total,
            )

        # --- ACTIVE MODE ---
        # Step 1: Select model (fallback chain + recovery)
        selected_model = self.fallback.select_model(float(estimated_total))

        if selected_model is None:
            # Chain exhausted — queue the request
            return await self._handle_queued_request(
                request, request_id, agent_id, model_requested,
                body, raw_body, start_time,
                estimated_total,
            )

        is_in_fallback = self.fallback.is_in_fallback()

        # Step 2: Apply load reduction if in fallback
        body = apply_load_reduction(body, selected_model, is_in_fallback)

        # Step 3: Enforce max_output_tokens
        body = enforce_max_output_tokens(body, selected_model, is_in_fallback)

        # Re-estimate after load reduction may have changed body
        prompt_text = self._extract_prompt_text(body)
        estimated_prompt_tokens = max(1, int(len(prompt_text) / CHARS_PER_TOKEN))
        max_output = body.get("generationConfig", {}).get(
            "maxOutputTokens", max_output
        )
        estimated_total = estimated_prompt_tokens + max_output

        # Step 4: Check/create explicit cache
        body = self.cache_manager.attach_cache_to_request(
            body, selected_model,
            is_active_model=True,
        )

        # Step 5: Reserve tokens
        reservation = self.ledger.create_reservation(
            model_id=selected_model,
            estimated_prompt_tokens=estimated_prompt_tokens,
            max_output_tokens=max_output,
        )
        if reservation is None:
            # Reservation failed (bucket check passed in select_model but
            # race condition drained it) — queue
            return await self._handle_queued_request(
                request, request_id, agent_id, model_requested,
                body, raw_body, start_time,
                estimated_total,
            )

        # Step 6: Forward to Gemini (with retry on 429)
        metrics = RequestMetrics(
            request_id=request_id,
            agent_id=agent_id,
            model_requested=model_requested,
            model_served=selected_model,
            fallback_depth=self.fallback.fallback_depth,
            estimated_prompt_tokens=estimated_prompt_tokens,
            max_output_tokens=max_output,
            total_reserved=estimated_total,
            prompt_char_count=len(prompt_text),
        )

        # Record bucket state
        pair = self.buckets.get(selected_model)
        if pair:
            metrics.bucket_tpm_available = pair.tpm_bucket.available
            metrics.bucket_rpm_available = pair.rpm_bucket.available

        response = await self._forward_with_retry(
            request, request_id, selected_model, body, metrics,
        )

        # Step 7: Settle reservation
        if response is not None:
            response_body = await self._try_parse_response(response)
            usage = self.metrics.extract_usage_metadata(response_body)

            actual_total = usage.get("total_tokens")
            if actual_total:
                metrics.actual_prompt_tokens = usage.get("prompt_tokens")
                metrics.actual_output_tokens = usage.get("output_tokens")
                metrics.actual_total_tokens = actual_total
                metrics.cached_content_token_count = usage.get("cached_tokens")
                credit = self.ledger.settle(reservation.reservation_id, actual_total)
                metrics.credit_back = credit
            else:
                # No usage metadata — release full reservation
                self.ledger.release(reservation.reservation_id)

            # Calibrate token ratio
            if metrics.actual_prompt_tokens:
                metrics.actual_chars_per_token = self.metrics.calibrate_token_ratio(
                    len(prompt_text), metrics.actual_prompt_tokens,
                )

            # Extract rate limit headers
            if hasattr(response, "_headers"):
                metrics.rate_limit_headers = self.metrics.extract_rate_limit_headers(
                    dict(response._headers)
                )

            metrics.upstream_latency_ms = (time.monotonic() - start_time) * 1000
            metrics.status_code = response.status
            self.metrics.record(metrics)

            # Build proxied response with injected tags
            return self._build_tagged_response(
                response, response_body,
                model_requested, selected_model,
                self.fallback.fallback_depth,
            )
        else:
            # Forward failed entirely
            self.ledger.release(reservation.reservation_id)
            metrics.error = "All retries exhausted"
            metrics.upstream_latency_ms = (time.monotonic() - start_time) * 1000
            self.metrics.record(metrics)
            return web.json_response(
                {"error": "Upstream Gemini API unreachable after retries"},
                status=502,
            )

    # ------------------------------------------------------------------
    # Queued request handling
    # ------------------------------------------------------------------

    async def _handle_queued_request(
        self,
        request: web.Request,
        request_id: str,
        agent_id: str,
        model_requested: str,
        body: Dict[str, Any],
        raw_body: bytes,
        start_time: float,
        estimated_total: int,
    ) -> web.Response:
        """Handle a request that must wait in the queue for capacity."""
        try:
            qr = await self.queue.enqueue(request_id, float(estimated_total))
        except QueueFullError as e:
            return web.json_response(
                {
                    "error": "Rate limit exceeded — queue full",
                    "queue_depth": e.depth,
                    "estimated_wait_seconds": round(e.estimated_wait, 1),
                },
                status=429,
                headers={"Retry-After": str(int(e.estimated_wait))},
            )

        # Wait for capacity
        got_capacity = await self.queue.wait_for_capacity(qr)
        if not got_capacity:
            return web.json_response(
                {"error": "Request timed out waiting for capacity"},
                status=503,
            )

        # Capacity available — retry the request with the assigned model
        # Re-enter the pipeline with the model the queue assigned
        return await self._handle_xeon_request(
            request, request_id, agent_id, model_requested,
            body, raw_body, start_time,
        )

    # ------------------------------------------------------------------
    # Shadow mode
    # ------------------------------------------------------------------

    async def _handle_shadow_mode(
        self,
        request: web.Request,
        request_id: str,
        agent_id: str,
        model_requested: str,
        body: Dict[str, Any],
        raw_body: bytes,
        start_time: float,
        estimated_total: int,
    ) -> web.Response:
        """Shadow mode: forward request unchanged, but log what WOULD happen."""
        # Simulate fallback selection
        selected = self.fallback.select_model(float(estimated_total))

        metrics = RequestMetrics(
            request_id=request_id,
            agent_id=agent_id,
            model_requested=model_requested,
        )

        if selected is None:
            metrics.would_queue = True
            self.metrics.record_shadow_event(
                "would_queue", request_id,
                {"estimated_tokens": estimated_total},
            )
        elif selected != FALLBACK_CHAIN[0]:
            metrics.would_fallback = True
            metrics.would_fallback_to = selected
            self.metrics.record_shadow_event(
                "would_fallback", request_id,
                {"from": FALLBACK_CHAIN[0], "to": selected},
            )

        # Check if current model bucket would throttle
        pair = self.buckets.get(model_requested or FALLBACK_CHAIN[0])
        if pair and not pair.has_capacity(float(estimated_total)):
            metrics.would_throttle = True
            self.metrics.record_shadow_event(
                "would_throttle", request_id,
                {
                    "model": model_requested,
                    "estimated_tokens": estimated_total,
                    "tpm_available": pair.tpm_bucket.available,
                    "rpm_available": pair.rpm_bucket.available,
                },
            )

        # Forward unchanged
        response = await self._passthrough(request, raw_body)

        metrics.status_code = response.status
        metrics.upstream_latency_ms = (time.monotonic() - start_time) * 1000
        self.metrics.record(metrics)

        return response

    # ------------------------------------------------------------------
    # Batch routing
    # ------------------------------------------------------------------

    async def _handle_batch_request(
        self,
        request_id: str,
        agent_id: str,
        model_id: str,
        body: Dict[str, Any],
    ) -> web.Response:
        """Route non-Xeon agent requests to the batch API."""
        logger.info(
            "Batch routing: request=%s agent=%s model=%s",
            request_id, agent_id, model_id,
        )

        try:
            future = await self.batch_router.submit(agent_id, model_id, body)
            # Wait for batch result (this may take a long time)
            result = await asyncio.wait_for(future, timeout=86400)  # 24h SLO
            return web.json_response(result)
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": "Batch job timed out (24h SLO exceeded)"},
                status=504,
            )
        except Exception as e:
            logger.error("Batch routing failed: %s", e)
            return web.json_response(
                {"error": f"Batch routing error: {e}"},
                status=500,
            )

    # ------------------------------------------------------------------
    # Forwarding and retry
    # ------------------------------------------------------------------

    async def _forward_with_retry(
        self,
        original_request: web.Request,
        request_id: str,
        model_id: str,
        body: Dict[str, Any],
        metrics: RequestMetrics,
    ) -> Optional[aiohttp.ClientResponse]:
        """Forward request to Gemini with exponential backoff on 429."""
        upstream_url = self._build_upstream_url(original_request.path, model_id)
        headers = self._build_upstream_headers(original_request)

        for attempt in range(7):  # initial + 6 retries
            try:
                resp = await self._session.request(
                    method=original_request.method,
                    url=upstream_url,
                    headers=headers,
                    json=body,
                    params={"key": self.config.api_key} if self.config.api_key else None,
                )

                if resp.status != 429 and resp.status < 500:
                    return resp

                if not should_retry(attempt, resp.status):
                    return resp

                # Extract headers for backoff calculation
                resp_headers = {k: v for k, v in resp.headers.items()}
                await backoff_sleep(attempt, resp_headers)

            except asyncio.TimeoutError:
                logger.warning(
                    "Request %s attempt %d timed out (%.0fs)",
                    request_id, attempt, RESERVATION_EXPIRY_SECONDS,
                )
                if attempt >= 2:
                    return None
            except aiohttp.ClientError as e:
                logger.error(
                    "Request %s attempt %d client error: %s",
                    request_id, attempt, e,
                )
                if attempt >= 2:
                    return None
                await backoff_sleep(attempt)

        return None

    async def _passthrough(
        self, request: web.Request, raw_body: bytes
    ) -> web.Response:
        """Forward a request directly to Gemini without any guardrail logic."""
        upstream_url = self._build_upstream_url_direct(request.path)
        headers = self._build_upstream_headers(request)

        try:
            resp = await self._session.request(
                method=request.method,
                url=upstream_url,
                headers=headers,
                data=raw_body,
                params={"key": self.config.api_key} if self.config.api_key else None,
            )
            resp_body = await resp.read()
            return web.Response(
                body=resp_body,
                status=resp.status,
                headers={
                    k: v for k, v in resp.headers.items()
                    if k.lower() not in ("transfer-encoding", "content-encoding")
                },
            )
        except Exception as e:
            logger.error("Passthrough failed: %s", e)
            return web.json_response(
                {"error": f"Upstream request failed: {e}"},
                status=502,
            )

    # ------------------------------------------------------------------
    # Response building
    # ------------------------------------------------------------------

    def _build_tagged_response(
        self,
        upstream_response: aiohttp.ClientResponse,
        response_body: Dict[str, Any],
        model_requested: str,
        model_served: str,
        fallback_depth: int,
    ) -> web.Response:
        """Build the response back to OpenClaw with injected Claw Guard tags."""
        # Copy upstream headers, skip hop-by-hop
        headers = {
            k: v for k, v in upstream_response.headers.items()
            if k.lower() not in (
                "transfer-encoding", "content-encoding", "connection",
            )
        }

        # Inject Claw Guard response tags (Section 2.4)
        headers["x-clawguard-model-requested"] = model_requested
        headers["x-clawguard-model-served"] = model_served
        headers["x-clawguard-fallback-depth"] = str(fallback_depth)

        return web.Response(
            body=json.dumps(response_body).encode(),
            status=upstream_response.status,
            content_type="application/json",
            headers=headers,
        )

    # ------------------------------------------------------------------
    # Agent identification
    # ------------------------------------------------------------------

    def _identify_agent(
        self, request: web.Request, body: Dict[str, Any]
    ) -> str:
        """Determine which OpenClaw agent originated this request.

        TODO: The exact mechanism for agent identification depends on how
              OpenClaw tags requests.  Possible locations:
              1. Custom header: ``X-OpenClaw-Agent-Id``
              2. Query parameter: ``?agent=xeon``
              3. Body field: ``metadata.agent_id``
              4. Part of the request path

        PLACEHOLDER: Checks header, query param, and body field.
        Falls back to "xeon" (conservative — treats unknown as real-time).
        """
        # Try custom header
        agent = request.headers.get("X-OpenClaw-Agent-Id", "").strip().lower()
        if agent:
            return agent

        # Try query parameter
        agent = request.query.get("agent", "").strip().lower()
        if agent:
            return agent

        # Try body metadata
        metadata = body.get("metadata", {})
        if isinstance(metadata, dict):
            agent = metadata.get("agent_id", "").strip().lower()
            if agent:
                return agent

        # Fallback: assume Xeon (conservative — applies guardrails)
        return XEON_AGENT_ID

    # ------------------------------------------------------------------
    # Model extraction from request path
    # ------------------------------------------------------------------

    def _extract_model_from_path(self, path: str) -> str:
        """Extract the model ID from the request URL path.

        TODO: Verify actual path format from OpenClaw/Gemini traffic.
              Expected patterns:
                /v1/models/gemini-3-pro:generateContent
                /v1/models/gemini-2.5-pro:streamGenerateContent

        PLACEHOLDER: Parses ``/v1/models/{model}:method`` format.
        """
        # /v1/models/{model_id}:{method}
        parts = path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part == "models" and i + 1 < len(parts):
                model_part = parts[i + 1]
                # Strip the :method suffix
                if ":" in model_part:
                    return model_part.split(":")[0]
                return model_part
        return ""

    # ------------------------------------------------------------------
    # URL building
    # ------------------------------------------------------------------

    def _build_upstream_url(self, original_path: str, target_model: str) -> str:
        """Build the upstream Gemini URL, potentially rewriting the model.

        If fallback changed the model, rewrite the path to use the new model.
        """
        original_model = self._extract_model_from_path(original_path)
        if original_model and target_model and original_model != target_model:
            # Rewrite the model in the path
            new_path = original_path.replace(original_model, target_model)
            return f"{self.config.gemini_base_url}{new_path}"
        return f"{self.config.gemini_base_url}{original_path}"

    def _build_upstream_url_direct(self, original_path: str) -> str:
        """Build upstream URL without any model rewriting."""
        return f"{self.config.gemini_base_url}{original_path}"

    def _build_upstream_headers(self, request: web.Request) -> Dict[str, str]:
        """Build headers for the upstream request.

        Strips hop-by-hop headers and OpenClaw-specific headers.
        """
        skip = {
            "host", "transfer-encoding", "connection",
            "x-openclaw-agent-id",
        }
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in skip
        }
        return headers

    # ------------------------------------------------------------------
    # Body parsing helpers
    # ------------------------------------------------------------------

    def _extract_prompt_text(self, body: Dict[str, Any]) -> str:
        """Extract the raw text from the request body for token estimation.

        TODO: This depends on the exact Gemini request format.
              Standard generateContent uses:
                contents: [{role: "user", parts: [{text: "..."}]}]
              plus systemInstruction and tools.
        """
        parts = []

        # System instruction
        si = body.get("systemInstruction")
        if si:
            parts.append(json.dumps(si))

        # Tools
        tools = body.get("tools")
        if tools:
            parts.append(json.dumps(tools))

        # Contents
        contents = body.get("contents", [])
        for entry in contents:
            if isinstance(entry, dict):
                for part in entry.get("parts", []):
                    if isinstance(part, dict) and "text" in part:
                        parts.append(part["text"])

        return " ".join(parts) if parts else json.dumps(body)

    async def _try_parse_response(
        self, response: aiohttp.ClientResponse
    ) -> Dict[str, Any]:
        """Try to parse the upstream response body as JSON."""
        try:
            body = await response.json()
            return body if isinstance(body, dict) else {}
        except Exception:
            try:
                text = await response.text()
                return json.loads(text)
            except Exception:
                return {}

    # ------------------------------------------------------------------
    # Status endpoint
    # ------------------------------------------------------------------

    async def handle_status(self, request: web.Request) -> web.Response:
        """Return Claw Guard internal state for monitoring."""
        return web.json_response({
            "mode": self.config.mode.value,
            "version": "0.1.0",
            "buckets": self.buckets.snapshot(),
            "fallback": self.fallback.snapshot(),
            "queue": self.queue.snapshot(),
            "reservations": {
                "in_flight": self.ledger.in_flight_count,
                "details": self.ledger.snapshot(),
            },
            "batch": self.batch_router.snapshot(),
            "cache": self.cache_manager.snapshot(),
            "metrics": self.metrics.snapshot(),
        })

    async def handle_health(self, request: web.Request) -> web.Response:
        """Simple health check endpoint."""
        return web.json_response({"status": "ok", "mode": self.config.mode.value})
