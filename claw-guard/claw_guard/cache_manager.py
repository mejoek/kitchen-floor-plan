"""Lazy explicit context caching for Gemini models.

Caches are created on demand — only when a model is actually used.
Section 2.6 of the architecture doc.

Gemini's explicit caching API stores pre-computed input tokens server-side,
referenced by cache name in subsequent requests.  Cached tokens are billed
at 90% less on Gemini 2.5+ and 75% less on Gemini 2.0.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import (
    CACHE_DEFAULT_TTL_SECONDS,
    CACHE_STABLE_TTL_SECONDS,
    CACHE_MIN_TOKENS,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Tracks a single explicit cache created on the Gemini side."""

    cache_name: str          # Gemini-assigned cache resource name
    model_id: str
    content_hash: str        # SHA-256 of the cached content
    content_type: str        # "system_prompt", "tools", "reference_docs", etc.
    estimated_tokens: int
    created_at: float = field(default_factory=time.monotonic)
    ttl_seconds: int = CACHE_DEFAULT_TTL_SECONDS
    last_used: float = field(default_factory=time.monotonic)

    @property
    def expires_at(self) -> float:
        return self.created_at + self.ttl_seconds

    @property
    def is_expired(self) -> bool:
        return time.monotonic() > self.expires_at

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at


class CacheManager:
    """Manages lazy creation and lifecycle of explicit Gemini caches."""

    def __init__(self, api_key: str = "", gemini_base_url: str = "") -> None:
        self._api_key = api_key
        self._base_url = gemini_base_url
        # Key: (model_id, content_hash) → CacheEntry
        self._caches: Dict[tuple, CacheEntry] = {}

    def identify_cacheable_content(
        self, request_body: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract cacheable content blocks from a request body.

        TODO: The exact structure depends on how OpenClaw formats requests.
              Typical cacheable blocks:
              - systemInstruction (system prompt / persona)
              - tools (function/tool definitions)
              - Large static content in the first entries of contents[]

        Returns a list of dicts with:
          - content_type: str
          - content: the raw content to hash
          - estimated_tokens: int
          - ttl_category: "stable" or "default"
        """
        blocks = []

        # System instruction — high priority, stable TTL
        system_instruction = request_body.get("systemInstruction")
        if system_instruction:
            content_str = json.dumps(system_instruction, sort_keys=True)
            blocks.append({
                "content_type": "system_prompt",
                "content": content_str,
                "estimated_tokens": max(1, len(content_str) // 4),
                "ttl_category": "stable",
            })

        # Tool definitions — high priority, stable TTL
        tools = request_body.get("tools")
        if tools:
            content_str = json.dumps(tools, sort_keys=True)
            blocks.append({
                "content_type": "tools",
                "content": content_str,
                "estimated_tokens": max(1, len(content_str) // 4),
                "ttl_category": "stable",
            })

        return blocks

    def get_or_create_cache(
        self,
        model_id: str,
        content_block: Dict[str, Any],
        is_active_model: bool,
    ) -> Optional[str]:
        """Return the cache name for the given content, creating if needed.

        Args:
            model_id: The model this cache is for.
            content_block: Output from identify_cacheable_content().
            is_active_model: True if this is the primary or active fallback model.
                Only create new caches for active models.

        Returns:
            The Gemini cache resource name, or None if caching is not applicable.
        """
        content_hash = self._hash_content(content_block["content"])
        cache_key = (model_id, content_hash)

        # Check for existing valid cache
        entry = self._caches.get(cache_key)
        if entry and not entry.is_expired:
            entry.last_used = time.monotonic()
            return entry.cache_name

        # Remove expired entry if present
        if entry and entry.is_expired:
            logger.info(
                "Cache expired: model=%s type=%s hash=%s",
                model_id, content_block["content_type"], content_hash[:12],
            )
            del self._caches[cache_key]

        # Only create for active models
        if not is_active_model:
            return None

        # Check minimum token threshold
        min_tokens = CACHE_MIN_TOKENS.get(model_id, 2048)
        if content_block["estimated_tokens"] < min_tokens:
            logger.debug(
                "Content below min cache threshold (%d < %d) for model %s",
                content_block["estimated_tokens"], min_tokens, model_id,
            )
            return None

        # Create the cache
        cache_name = self._create_cache(model_id, content_block, content_hash)
        if cache_name:
            ttl = (
                CACHE_STABLE_TTL_SECONDS
                if content_block.get("ttl_category") == "stable"
                else CACHE_DEFAULT_TTL_SECONDS
            )
            self._caches[cache_key] = CacheEntry(
                cache_name=cache_name,
                model_id=model_id,
                content_hash=content_hash,
                content_type=content_block["content_type"],
                estimated_tokens=content_block["estimated_tokens"],
                ttl_seconds=ttl,
            )
        return cache_name

    def attach_cache_to_request(
        self,
        request_body: Dict[str, Any],
        model_id: str,
        is_active_model: bool,
    ) -> Dict[str, Any]:
        """Identify cacheable content, get/create caches, and attach references.

        TODO: The exact mechanism for attaching a cache reference depends on
              the Gemini API.  The current API uses:
                ``cachedContent: "cachedContents/{cache_name}"``
              as a top-level field in the generateContent request.

        Returns the (possibly modified) request body.
        """
        blocks = self.identify_cacheable_content(request_body)
        if not blocks:
            return request_body

        # For now, we only cache one block (system prompt takes priority)
        # Gemini's cachedContent field supports one cache reference per request.
        for block in blocks:
            cache_name = self.get_or_create_cache(model_id, block, is_active_model)
            if cache_name:
                # Attach cache reference
                # TODO: Verify this is the correct field name from Gemini API docs
                request_body["cachedContent"] = f"cachedContents/{cache_name}"
                logger.debug(
                    "Attached cache %s (type=%s) to request for model %s",
                    cache_name, block["content_type"], model_id,
                )
                break  # only one cache per request

        return request_body

    def _create_cache(
        self,
        model_id: str,
        content_block: Dict[str, Any],
        content_hash: str,
    ) -> Optional[str]:
        """Create an explicit cache on the Gemini API.

        TODO: Implement actual Gemini cache creation API call.

        The Gemini Caching API:
          POST /v1beta/cachedContents
          {
            "model": "models/{model_id}",
            "displayName": "claw-guard-{content_type}-{hash[:8]}",
            "contents": [...],  // or systemInstruction, tools, etc.
            "ttl": "{ttl_seconds}s",
          }

        Returns the cache resource name on success, None on failure.

        PLACEHOLDER: Returns a synthetic cache name for now.
        """
        cache_display = f"claw-guard-{content_block['content_type']}-{content_hash[:8]}"

        logger.info(
            "CACHE CREATE [placeholder]: model=%s type=%s hash=%s tokens=%d",
            model_id, content_block["content_type"],
            content_hash[:12], content_block["estimated_tokens"],
        )

        # --- PLACEHOLDER ---
        # In production, this would make an HTTP POST to:
        #   {base_url}/cachedContents
        # and return the ``name`` field from the response.
        #
        # For now, return a synthetic name.
        return f"placeholder-{cache_display}"
        # --- END PLACEHOLDER ---

    def _hash_content(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()

    def cleanup_expired(self) -> int:
        """Remove expired cache entries.  Returns count removed."""
        expired_keys = [
            k for k, v in self._caches.items() if v.is_expired
        ]
        for k in expired_keys:
            del self._caches[k]
        if expired_keys:
            logger.info("Cleaned up %d expired cache entries", len(expired_keys))
        return len(expired_keys)

    def snapshot(self) -> dict:
        return {
            "active_caches": len(self._caches),
            "entries": [
                {
                    "model_id": e.model_id,
                    "content_type": e.content_type,
                    "estimated_tokens": e.estimated_tokens,
                    "age_seconds": round(e.age_seconds, 1),
                    "ttl_seconds": e.ttl_seconds,
                    "is_expired": e.is_expired,
                    "cache_name": e.cache_name,
                }
                for e in self._caches.values()
            ],
        }
