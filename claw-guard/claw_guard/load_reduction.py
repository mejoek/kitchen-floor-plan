"""Active load reduction during fallback mode.

When Xeon is locked to a fallback model, Claw Guard reduces token pressure
to accelerate primary bucket recovery (Section 2.4).

Three mechanisms:
  1. Context pruning — trim conversation history to last N turns
  2. Stricter max_output_tokens
  3. Cache enforcement (reject requests that can't use cache if caching is active)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional

from .config import (
    CONTEXT_PRUNE_MAX_TURNS,
    MAX_OUTPUT_TOKENS,
)

logger = logging.getLogger(__name__)


def apply_load_reduction(
    request_body: Dict[str, Any],
    model_id: str,
    is_in_fallback: bool,
) -> Dict[str, Any]:
    """Mutate (a copy of) the request body to apply load reduction when in fallback.

    Returns the (possibly modified) body.  If not in fallback, returns unchanged.
    """
    if not is_in_fallback:
        return request_body

    body = copy.deepcopy(request_body)

    body = _enforce_max_output_tokens(body, model_id, is_fallback=True)
    body = _prune_context(body)

    return body


def enforce_max_output_tokens(
    request_body: Dict[str, Any],
    model_id: str,
    is_in_fallback: bool,
) -> Dict[str, Any]:
    """Ensure max_output_tokens is present with appropriate default.

    Called on ALL Xeon requests, not just fallback.
    """
    body = copy.deepcopy(request_body)
    return _enforce_max_output_tokens(body, model_id, is_fallback=is_in_fallback)


def _enforce_max_output_tokens(
    body: Dict[str, Any],
    model_id: str,
    is_fallback: bool,
) -> Dict[str, Any]:
    """Inject or reduce max_output_tokens on the request body.

    TODO: The exact field name and nesting depends on how OpenClaw formats
          the request body for the Gemini API.  Gemini's generateContent API
          uses ``generationConfig.maxOutputTokens``.  Adjust once we observe
          real traffic in Phase 1.
    """
    defaults = MAX_OUTPUT_TOKENS.get(model_id)
    if defaults is None:
        return body

    target = defaults.fallback if is_fallback else defaults.normal

    # Gemini REST API: generationConfig.maxOutputTokens
    gen_config = body.setdefault("generationConfig", {})
    current = gen_config.get("maxOutputTokens")

    if current is None:
        gen_config["maxOutputTokens"] = target
        logger.debug("Injected maxOutputTokens=%d for model %s", target, model_id)
    elif is_fallback and current > target:
        gen_config["maxOutputTokens"] = target
        logger.debug(
            "Reduced maxOutputTokens %d → %d (fallback) for model %s",
            current, target, model_id,
        )

    return body


def _prune_context(body: Dict[str, Any]) -> Dict[str, Any]:
    """Trim conversation history to the last N complete turns.

    A turn is one user message + one model response.  We walk backwards
    to find complete pairs so we never cut mid-turn (e.g. leaving a user
    message without its model response), and we ensure the kept history
    starts with a user message.

    TODO: The exact field name depends on how OpenClaw structures the
          ``contents`` array in the Gemini generateContent request.
          The standard Gemini format uses ``contents: [{role, parts}, ...]``.
          Adjust once we observe real traffic.
    """
    contents = body.get("contents")
    if not isinstance(contents, list):
        return body

    # Walk backwards counting complete turns (model response + preceding user message)
    turns_found = 0
    keep_from = len(contents)
    i = len(contents) - 1

    while i >= 1 and turns_found < CONTEXT_PRUNE_MAX_TURNS:
        current = contents[i]
        previous = contents[i - 1]

        current_role = current.get("role", "") if isinstance(current, dict) else ""
        prev_role = previous.get("role", "") if isinstance(previous, dict) else ""

        if current_role == "model" and prev_role == "user":
            turns_found += 1
            keep_from = i - 1
            i -= 2
        else:
            i -= 1

    if keep_from > 0 and keep_from < len(contents):
        original_len = len(contents)
        body["contents"] = contents[keep_from:]
        logger.info(
            "Context pruned: %d entries → %d (%d complete turns)",
            original_len, len(body["contents"]), turns_found,
        )

    return body


def get_max_output_tokens(model_id: str, is_in_fallback: bool) -> int:
    """Return the effective max_output_tokens for a model and mode."""
    defaults = MAX_OUTPUT_TOKENS.get(model_id)
    if defaults is None:
        # Conservative fallback if model unknown
        return 512
    return defaults.fallback if is_in_fallback else defaults.normal
