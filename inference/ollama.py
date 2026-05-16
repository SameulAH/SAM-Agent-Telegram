import json
import re
import time
import logging
import httpx
from .base import ModelBackend
from .types import ModelRequest, ModelResponse

# ──────────────────────────────────────────────────────────────────────────────
# Import the authoritative SYSTEM_PROMPT from the prompt builder layer.
# This replaces the previously inline _SYSTEM_PROMPT so the behavioral
# contract is defined in a single place.
# ──────────────────────────────────────────────────────────────────────────────
from agent.prompting.prompt_builder import SYSTEM_PROMPT as _SYSTEM_PROMPT  # noqa: E402

logger = logging.getLogger(__name__)

# Regex to locate the [TOOL_CALL] marker (case-insensitive for robustness)
# Also matches [Web_Search] and [web_search] emitted by phi3:mini and similar models
_TOOL_CALL_MARKER_RE = re.compile(r"\[TOOL_CALL\]|\[Web_Search\]", re.IGNORECASE)
# Fallback: pull the query field out of truncated JSON.
# The closing quote is optional so truncated values like "define got your messa)
# (missing the closing ") are still captured.
_QUERY_FALLBACK_RE = re.compile(r'"query"\s*:\s*"([^"]*)"?')
# Pattern for bare {"query": "..."} that phi3:mini emits after [Web_Search]
_BARE_QUERY_RE = re.compile(r'\{\s*"query"\s*:\s*"([^"]*)"', re.IGNORECASE)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BASE_DELAY_S = 1.0  # doubles each attempt: 1s, 2s, 4s


def _extract_tool_call(output: str):
    """
    Find [TOOL_CALL]{...} or [Web_Search]{...} in model output using brace-counting.

    The old single-regex approach ({.*?}) stops at the FIRST closing brace,
    which is the inner 'arguments' object — producing invalid JSON every time.
    Brace-counting correctly handles nested objects.

    Also handles phi3:mini's [Web_Search]{"query": "..."} shorthand by
    normalising it into the standard {"name": "web_search", "arguments": {...}}.

    Returns:
        (tool_call_dict, cleaned_output) on success
        (None,           cleaned_output) if marker found but JSON unrecoverable
        (None,           original_output) if no marker at all
    """
    marker = _TOOL_CALL_MARKER_RE.search(output)
    if not marker:
        return None, output

    marker_text = marker.group(0).lower()  # "[tool_call]" or "[web_search]"
    is_web_search_shorthand = "web_search" in marker_text

    # Always strip [TOOL_CALL]/[Web_Search] and everything after it from the visible output
    clean = output[: marker.start()].strip()

    brace_start = output.find("{", marker.end())
    if brace_start == -1:
        return None, clean  # marker present but no JSON body

    # Walk forward counting braces to find the matching close
    depth = 0
    brace_end = -1
    for i, ch in enumerate(output[brace_start:], brace_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                brace_end = i + 1
                break

    json_str = output[brace_start:brace_end] if brace_end != -1 else output[brace_start:]

    try:
        parsed = json.loads(json_str)
        # [Web_Search]{"query": "..."} — bare query dict, normalise it
        if is_web_search_shorthand or ("query" in parsed and "name" not in parsed):
            return {
                "name": "web_search",
                "arguments": {"query": parsed.get("query", "")},
            }, clean
        return parsed, clean
    except json.JSONDecodeError:
        # Truncated / malformed JSON — try to at least salvage the query string
        q = _QUERY_FALLBACK_RE.search(json_str) or _BARE_QUERY_RE.search(output[marker.end():])
        if q:
            return {
                "name": "web_search",
                "arguments": {"query": q.group(1)},
            }, clean
        return None, clean  # nothing usable, but marker is still hidden from user


def _try_loose_tool_call(output: str):
    """
    Fallback tool-call detection for models that don't use [TOOL_CALL] marker.

    Handles:
      • {"name": "web_search", "arguments": {"query": "…"}}  — raw structured JSON
      • web_search{"query": "…"}                              — loose function syntax
      • web_search({"query": "…"})                           — function-call style

    Returns:
        (tool_call_dict, cleaned_output) on success
        (None,           original_output) if no pattern found
    """
    if not output or not output.strip():
        return None, output

    # ── Strategy 1: raw structured JSON {"name": "...", "arguments": {...}} ──
    m = re.search(
        r'\{\s*"name"\s*:\s*"([\w_]+)"\s*,\s*"arguments"\s*:\s*(\{)',
        output,
    )
    if m:
        tool_name = m.group(1)
        brace_pos = m.start(2)
        depth = 0
        brace_end = -1
        for i, ch in enumerate(output[brace_pos:], brace_pos):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    brace_end = i + 1
                    break
        if brace_end != -1:
            try:
                args = json.loads(output[brace_pos:brace_end])
                return {"name": tool_name, "arguments": args}, ""
            except json.JSONDecodeError:
                pass

    # ── Strategy 2: loose syntax — known_tool{...} or known_tool({...}) ──────
    for tool_name in ("web_search",):
        pattern = re.compile(
            rf"\b{re.escape(tool_name)}\s*\(?\s*(\{{)",
            re.IGNORECASE,
        )
        m = pattern.search(output)
        if m:
            brace_pos = m.start(1)
            depth = 0
            brace_end = -1
            for i, ch in enumerate(output[brace_pos:], brace_pos):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        brace_end = i + 1
                        break
            if brace_end != -1:
                json_str = output[brace_pos:brace_end]
                try:
                    args = json.loads(json_str)
                    # Normalise: {"arguments": {...}} → unwrap
                    if "arguments" in args:
                        args = args["arguments"]
                    return {"name": tool_name, "arguments": args}, ""
                except json.JSONDecodeError:
                    # Salvage query string from malformed JSON
                    qm = re.search(r'"query"\s*:\s*"([^"]*)"', json_str)
                    if qm:
                        return {
                            "name": tool_name,
                            "arguments": {"query": qm.group(1)},
                        }, ""

    return None, output


class OllamaModelBackend(ModelBackend):
    """
    Ollama backend for local model inference.

    Uses /api/chat so a system prompt can be included.
    Parses [TOOL_CALL]{...} markers from the model output and stores the
    decoded tool call in ModelResponse.metadata["tool_call"] so the
    LangGraph orchestrator can route to the tool execution node.

    Retries transient failures (connection errors, timeouts) up to
    _MAX_RETRIES times with exponential backoff before returning a
    recoverable_error status. This handles Ollama container restarts
    and transient network blips without surfacing errors to the user.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama backend.

        Args:
            model_name: Name of the model (e.g. "phi3:mini", "llama3")
            base_url:   Base URL of the Ollama service
        """
        self.model_name = model_name
        self.base_url = base_url

    def generate(self, request: ModelRequest) -> ModelResponse:
        """
        Generate a response using Ollama /api/chat.

        Flow:
          1. Build a messages list: [system, (optional context), user]
          2. POST to /api/chat with up to _MAX_RETRIES retries on transient errors
          3. Parse response for [TOOL_CALL] marker
          4. If found → set metadata["tool_call"] so orchestrator routes to tool node
          5. If request.context is provided (second call after tool execution) →
             prepend tool results to the user message so the model can cite them.

        Args:
            request: ModelRequest with prompt, optional context, and timeout

        Returns:
            ModelResponse with output and tool_call in metadata when applicable
        """
        base_metadata = {
            "backend": "ollama",
            "model": self.model_name,
            "trace_id": request.trace_id,
        }

        # Use request.system_prompt if provided (Phase Consciousness),
        # otherwise fallback to authoritative _SYSTEM_PROMPT.
        sys_p = request.system_prompt if request.system_prompt else _SYSTEM_PROMPT
        messages = [{"role": "system", "content": sys_p}]

        # Build user message — inject tool results on second call
        if request.context:
            user_content = (
                f"Search results retrieved for your query:\n\n"
                f"{request.context}\n\n"
                f"---\n"
                f"Using the above results, please answer:\n{request.prompt}"
            )
        else:
            user_content = request.prompt

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        url = f"{self.base_url}/api/chat"
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                with httpx.Client(timeout=request.timeout_s) as client:
                    resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                output: str = data.get("message", {}).get("content", "")

                # ── Parse [TOOL_CALL] marker ────────────────────────────────
                metadata = dict(base_metadata)
                tool_call_data, output = _extract_tool_call(output)
                # Fallback: detect loose patterns if [TOOL_CALL] marker was not used
                if not tool_call_data:
                    tool_call_data, output = _try_loose_tool_call(output)
                if tool_call_data:
                    metadata["tool_call"] = tool_call_data

                return ModelResponse(
                    status="success",
                    output=output,
                    metadata=metadata,
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    "Ollama timeout on attempt %d/%d (trace=%s)",
                    attempt + 1, _MAX_RETRIES, request.trace_id,
                )

            except httpx.TransportError as e:
                last_error = e
                logger.warning(
                    "Ollama transport error on attempt %d/%d: %s (trace=%s)",
                    attempt + 1, _MAX_RETRIES, e, request.trace_id,
                )

            except httpx.HTTPStatusError as e:
                # 4xx errors are not retryable (bad request, model not found, etc.)
                logger.error(
                    "Ollama HTTP %d error (not retrying): %s (trace=%s)",
                    e.response.status_code, e, request.trace_id,
                )
                return ModelResponse(
                    status="fatal_error",
                    error_type="backend_unavailable",
                    metadata={**base_metadata, "error": str(e)},
                )

            except Exception as e:
                last_error = e
                logger.error(
                    "Ollama unexpected error on attempt %d/%d: %s (trace=%s)",
                    attempt + 1, _MAX_RETRIES, e, request.trace_id,
                )

            # Exponential backoff before next retry (skip sleep on last attempt)
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                logger.info("Retrying Ollama in %.1fs...", delay)
                time.sleep(delay)

        # All retries exhausted
        error_type = "timeout" if isinstance(last_error, httpx.TimeoutException) else "backend_unavailable"
        logger.error(
            "Ollama failed after %d attempts (trace=%s): %s",
            _MAX_RETRIES, request.trace_id, last_error,
        )
        return ModelResponse(
            status="recoverable_error",
            error_type=error_type,
            metadata={**base_metadata, "error": str(last_error)},
        )
