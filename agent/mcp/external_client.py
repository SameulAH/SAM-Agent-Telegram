"""
MCP External Client — Multi-Provider Semantic Web Search.

Three providers available via Smithery (priority order: Exa → Brave → Linkup):

  Provider   Tool                Best For
  ─────────  ──────────────────  ──────────────────────────────────────────
  Exa        web_search_exa      Neural/semantic search, real-time news
  Brave      brave_web_search    Privacy-first web + news, no browser needed
  Linkup     linkup-search       Real-time facts, source-backed answers

Authentication — Direct mode (simplest):
  EXA_API_KEY      → https://mcp.exa.ai/mcp
  BRAVE_API_KEY    → https://brave.run.tools?braveApiKey=...
  LINKUP_API_KEY   → https://linkup-mcp-server--linkupplatform.run.tools?apiKey=...

Authentication — Smithery Connect proxy mode (credentials stored server-side):
  SMITHERY_API_KEY       → Bearer token (smithery.ai/account/api-keys)
  SMITHERY_NAMESPACE     → e.g. myNamespace
  EXA_CONNECTION_ID      → auto-created by connectivity_test.py
  BRAVE_CONNECTION_ID    → auto-created by connectivity_test.py
  LINKUP_CONNECTION_ID   → auto-created by connectivity_test.py

API keys (get one for free):
  Exa    → https://dashboard.exa.ai/api-keys       (1 000 req/mo free)
  Brave  → https://brave.com/search/api/            (2 000 req/mo free)
  Linkup → https://app.linkup.so/home               (free tier available)

Setup (one-time per provider):
  python agent/mcp/connectivity_test.py
  → Validates keys, optionally creates Smithery connections, prints IDs

Invariants:
- Credentials never logged or stored in state
- Pydantic response validation
- Guardrails: MAX_RESULTS, MAX_SNIPPET_LEN, CHAR_BUDGET
- Trace events: mcp_request_sent, mcp_response_received, tool_execution_failed
- Timeout = 10 s (search APIs are fast; no browser-session overhead)
- Never raises — all errors return MCPResponse(status="error", results=[])
- Handles both JSON and SSE (text/event-stream) MCP transports
"""

import asyncio
import concurrent.futures
import json
import os
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import urlencode

import httpx
from pydantic import BaseModel, Field

from agent.mcp.guardrails import MCPGuardrails


# ──────────────────────────────────────────────────────────────
# SCHEMAS
# ──────────────────────────────────────────────────────────────


class BrowserBaseArgs(BaseModel):
    """Search input — name kept for backward compatibility."""

    query: str = Field(..., min_length=3, description="Search query (min 3 chars)")
    max_results: int = Field(default=3, ge=1, le=5, description="Max results (1–5)")


class BrowserBaseResult(BaseModel):
    """A single search result."""

    title: str
    url: str
    snippet: str


class MCPResponse(BaseModel):
    """Structured response from the MCP server."""

    status: Literal["success", "error"]
    results: List[BrowserBaseResult] = Field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# PROVIDERS
# ──────────────────────────────────────────────────────────────


class SearchProvider(str, Enum):
    """Available search providers (priority order: EXA → BRAVE → LINKUP)."""

    EXA    = "exa"
    BRAVE  = "brave"
    LINKUP = "linkup"
    SEARXNG = "searxng"


# Static configuration per provider
_PROVIDER_CONFIG: Dict[str, Dict] = {
    SearchProvider.EXA: {
        "direct_url":  "https://mcp.exa.ai/mcp",
        "query_param": "exaApiKey",
        "tool_name":   "web_search_exa",
        "args_fn":     lambda q, n: {"query": q, "numResults": n},
    },
    SearchProvider.BRAVE: {
        "direct_url":  "https://brave.run.tools",
        "query_param": "braveApiKey",
        "tool_name":   "brave_web_search",
        "args_fn":     lambda q, n: {"query": q, "count": n},
    },
    SearchProvider.LINKUP: {
        "direct_url":  "https://linkup-mcp-server--linkupplatform.run.tools",
        "query_param": "apiKey",
        "tool_name":   "linkup-search",
        "args_fn":     lambda q, n: {
            "query":      q,
            "outputType": "sourcedAnswer",
            "depth":      "standard",
        },
    },
    SearchProvider.SEARXNG: {
        "direct_url":  os.getenv("SEARXNG_URL", "https://searx.be"),
        "query_param": "",
        "tool_name":   "searxng_search",
        "args_fn":     lambda q, n: {"q": q, "format": "json", "pageno": 1},
    },
}


# ──────────────────────────────────────────────────────────────
# CLIENT
# ──────────────────────────────────────────────────────────────


class MCPClient:
    """
    Multi-provider MCP web-search client.

    Picks the first provider that has valid credentials (Exa → Brave → Linkup)
    and makes a single tools/call JSON-RPC request — no browser session needed.

    Usage:
        client = MCPClient()
        response = await client.search(BrowserBaseArgs(query="AI news today"))

    Guarantees:
    - Never raises (returns MCPResponse(status="error") on any failure)
    - Credentials never logged or put in request body
    - Validates response schema via Pydantic
    - Enforces guardrails via MCPGuardrails
    - Emits trace events if tracer provided
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        smithery_api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        # ── Provider API keys (direct mode) ──────────────────
        exa_api_key: Optional[str] = None,
        brave_api_key: Optional[str] = None,
        linkup_api_key: Optional[str] = None,
        # ── Provider connection IDs (Smithery Connect proxy) ─
        exa_connection_id: Optional[str] = None,
        brave_connection_id: Optional[str] = None,
        linkup_connection_id: Optional[str] = None,
        searxng_url: Optional[str] = None,
        # ── Backward-compat (BrowserBase era) ────────────────
        # api_key maps to exa_api_key; project_id / connection_id silently ignored
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        timeout: float = MCPGuardrails.MCP_TIMEOUT_S,
    ):
        self._base_url_override = base_url   # unit-test hook — bypasses URL building
        self._smithery_key = smithery_api_key or os.getenv("SMITHERY_API_KEY", "")
        self._namespace    = namespace or os.getenv("SMITHERY_NAMESPACE", "")
        self.timeout       = timeout

        # Provider keys — api_key falls back to Exa for backward compatibility
        self._exa_key    = exa_api_key    or api_key or os.getenv("EXA_API_KEY", "")
        self._brave_key  = brave_api_key  or os.getenv("BRAVE_API_KEY", "")
        self._linkup_key = linkup_api_key or os.getenv("LINKUP_API_KEY", "")

        # Smithery Connect connection IDs (one per provider, optional)
        self._exa_conn    = exa_connection_id    or os.getenv("EXA_CONNECTION_ID", "")
        self._brave_conn  = brave_connection_id  or os.getenv("BRAVE_CONNECTION_ID", "")
        self._linkup_conn = linkup_connection_id or os.getenv("LINKUP_CONNECTION_ID", "")
        self._searxng_url = searxng_url or os.getenv("SEARXNG_URL", "https://searx.be")

        # Resolve active provider and URL once at construction time
        self._provider = self._active_provider()
        self.mcp_url   = self._build_url()

    # ── Provider resolution ───────────────────────────────────

    def _active_provider(self) -> Optional[SearchProvider]:
        """Return the first provider that has valid credentials."""
        ph = "your_"

        def ok(v: str) -> bool:
            return bool(v) and not v.startswith(ph)

        def has_proxy(conn: str) -> bool:
            return ok(self._smithery_key) and ok(self._namespace) and ok(conn)

        if ok(self._exa_key)    or has_proxy(self._exa_conn):
            return SearchProvider.EXA
        if ok(self._brave_key)  or has_proxy(self._brave_conn):
            return SearchProvider.BRAVE
        if ok(self._linkup_key) or has_proxy(self._linkup_conn):
            return SearchProvider.LINKUP
        if ok(self._searxng_url):
            return SearchProvider.SEARXNG
        return None

    def _build_url(self) -> str:
        """Compute MCP endpoint URL for the active provider."""
        # Test / direct override — unit tests inject a mock URL here
        if self._base_url_override:
            return self._base_url_override

        if self._provider is None:
            return ""

        # SearXNG is a direct-access provider — no API key or proxy needed
        if self._provider == SearchProvider.SEARXNG:
            return self._searxng_url.rstrip("/")

        cfg  = _PROVIDER_CONFIG[self._provider]
        conn = {
            SearchProvider.EXA:    self._exa_conn,
            SearchProvider.BRAVE:  self._brave_conn,
            SearchProvider.LINKUP: self._linkup_conn,
        }.get(self._provider, "")
        key = {
            SearchProvider.EXA:    self._exa_key,
            SearchProvider.BRAVE:  self._brave_key,
            SearchProvider.LINKUP: self._linkup_key,
        }.get(self._provider, "")

        # Prefer Smithery Connect proxy when a connection ID is available
        if conn and self._namespace and self._smithery_key:
            return f"https://api.smithery.ai/connect/{self._namespace}/{conn}/mcp"

        # Direct mode — API key goes in Authorization header (Bearer token)
        # run.tools endpoints no longer accept the key as a query parameter.
        if key:
            return cfg["direct_url"]

        return cfg["direct_url"]

    def _build_headers(self) -> dict:
        """Auth headers for the active provider.

        Priority:
          1. Smithery Connect proxy  → Authorization: Bearer <smithery_key>
          2. Exa direct (mcp.exa.ai) → x-api-key: <exa_key>
          3. Other direct providers  → Authorization: Bearer <provider_api_key>
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._smithery_key:
            headers["Authorization"] = f"Bearer {self._smithery_key}"
        elif self._provider == SearchProvider.EXA and self._exa_key:
            # mcp.exa.ai uses x-api-key header (not Bearer)
            headers["x-api-key"] = self._exa_key
        elif self._provider is not None:
            direct_key = {
                SearchProvider.BRAVE:  self._brave_key,
                SearchProvider.LINKUP: self._linkup_key,
            }.get(self._provider, "")
            if direct_key:
                headers["Authorization"] = f"Bearer {direct_key}"
        return headers

    def _credentials_present(self) -> bool:
        """True if at least one provider has credentials configured."""
        return self._provider is not None

    # ──────────────────────────────────────────────────────────
    # ASYNC PRIMARY INTERFACE
    # ──────────────────────────────────────────────────────────

    async def search(
        self,
        args: BrowserBaseArgs,
        tracer: Optional[Any] = None,
        trace_metadata: Optional[Any] = None,
    ) -> MCPResponse:
        """
        Semantic web search using the best available provider.

        Provider priority: Exa → Brave → Linkup
        Single tools/call RPC — no browser session required.

        Trace events emitted:
            mcp_request_sent, mcp_response_received, tool_execution_failed
        """
        if not self._credentials_present():
            self._emit_event(tracer, "tool_execution_failed", {
                "reason": "missing_credentials",
                "hint": "Set EXA_API_KEY, BRAVE_API_KEY, or LINKUP_API_KEY in .env",
            }, trace_metadata)
            return MCPResponse(status="error", results=[])

        if not self.mcp_url.startswith("http"):
            self._emit_event(tracer, "tool_execution_failed",
                             {"reason": "invalid_mcp_url"}, trace_metadata)
            return MCPResponse(status="error", results=[])

        cfg       = _PROVIDER_CONFIG[self._provider]
        tool_name = cfg["tool_name"]
        arguments = cfg["args_fn"](args.query, args.max_results)

        self._emit_event(tracer, "mcp_request_sent", {
            "provider":    self._provider,
            "query":       args.query,
            "max_results": args.max_results,
        }, trace_metadata)

        payload = {
            "jsonrpc": "2.0",
            "method":  "tools/call",
            "params":  {"name": tool_name, "arguments": arguments},
            "id":      1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                if self._provider == SearchProvider.SEARXNG:
                    # Native SearXNG GET request
                    response = await client.get(
                        f"{self.mcp_url}/search", params=arguments, headers=self._build_headers()
                    )
                else:
                    # Standard MCP JSON-RPC POST request
                    response = await client.post(
                        self.mcp_url, json=payload, headers=self._build_headers()
                    )
                response.raise_for_status()
                
                # Handle Native SearXNG response vs MCP JSON-RPC
                if self._provider == SearchProvider.SEARXNG:
                    raw_data = response.json()
                    # Convert SearXNG results to standard format
                    search_results = []
                    for r in raw_data.get("results", [])[:args.max_results]:
                        search_results.append(BrowserBaseResult(
                            title=r.get("title", "Result"),
                            url=r.get("url", ""),
                            snippet=r.get("content", "") or r.get("snippet", "")
                        ))
                    return MCPResponse(status="success", results=MCPGuardrails.sanitize_results(search_results))
                
                rpc_data = self._parse_response(response)

        except httpx.TimeoutException:
            self._emit_event(tracer, "tool_execution_failed",
                             {"reason": "timeout", "timeout_s": self.timeout},
                             trace_metadata)
            return MCPResponse(status="error", results=[])

        except httpx.HTTPStatusError as e:
            self._emit_event(tracer, "tool_execution_failed",
                             {"reason": "http_error",
                              "status_code": e.response.status_code},
                             trace_metadata)
            return MCPResponse(status="error", results=[])

        except Exception as e:
            self._emit_event(tracer, "tool_execution_failed",
                             {"reason": "unexpected_error", "error": type(e).__name__},
                             trace_metadata)
            return MCPResponse(status="error", results=[])

        if "error" in rpc_data and "result" not in rpc_data:
            err = rpc_data["error"]
            self._emit_event(tracer, "tool_execution_failed", {
                "reason":  "mcp_rpc_error",
                "code":    err.get("code") if isinstance(err, dict) else None,
                "message": (err.get("message", str(err))
                            if isinstance(err, dict) else str(err)),
            }, trace_metadata)
            return MCPResponse(status="error", results=[])

        raw_results = self._extract_results(rpc_data, args.max_results)
        sanitized   = MCPGuardrails.sanitize_results(raw_results)

        self._emit_event(tracer, "mcp_response_received", {
            "provider":        self._provider,
            "status":          "success",
            "raw_count":       len(raw_results),
            "sanitized_count": len(sanitized),
        }, trace_metadata)

        return MCPResponse(status="success", results=sanitized)

    async def browse_url(
        self,
        url: str,
        instruction: str,
        max_results: int = 3,
        tracer: Optional[Any] = None,
        trace_metadata: Optional[Any] = None,
    ) -> MCPResponse:
        """
        Fetch a specific URL and extract information.

        Uses linkup-fetch when Linkup is active (direct page fetch).
        Falls back to a site-scoped search via Exa or Brave.

        Args:
            url:         Target URL to fetch (e.g. "https://news.google.com").
            instruction: What to extract — used as search context for Exa/Brave.
            max_results: Max items to return (1–5).

        Examples:
            await client.browse_url(
                "https://news.google.com",
                "Top 3 breaking news headlines with title and URL",
            )
            await client.browse_url(
                "https://www.amazon.com/s?k=laptop+stand",
                "Top 3 products with name, price, rating, URL",
            )
        """
        if not self._credentials_present():
            return MCPResponse(status="error", results=[])
        if not self.mcp_url.startswith("http"):
            return MCPResponse(status="error", results=[])

        max_results = max(1, min(max_results, MCPGuardrails.MAX_RESULTS))

        # Linkup has a dedicated page-fetch tool
        if self._provider == SearchProvider.LINKUP:
            payload = {
                "jsonrpc": "2.0",
                "method":  "tools/call",
                "params":  {"name": "linkup-fetch", "arguments": {"url": url}},
                "id":      1,
            }
            self._emit_event(tracer, "mcp_request_sent",
                             {"url": url, "max_results": max_results}, trace_metadata)
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.mcp_url, json=payload, headers=self._build_headers()
                    )
                    response.raise_for_status()
                    rpc_data = self._parse_response(response)
            except Exception:
                return MCPResponse(status="error", results=[])

            raw       = self._extract_results(rpc_data, max_results)
            sanitized = MCPGuardrails.sanitize_results(raw)
            self._emit_event(tracer, "mcp_response_received",
                             {"url": url, "sanitized_count": len(sanitized)},
                             trace_metadata)
            return MCPResponse(
                status="success" if sanitized else "error",
                results=sanitized,
            )

        # Exa / Brave — site-scoped search as fallback
        query_args = BrowserBaseArgs(
            query=f"site:{url} {instruction}",
            max_results=max_results,
        )
        return await self.search(query_args, tracer=tracer,
                                 trace_metadata=trace_metadata)

    # ──────────────────────────────────────────────────────────
    # SYNC WRAPPERS
    # ──────────────────────────────────────────────────────────

    def search_sync(
        self,
        args: BrowserBaseArgs,
        tracer: Optional[Any] = None,
        trace_metadata: Optional[Any] = None,
    ) -> MCPResponse:
        """Synchronous wrapper for search() — safe in both sync and async contexts."""
        return self._run_sync(
            self.search(args, tracer=tracer, trace_metadata=trace_metadata)
        )

    def browse_url_sync(
        self,
        url: str,
        instruction: str,
        max_results: int = 3,
        tracer: Optional[Any] = None,
        trace_metadata: Optional[Any] = None,
    ) -> MCPResponse:
        """Synchronous wrapper for browse_url()."""
        return self._run_sync(
            self.browse_url(url, instruction, max_results,
                            tracer=tracer, trace_metadata=trace_metadata)
        )

    def _run_sync(self, coro) -> MCPResponse:
        """Run an async coroutine synchronously — safe in both sync and async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result(timeout=self.timeout + 5.0)
            else:
                return loop.run_until_complete(coro)
        except Exception:
            return MCPResponse(status="error", results=[])

    # ──────────────────────────────────────────────────────────
    # MCP RESPONSE PARSING
    # ──────────────────────────────────────────────────────────

    def _parse_response(self, response: httpx.Response) -> dict:
        """Parse HTTP response as JSON-RPC — handles JSON and SSE transport."""
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            return self._parse_sse(response.text)
        try:
            return response.json()
        except Exception:
            return self._parse_sse(response.text)

    @staticmethod
    def _parse_sse(text: str) -> dict:
        """Extract the last JSON payload from an SSE stream (data: <json>)."""
        last: dict = {}
        for line in text.splitlines():
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload and payload != "[DONE]":
                    try:
                        last = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
        return last

    def _extract_results(self, rpc_data: dict, max_results: int) -> List[BrowserBaseResult]:
        """
        Convert a successful MCP tools/call response into BrowserBaseResult list.

        Handles response shapes from all three providers:
          Exa    → result.content[0].text = JSON {"results": [{title, url, text}, ...]}
          Brave  → result.content[0].text = JSON {"web": {"results": [{title, url, description}, ...]}}
          Linkup → result.content[0].text = JSON {"answer": "...", "sources": [{name, url, snippet}, ...]}

        MCP JSON-RPC envelope:
            {
              "jsonrpc": "2.0",
              "result": {
                "content": [{"type": "text", "text": "<JSON string>"}],
                "isError": false
              },
              "id": 1
            }
        """
        result = rpc_data.get("result", {})
        if result.get("isError"):
            return []

        for item in result.get("content", []):
            if item.get("type") != "text":
                continue
            text = item.get("text", "").strip()
            if not text:
                continue

            parsed = self._try_parse_json_results(text, max_results)
            if parsed is not None:
                return parsed[:max_results]

            # mcp.exa.ai plain-text format:
            #   Title: ...\nPublished Date: ...\nURL: ...\nText: ...\n\n
            exa_parsed = self._try_parse_exa_text_results(text, max_results)
            if exa_parsed:
                return exa_parsed

            # Generic plain-text fallback (last resort)
            return [BrowserBaseResult(
                title="Search Result",
                url="",
                snippet=text[:MCPGuardrails.MAX_SNIPPET_LEN],
            )]

        return []

    @staticmethod
    def _try_parse_exa_text_results(
        text: str,
        max_results: int,
    ) -> List[BrowserBaseResult]:
        """
        Parse the plain-text format returned by mcp.exa.ai/mcp:

            Title: <title>
            Published Date: <date>
            URL: <url>
            Text: <body text>

        Multiple results are separated by blank lines or by the next 'Title:' block.
        """
        import re as _re
        results: List[BrowserBaseResult] = []

        # Split on double newlines or on lines that start a new Title: block
        blocks = _re.split(r'\n(?=Title:)', text.strip())

        for block in blocks:
            if not block.strip():
                continue
            title_m = _re.search(r'^Title:\s*(.+)', block, _re.MULTILINE)
            url_m   = _re.search(r'^URL:\s*(https?://\S+)', block, _re.MULTILINE)
            text_m  = _re.search(r'^Text:\s*(.+)', block, _re.MULTILINE | _re.DOTALL)

            if not title_m:
                continue

            title   = title_m.group(1).strip()
            url     = url_m.group(1).strip() if url_m else ""
            snippet = text_m.group(1).strip()[:MCPGuardrails.MAX_SNIPPET_LEN] if text_m else ""

            results.append(BrowserBaseResult(title=title, url=url, snippet=snippet))
            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _try_parse_json_results(
        text: str,
        max_results: int,
    ) -> Optional[List[BrowserBaseResult]]:
        """
        Parse provider JSON into BrowserBaseResult list.

        Supports:
          Exa:    {"results": [{title, url, text/snippet}, ...]}
          Brave:  {"web": {"results": [{title, url, description}, ...]}}
          Linkup: {"answer": "...", "sources": [{name, url, snippet}, ...]}
          Generic arrays and single-object dicts
        """
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        items: List[dict] = []

        if isinstance(data, list):
            items = data

        elif isinstance(data, dict):
            # Exa: top-level "results" array
            if "results" in data:
                items = data["results"]

            # Brave: nested web.results
            elif "web" in data:
                items = data["web"].get("results", [])

            # Linkup: "sources" array (preferred over plain "answer")
            elif "sources" in data:
                items = data["sources"]

            # Linkup: plain "answer" with no sources
            elif "answer" in data:
                items = [{"title": "Answer", "url": "", "snippet": data["answer"]}]

            # Generic fallbacks
            else:
                items = (
                    data.get("items")
                    or data.get("organic")
                    or data.get("webPages", {}).get("value", [])
                    or [data]
                )
        else:
            return []

        out: List[BrowserBaseResult] = []
        for r in items[:max_results]:
            if not isinstance(r, dict):
                continue
            title   = r.get("title")   or r.get("name")    or "Result"
            url     = r.get("url")     or r.get("link")    or r.get("href") or ""
            snippet = (
                r.get("snippet")
                or r.get("description")
                or r.get("text")
                or r.get("body")
                or r.get("content")
                or r.get("summary")
                or ""
            )
            out.append(BrowserBaseResult(
                title=str(title),
                url=str(url),
                snippet=str(snippet),
            ))

        return out

    # ──────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _emit_event(
        tracer: Optional[Any],
        event_name: str,
        metadata: dict,
        trace_metadata: Optional[Any],
    ) -> None:
        """Safely emit a trace event. Never raises."""
        if tracer is None or trace_metadata is None:
            return
        try:
            tracer.record_event(event_name, metadata, trace_metadata)
        except Exception:
            pass
