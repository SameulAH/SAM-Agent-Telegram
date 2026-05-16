# SAM — Stateful Agent Model

[![CI](https://github.com/SameulAH/SAM-Agent-Telegram/actions/workflows/ci.yml/badge.svg)](https://github.com/SameulAH/SAM-Agent-Telegram/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph 1.x](https://img.shields.io/badge/langgraph-1.x-orange.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/Dockerfile.agent)

SAM is a **production-grade stateful personal AI agent** built on LangGraph. It receives messages from Telegram (and WhatsApp), runs them through a deterministic 15-node execution graph, reasons using a local LLM (Ollama), searches the web when needed, and remembers facts about the user across all sessions — permanently.

The system is explicitly designed for **operational reliability over capability breadth**: every failure mode is typed, every memory operation is non-fatal, and every routing decision is deterministic and auditable.

---

## Table of Contents

1. [Design Philosophy](#1-design-philosophy)
2. [Architecture Decision Records](#2-architecture-decision-records)
3. [Phase Architecture](#3-phase-architecture)
4. [Infrastructure Overview](#4-infrastructure-overview)
5. [Service Catalogue](#5-service-catalogue)
6. [Message Flow — End to End](#6-message-flow--end-to-end)
7. [The Agent Graph — Node by Node](#7-the-agent-graph--node-by-node)
8. [AgentState — The Central Contract](#8-agentstate--the-central-contract)
9. [Memory Architecture](#9-memory-architecture)
10. [Tool Execution (MCP Web Search)](#10-tool-execution-mcp-web-search)
11. [Speech Pipeline (STT / TTS)](#11-speech-pipeline-stt--tts)
12. [Autonomous Heartbeat](#12-autonomous-heartbeat)
13. [Observability & Tracing](#13-observability--tracing)
14. [Design Principles](#14-design-principles)
15. [Testing Strategy](#15-testing-strategy)
16. [Known Limitations](#16-known-limitations)
17. [API Reference](#17-api-reference)
18. [Configuration Reference](#18-configuration-reference)
19. [Deployment Guide](#19-deployment-guide)
20. [Project Structure](#20-project-structure)
21. [Glossary](#21-glossary)
22. [Contributing Guide](#22-contributing-guide)
23. [Prompt Engineering & Token Budget](#23-prompt-engineering--token-budget)
24. [Performance Profile](#24-performance-profile)
25. [Security Model](#25-security-model)
26. [Local Development & Troubleshooting](#26-local-development--troubleshooting)

---

## 1. Design Philosophy

SAM is built around four engineering commitments that drove every architectural decision:

### 1.1 Determinism over intelligence
Control flow is never delegated to the LLM. All routing decisions — whether to read memory, whether to call a tool, what to do after a tool result — live in a single `decision_logic_node` using rule-based logic and explicit state flags. This makes the agent's behavior fully auditable and testable without running a real LLM.

### 1.2 Non-fatal failures everywhere
Memory unavailability, tracing errors, tool timeouts, and LLM failures all return typed error responses rather than raising exceptions. The agent degrades gracefully: if Qdrant is down, the conversation continues without long-term context. If Whisper fails, the voice message is acknowledged but not processed. Nothing crashes the serving process.

### 1.3 Advisory memory, not authoritative memory
Memory informs the agent's responses but never controls its behavior. Retrieved facts are injected into the prompt as context — they do not gate routing decisions, change the command sequence, or alter output guardrails. The LLM decides how to use memory; the graph decides whether to retrieve it.

### 1.4 Swappable backends behind stable interfaces
Every external dependency (LLM, STM, LTM, STT, TTS, tracer) is accessed through an abstract interface with at least one stub implementation. Any backend can be replaced — or mocked with the stub — without changing orchestration code. This enables fully deterministic CI runs with zero external services.

---

## 2. Architecture Decision Records

These are the explicit design choices made, the alternatives considered, and the rationale.

### ADR-001: LangGraph for orchestration

**Decision:** Use LangGraph `StateGraph` as the execution framework.

**Alternatives considered:**
- **Plain Python state machine** — gives full control but requires building retry, streaming, and async execution from scratch
- **LangChain AgentExecutor** — designed for ReAct loops; hides the execution graph, making debugging and testing hard
- **Custom async pipeline** — flexible but no built-in state persistence or conditional edge support

**Rationale:** LangGraph exposes the graph structure explicitly, supports typed state objects (`AgentState`), and allows both sync and async node execution. The `ainvoke()` interface integrates cleanly with FastAPI's async runtime. The node-as-function model maps directly to the single-responsibility principle: each node does one thing.

---

### ADR-002: Deterministic memory intent detection (Phase DMA)

**Decision:** Use precompiled regex patterns to detect memory read/write intent — zero LLM calls for this decision.

**Alternatives considered:**
- **LLM-based intent classification** — higher accuracy, but adds 3–8 seconds latency before the main model call
- **Rule-based keyword matching** — simpler, but produces false positives on substrings (e.g. "that" inside "capital of France")
- **Fine-tuned classifier model** — best accuracy, but requires labelled data and a separate inference service

**Rationale:** For a personal assistant, recall on personal-fact patterns (birth date, name, location) is more important than precision. The regex patterns are word-boundary anchored (`\b...\b`) to reduce false positives. Precompiling at class load time (vs per-call `re.search`) eliminates repeated compilation overhead. LLM cost for intent detection is disproportionate given the pattern is narrow and well-defined.

---

### ADR-003: Two-tier memory (SQLite STM + Qdrant LTM)

**Decision:** Use SQLite for session context (short-term memory) and Qdrant for personal facts (long-term memory).

**Alternatives considered:**
- **Single vector DB for both** — simpler schema, but semantic search on raw conversation turns is noisy and expensive
- **Redis for STM** — faster, but adds a network dependency for a single-instance deployment; SQLite WAL mode provides comparable reliability for the write rates we see
- **PostgreSQL for STM** — better for scale, but heavy operational overhead for what is essentially a key-value upsert per turn
- **Pinecone or Weaviate for LTM** — managed services reduce ops burden, but introduce external dependency and cost

**Rationale:** STM is accessed by exact key (`conversation_id + key`) — a hash lookup, not a similarity search. SQLite with WAL mode provides crash-safe, low-latency upsert with zero infrastructure. LTM is accessed by semantic similarity (what did the user say they like?) — a vector database is the correct primitive. Qdrant runs locally in Docker, is free, and supports append-only writes that preserve the full fact history.

---

### ADR-004: Single-pass LLM architecture

**Decision:** The LLM is called at most twice per turn — once optionally (pre-tool, if a tool is force-triggered by heuristics) and once for the main reply. The model never loops back to request additional tools.

**Alternatives considered:**
- **ReAct-style tool-use loop** — agent calls tools multiple times in one turn based on observation; allows richer reasoning but risks infinite loops and unpredictable latency
- **Tool-calling API (OpenAI function calling)** — clean interface, but ties the system to a specific API; Ollama models have inconsistent function-calling support

**Rationale:** For a personal assistant responding on Telegram, predictable latency is more important than deep multi-tool reasoning. A single tool call per turn covers > 90% of queries (web search for current data). The guard `MAX_TOOL_CALLS_PER_TURN = 1` is enforced in both `decision_logic_node` and `tool_execution_node` (belt and suspenders). Latency is bounded: two Ollama calls + one search ≈ 20–40 seconds on CPU.

---

### ADR-005: Local LLM (Ollama) over API

**Decision:** Use Ollama running locally for inference, not a hosted API (OpenAI, Anthropic, etc.).

**Alternatives considered:**
- **OpenAI GPT-4o / GPT-4o-mini** — best quality, simple API, but cost per message is non-trivial for a personal assistant used daily; data sent to a third party
- **Anthropic Claude via API** — similar trade-offs to OpenAI
- **Groq** — extremely fast inference for open models, but adds an external dependency; data leaves the device

**Rationale:** This is a personal assistant handling private biographical data (name, location, preferences, schedule). Keeping inference local means no data leaves the machine. Ollama `phi3:latest` runs adequately on CPU and excellently on GPU, with acceptable latency for asynchronous messaging (Telegram does not require sub-second response times).

---

### ADR-006: Tracing as a passive observer

**Decision:** The tracing layer is strictly read-only with respect to agent state. Tracing failures are silently caught and never propagate.

**Rationale:** Observability infrastructure must never be a single point of failure for agent execution. If LangSmith is down, the user should still get a reply. The `Tracer` interface formally documents this contract: all methods `MUST NOT raise exceptions`, `MUST NOT affect control flow`, `MUST NOT modify agent state`. This is enforced by the abstract interface definition and tested by `tests/observability/test_tracing_failure_safety.py`.

---

## 3. Phase Architecture

The system was built incrementally. Each phase added capabilities without breaking prior ones. Understanding the phases explains the naming conventions throughout the codebase.

| Phase | Name | What it added |
|-------|------|--------------|
| 1 | **Skeleton** | Core LangGraph graph, router, state init, decision logic, preprocessing, model call, error router, format response |
| 2 | **Short-Term Memory** | SQLite STM, `memory_read_node`, `memory_write_node`, memory authorization flags |
| 3.2 | **Long-Term Memory** | Qdrant LTM, `long_term_memory_read_node`, `long_term_memory_write_node` |
| DMA | **Deterministic Memory Access** | `memory_access_decision_node`, `fact_extraction_node`, `write_authorization_node` — rule-based intent detection, zero LLM cost |
| PA | **Personal Awareness** | Personal fact extraction from user input (regex + confidence scoring), biographical data pipeline |
| MCP | **Model Context Protocol** | `tool_execution_node`, multi-provider web search (Exa → Brave → Linkup → SearXNG), tool context injection |
| 5 | **Freshness Tuning** | Pruned over-broad freshness keywords that caused unnecessary tool calls (removed: `now`, `currently`, `update`, `updates`, `live`) |
| Consciousness | **Reflection** | `reflection_node` background insight extraction after every turn, written to Qdrant LTM |
| Humanizing | **Persistent Context** | `requires_memory_read = True` always (agent always knows when it last spoke to you); `requires_memory_write` determined by DMA pattern detection |
| Additive | **Observability** | `execution_context` in AgentState; node-level tracing with `start_span` / `end_span` |

**Phase naming in code:** state fields, node names, and comments explicitly tag which phase introduced them. For example, `# Phase DMA` appears in `state_schema.py` at line 73. This makes git blame meaningful — you can trace any feature to its introduction phase.

---

## 4. Infrastructure Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             EXTERNAL WORLD                                  │
│                                                                             │
│   Telegram ──────────────────────────────── WhatsApp                       │
│      │  webhook POST /webhook/telegram            │ webhook POST            │
└──────┼─────────────────────────────────────────── ┼─────────────────────────┘
       │                                            │
       ▼                                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ngrok static tunnel  :443                                 │
│         https://holden-archeological-kinsley.ngrok-free.app                 │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       sam-agent container  :8000                             │
│                                                                              │
│  FastAPI application (agent/api.py — production entry point)                │
│  ├── /webhook/telegram  ──► TelegramWebhookHandler                          │
│  │     dedup (TTLCache 5000/5min) + rate limit (3req/5s/user)               │
│  ├── /webhook/whatsapp  ──► WhatsAppWebhookHandler                          │
│  │     HMAC-SHA256 signature verification                                   │
│  └── /invoke            ──► Direct agent invocation                         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │              SAMAgentOrchestrator  (LangGraph StateGraph)              │ │
│  │                                                                        │ │
│  │  15 nodes · deterministic routing · single-pass LLM · non-fatal       │ │
│  │                                                                        │ │
│  │  AgentState  ─────────────────────────────────────────────────────    │ │
│  │  (100 fields: identity, input, processing, memory, tool, output)      │ │
│  └─────┬─────────────┬──────────────┬───────────────┬────────────────────┘ │
│        │             │              │               │                       │
│        ▼             ▼              ▼               ▼                       │
│   sam-agent-    sam-agent-     MCP Web         LangSmith                   │
│   ollama        qdrant         Search          Tracer                      │
│   :11434        :6333          (Exa/Brave/      (remote)                   │
│   (phi3:latest) (LTM)          Linkup/          or OTel                    │
│                                SearXNG)         collector                  │
│                                                 :4317                      │
│        │             │                                                     │
│        ▼             ▼                                                     │
│  SQLite (STM)  Qdrant (LTM)                                               │
│  /app/data/    long_term_                                                  │
│  memory.db     memory                                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Service Catalogue

| Service | Image | Port(s) | Role | Restart policy |
|---------|-------|---------|------|---------------|
| `sam-agent` | Custom (`docker/Dockerfile.agent`) | **8000** | Core agent — FastAPI + LangGraph | `unless-stopped` |
| `sam-agent-ollama` | `ollama/ollama:latest` | 11434 (internal) | LLM inference | `unless-stopped` |
| `sam-agent-qdrant` | `qdrant/qdrant:latest` | **6333** | Vector DB — long-term memory | `unless-stopped` |
| `sam-agent-searxng` | `searxng/searxng:latest` | **8888** | Self-hosted metasearch (free fallback) | `unless-stopped` |
| `otel-collector` | `otel/opentelemetry-collector:latest` | 4317 (gRPC), 4318 (HTTP) | Trace aggregation | (as-needed) |
| `jaeger` | `jaegertracing/all-in-one:latest` | **16686** (UI), 14250 | Trace visualisation | (as-needed) |

**Network:** all containers share `sam_network` (bridge). Inter-service DNS uses container names (`http://ollama:11434`, `http://qdrant:6333`).

**Volumes:** `sqlite_data`, `ollama_data`, `qdrant_data`, `whisper_cache`, `coqui_cache`, `searxng_data`.

### Entry Points

| File | Mode | Command |
|------|------|---------|
| `agent/api.py` | **Production** (Docker) | `python -m agent.api --host 0.0.0.0 --port 8000` |
| `main.py` | **Development** (local) | `uvicorn main:app --reload --host 0.0.0.0 --port 8000` |

---

## 6. Message Flow — End to End

### Full journey: Telegram message → SAM reply

```
User types "What's the BTC price?"
         │
         │  Telegram webhook POST
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  webhook/telegram.py                                         │
 │                                                              │
 │  1. Deduplication check                                      │
 │     TTLCache(maxsize=5000, ttl=300s) on update_id           │
 │     → duplicate? silently ACK 200, drop                      │
 │                                                              │
 │  2. Rate limit check                                         │
 │     TTLCache per user_id                                     │
 │     default: 3 requests / 5 seconds / user                  │
 │     → over limit? ACK 200, drop silently (no error to user)  │
 │                                                              │
 │  3. Payload validation                                       │
 │     → no message / no text / no voice? ACK 200, skip        │
 │                                                              │
 │  4. Return HTTP 200 IMMEDIATELY                              │
 │     Telegram retries if it doesn't get 200 in <10s          │
 │                                                              │
 │  5. BackgroundTask: _process_text_async()                    │
 │     (all heavy work runs here, after 200 is already sent)   │
 └──────────────────────────────────────────────────────────────┘
         │
         │  NormalizedMessage(platform, user_id, chat_id, content, ...)
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  agent/orchestrator.py — SAMOrchestrator.invoke()           │
 │                                                              │
 │  Instantiates: OllamaModelBackend                           │
 │                SQLiteShortTermMemoryStore                    │
 │                QdrantLongTermMemoryStore                     │
 │                LangSmithTracer                               │
 │                                                              │
 │  Calls SAMAgentOrchestrator.invoke(raw_input=...)           │
 │  (awaits graph completion)                                   │
 │                                                              │
 │  After reply: schedules delayed_reflect() background task   │
 │  (5s delay, asyncio.create_task)                            │
 └──────────────────────────────────────────────────────────────┘
         │
         │  awaits graph.ainvoke(initial_AgentState)
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  LangGraph — 15-node deterministic DAG                       │
 │  (see Section 7 for node-by-node breakdown)                 │
 │                                                              │
 │  Key path for "BTC price":                                   │
 │  router → state_init → decision("preprocess")               │
 │  → task_preprocessing                                        │
 │  → memory_access_decision(write=F, read=T)                  │
 │  → memory_read(SQLite)                                       │
 │  → decision("long_term_memory_read")                        │
 │  → long_term_memory_read(Qdrant)                            │
 │  → decision("execute_tool") ← "price" keyword detected      │
 │  → tool_execution(Exa search: "BTC price")                  │
 │  → decision("call_model")                                   │
 │  → model_call(Ollama, with tool_context injected)           │
 │  → decision("memory_write")                                 │
 │  → memory_write(SQLite)                                     │
 │  → decision("long_term_memory_write")                       │
 │  → long_term_memory_write(Qdrant)                           │
 │  → decision("format")                                       │
 │  → format_response(≤800 chars, ≤5 sentences)                │
 │  → END                                                       │
 └──────────────────────────────────────────────────────────────┘
         │
         │  final_output: "Bitcoin is trading at $67,420 as of..."
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Response delivery                                           │
 │                                                              │
 │  lines = [l for l in response.splitlines() if l.strip()]   │
 │                                                              │
 │  if len(lines) > 5:                                         │
 │      → gTTS/Coqui: text → OGG → send_voice_message()       │
 │  else:                                                       │
 │      → transport.send_response(chat_id, text)               │
 └──────────────────────────────────────────────────────────────┘
         │
         │  [5 seconds later, in background]
         ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  reflection_node (consciousness)                            │
 │  → Ollama: extract insights from this turn                  │
 │  → Qdrant: append new facts to LTM                          │
 └──────────────────────────────────────────────────────────────┘
```

---

## 7. The Agent Graph — Node by Node

### Topology

```
START
  │
  ▼
router_node           ← classify modality (text / audio / image)
  │
  ▼
state_init_node       ← lock in conversation_id + trace_id (immutable)
  │
  ▼
decision_logic_node ◄──────────────────────────────────────────────┐
  │   (the SOLE routing authority — called multiple times per turn) │
  │                                                                 │
  ├── "preprocess" ──────────────────────────────────────────────► │
  │     task_preprocessing_node                                     │
  │       │                                                         │
  │       ▼                                                         │
  │     memory_access_decision_node  ─────── "fact_extraction" ──► │
  │       │                                        │                │
  │       │ "memory_read"                          ▼                │
  │       │                              fact_extraction_node       │
  │       │                                        │                │
  │       │                              write_authorization_node   │
  │       │                                        │                │
  │       │◄───────────────────────────────────────┘                │
  │       ▼                                                         │
  │     memory_read_node (SQLite STM) ──────────────────────────► ─┤
  │     long_term_memory_read_node (Qdrant LTM) ────────────────► ─┤
  │                                                                 │
  ├── "execute_tool" ────────────────────────────────────────────► │
  │     tool_execution_node (Exa/Brave/Linkup/SearXNG)            │
  │                                                                 │
  ├── "call_model" ──────────────────────────────────────────────► │
  │     model_call_node (Ollama)                                   │
  │       ├── success ─────────────────────────────────────────── ─┤
  │       └── failure → error_router_node ──────────────────────► │
  │                                                                 │
  ├── "memory_write" ─────────────────────────────────────────── ──┤
  │     memory_write_node (SQLite STM upsert)                      │
  │                                                                 │
  ├── "long_term_memory_write" ───────────────────────────────── ──┤
  │     long_term_memory_write_node (Qdrant append)                │
  │                                                                 │
  └── "format" ──────────────────────────────────────────────────► │
        format_response_node                                        │
          │                                                         │
          ▼                                                         │
         END                                                        │
                                                                    │
         [background, after END]                                    │
         reflection_node ──────────────────────────────────────────┘
```

### Node Reference

| Node | Phase | Responsibility | Writes to state | Must NOT |
|------|-------|---------------|-----------------|---------|
| `router_node` | 1 | Classify input modality | `input_type` | Call model, access memory |
| `state_init_node` | 1 | Lock in identity fields | `conversation_id`, `trace_id` | Overwrite existing IDs |
| `decision_logic_node` | 1 | Emit next `command` | `command`, (some memory flags) | Execute tasks, call model |
| `task_preprocessing_node` | 1 | Normalise raw input | `preprocessing_result` | Branch logic |
| `memory_access_decision_node` | DMA | Detect write/read intent (regex) | `requires_memory_write`, `requires_memory_read`, `memory_read_authorized` | Call LLM |
| `fact_extraction_node` | DMA/PA | Extract personal facts (regex + confidence ≥ 0.7) | `extracted_facts` | Write to memory |
| `write_authorization_node` | DMA | Validate facts against guardrail limits | `memory_write_authorized`, `write_authorization_checked` | Execute writes |
| `memory_read_node` | 2 | Read conversation context from SQLite | `memory_read_result`, `memory_read_status` | Raise on failure |
| `long_term_memory_read_node` | 3.2 | Retrieve facts from Qdrant | `long_term_memory_read_result`, `long_term_memory_read_status` | Influence routing |
| `model_call_node` | 1 | Call Ollama LLM | `model_response`, `model_metadata`, `tool_call` | Write memory, set command |
| `tool_execution_node` | MCP | Execute web search, format results | `tool_result`, `tool_context`, `tool_executed`, `tool_call_count` | Write memory, set command |
| `memory_write_node` | 2 | Upsert conversation context in SQLite | `memory_write_status` | Raise on failure |
| `long_term_memory_write_node` | 3.2 | Append extracted facts to Qdrant | `long_term_memory_write_status` | Delete/update existing facts |
| `error_router_node` | 1 | Handle model failures | `final_output`, `error_type` | Crash process |
| `format_response_node` | 1 | Apply output guardrails, set final reply | `final_output` | Truncate mid-sentence |
| `reflection_node` | Consciousness | Background insight extraction | `reflections` (+ Qdrant write) | Block main response |

### Output Guardrails (format_response_node)

Applied in order — sentences first, then characters:

```python
MAX_OUTPUT_SENTENCES: int = 5    # soft ceiling — truncate to 5 sentences
MAX_OUTPUT_CHARS: int = 800      # hard ceiling — truncate at sentence boundary
```

---

## 8. AgentState — The Central Contract

`AgentState` is a Python `dataclass` that flows through the entire graph. It is the single source of truth for all execution context. Understanding its invariants is essential for extending the agent.

### Invariants (from `agent/state_schema.py`)

```
1. conversation_id and trace_id are IMMUTABLE once set by state_init_node.
   No downstream node may overwrite them.

2. preprocessing_result, model_response, and final_output are written
   only by their designated nodes (task_preprocessing_node, model_call_node,
   format_response_node respectively).

3. command is the ONLY control flow signal. Only decision_logic_node
   writes it. No other node sets the command field.

4. error_type is set ONLY by error_router_node.

5. Memory fields (memory_read_result, long_term_memory_read_result, etc.)
   store pointers and metadata — never raw knowledge. The LLM decides
   how to interpret memory content; the graph never reads it.

6. long_term_memory_* fields are advisory only. Their content never
   influences routing decisions.

7. tool_execution_node NEVER writes to memory_* fields.
   tool_execution_node NEVER sets command.
```

### State Field Map

```
AgentState
│
├── Identity (immutable after state_init_node)
│   ├── conversation_id: str         e.g. "telegram_903341171"
│   ├── trace_id: str                e.g. "f0132022-970e-..."
│   └── created_at: str              ISO timestamp
│
├── Input
│   ├── input_type: str              "text" | "audio" | "image"
│   └── raw_input: str               original user message
│
├── Processing
│   └── preprocessing_result: str    cleaned/normalised input
│
├── Model
│   ├── model_response: ModelResponse  {status, output, error_type, metadata}
│   └── model_metadata: Dict           model-specific metadata
│
├── Output
│   ├── final_output: str            guardrail-enforced reply
│   ├── error_type: str              set only by error_router_node
│   └── persona_name: str            defaults to "SAM"
│
├── Control
│   └── command: str                 preprocess|call_model|execute_tool|
│                                    memory_read|memory_write|
│                                    long_term_memory_read|long_term_memory_write|
│                                    format|end
│
├── Short-Term Memory (Phase 2)
│   ├── memory_available: bool
│   ├── memory_read_authorized: bool
│   ├── memory_write_authorized: bool
│   ├── memory_read_result: Dict
│   ├── memory_read_status: str      None|success|failed|not_found
│   └── memory_write_status: str
│
├── Long-Term Memory (Phase 3.2)
│   ├── long_term_memory_requested: bool
│   ├── long_term_memory_status: str     "available"|"unavailable"
│   ├── long_term_memory_read_result: Dict
│   ├── long_term_memory_read_status: str
│   └── long_term_memory_write_status: str
│
├── Deterministic Memory Access (Phase DMA)
│   ├── requires_memory_write: bool   declarative fact detected
│   ├── requires_memory_read: bool    always True (Phase Humanizing)
│   ├── extracted_facts: List         personal facts for LTM write
│   └── write_authorization_checked: bool
│
├── Tool Execution (Phase MCP)
│   ├── tool_executed: bool           True after tool_execution_node
│   ├── tool_call_count: int          increments per execution (max 1)
│   ├── tool_call: Dict               pending call {name, arguments}
│   ├── tool_result: Dict             raw tool response
│   └── tool_context: str             formatted, injection-safe results
│
├── Consciousness (Phase Consciousness)
│   └── reflections: List[Dict]       insights learned this turn
│
└── Observability (Phase Additive)
    └── execution_context: AgentExecutionContext
```

### Validation

`AgentState.__post_init__()` enforces four hard constraints at instantiation:

```python
- conversation_id must not be empty
- trace_id must not be empty
- input_type must be "text", "audio", or "image"
- raw_input must not be empty
```

These raise `ValueError` immediately — invalid states never enter the graph.

---

## 9. Memory Architecture

### Two-tier design rationale

| Concern | Short-Term Memory | Long-Term Memory |
|---------|-----------------|-----------------|
| Backend | SQLite (WAL mode) | Qdrant (vector DB) |
| Scope | Per conversation session | Cross-session, permanent |
| Access pattern | Exact key lookup | Semantic similarity search |
| TTL | 7 days (configurable) | Append-only, no expiry |
| Written by | `memory_write_node` | `long_term_memory_write_node` + `reflection_node` |
| Read by | `memory_read_node` | `long_term_memory_read_node` |
| Failure mode | Returns `status="unavailable"` | Returns empty results |
| Purpose | Conversation continuity | Personal knowledge graph |

### STM — Schema

```sql
CREATE TABLE short_term_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    key             TEXT    NOT NULL,
    data            TEXT    NOT NULL,          -- JSON blob
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, key)               -- upsert target
);

CREATE INDEX idx_conversation_key
    ON short_term_memory(conversation_id, key);

PRAGMA journal_mode = WAL;   -- concurrent reads during writes
PRAGMA synchronous = FULL;   -- fsync on commit — crash safe
```

**Eviction:** on every `write()`, rows where `(now - updated_at) > STM_TTL_SECONDS` (default 604800s / 7 days) are deleted in the same transaction. Non-fatal — silently skipped on error.

### LTM — Schema

```
Qdrant collection: long_term_memory
Vector size: 384 dimensions
Storage: append-only (no updates, no deletes)

Payload fields per vector:
  conversation_id: str     — scope identifier
  fact_type: str           — "personal_fact" | "preference" | "goal" | "mood"
  content: str             — fact text
  confidence: float        — extraction confidence (0.0–1.0)
  created_at: str          — ISO timestamp
  source: str              — "fact_extraction" | "reflection"
```

### Personal Fact Extraction (Phase DMA + PA)

The `fact_extraction_node` uses 13 precompiled write patterns to detect personal facts with zero LLM cost:

```python
_WRITE_PATTERNS = (
    re.compile(r"\bi\s+(?:currently\s+)?live\s+in\b",      re.IGNORECASE),
    re.compile(r"\bmy\s+name\s+is\b",                       re.IGNORECASE),
    re.compile(r"\bi\s+work\s+(?:as|at|for)\b",             re.IGNORECASE),
    re.compile(r"\bmy\s+(?:favorite|favourite)\b",          re.IGNORECASE),
    re.compile(r"\bi\s+am\s+from\b",                        re.IGNORECASE),
    re.compile(r"\bi\s+was\s+born\s+in\b",                  re.IGNORECASE),
    re.compile(r"\bi\s+prefer\b",                           re.IGNORECASE),
    re.compile(r"\bi\s+(?:usually|always|never)\b",         re.IGNORECASE),
    re.compile(r"\bi\s+use\b",                              re.IGNORECASE),
    re.compile(r"\bcall\s+me\b",                            re.IGNORECASE),
    re.compile(r"\bmy\s+(?:birthday|birthdate)\s+is\b",     re.IGNORECASE),
    re.compile(r"\bi\s+am\s+(?:a|an)\b",                   re.IGNORECASE),
    re.compile(r"\bi\s+study\b",                            re.IGNORECASE),
)
```

Facts below **confidence 0.7** are discarded before write authorization.

### Retrieval Intent Detection (Phase DMA)

12 precompiled read patterns detect when the user is referencing past context:

```python
_READ_PATTERNS = (
    re.compile(r"\bwhat\s+did\s+i\b",          re.IGNORECASE),
    re.compile(r"\bwhere\s+do\s+i\b",           re.IGNORECASE),
    re.compile(r"\bwhere\s+did\s+i\b",          re.IGNORECASE),
    re.compile(r"\byou\s+said\s+earlier\b",     re.IGNORECASE),
    re.compile(r"\bas\s+i\s+mentioned\b",       re.IGNORECASE),
    re.compile(r"\bremind\s+me\b",              re.IGNORECASE),
    re.compile(r"\bmy\s+last\b",                re.IGNORECASE),
    re.compile(r"\bdo\s+you\s+remember\b",      re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:is|are)\s+my\b",  re.IGNORECASE),
    re.compile(r"\btell\s+me\s+(?:about\s+)?my\b", re.IGNORECASE),
    re.compile(r"\bwho\s+am\s+i\b",            re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+my\s+name\b",   re.IGNORECASE),
)
```

---

## 10. Tool Execution (MCP Web Search)

### Trigger heuristics (in `decision_logic_node`)

```python
query_lower = preprocessing_result.lower()

has_financial  = any(x in query_lower for x in
                     ["price", "btc", "eth", "$", "coin", "stock", "market"])
has_freshness  = any(kw in query_lower for kw in _FRESHNESS_KEYWORDS)
has_info_intent = any(x in query_lower for x in
                      ["news", "update", "happened", "weather", "score", "latest"])
```

**Freshness keywords (Phase 5 — curated):**
```python
_FRESHNESS_KEYWORDS = frozenset({
    "today", "latest", "recent", "breaking", "news",
    "right now", "this week", "this month", "current events",
})
```

*Removed in Phase 5 for being too broad (caused unnecessary tool calls adding ~20s latency):*
`"now"`, `"currently"`, `"update"`, `"updates"`, `"live"`

### Provider cascade

```
tool_execution_node
    │
    ├─ 1. Exa          EXA_API_KEY set?       → neural/semantic, real-time news
    ├─ 2. Brave        BRAVE_API_KEY set?      → privacy-first web + news
    ├─ 3. Linkup       LINKUP_API_KEY set?     → real-time facts, source citations
    └─ 4. SearXNG      SEARXNG_URL set         → free self-hosted metasearch
         (always available via internal container at http://searxng:8080)

Returns on first successful provider.
Never raises — all failures return MCPResponse(status="error", results=[])
```

### Guardrails (Phase 4 values — from `agent/mcp/guardrails.py`)

| Constant | Value | Phase 4 change |
|---------|-------|---------------|
| `MAX_TOOL_CALLS_PER_TURN` | **1** | — (always was 1) |
| `MAX_RESULTS` | **3** | Reduced from 5 |
| `MAX_SNIPPET_LEN` | **200 chars** | Reduced from 300 |
| `MAX_TOTAL_CHARS` | **800 chars** | Reduced from 1500 |
| `MCP_TIMEOUT_S` | **15.0 seconds** | (Exa live-crawl can take 8–12s) |

*Phase 4 rationale: smaller payload = faster second model pass. The previous 1500-char budget was the #2 latency contributor.*

### Tool call format

Models emit tool calls using the `[TOOL_CALL]` marker:

```
[TOOL_CALL]{"name": "web_search", "arguments": {"query": "bitcoin price USD"}}
```

The parser also handles phi3:mini's shorthand:

```
[Web_Search]{"query": "bitcoin price USD"}
```

Fallback strategies if neither marker is present:
1. Raw structured JSON: `{"name": "web_search", "arguments": {"query": "..."}}`
2. Loose syntax: `web_search{"query": "..."}` or `web_search({"query": "..."})`

All parsing is done by `_extract_tool_call()` and `_try_loose_tool_call()` in `inference/ollama.py` — never by the orchestrator itself.

---

## 11. Speech Pipeline (STT / TTS)

### Voice input — Speech to Text

```
Telegram voice message (OGG/Opus)
  │
  ▼
webhook/telegram_voice.py
  │  GET https://api.telegram.org/file/{file_id}
  ▼
raw OGG bytes
  │
  ▼
services/stt/whisper.py  (WhisperLocalSTTBackend)
  │  model: tiny / base / small / medium / large
  │  device: cpu | cuda
  ▼
STTResponse { status, text, confidence }
  │
  └─ status == "success"  →  text enters normal pipeline
  └─ status == "failed"   →  fallback: acknowledge but skip
```

### Voice output — Text to Speech

```
format_response_node  →  final_output (text string)
  │
  ▼
webhook/telegram_voice.py
  │
  lines = [l for l in response.splitlines() if l.strip()]
  │
  ├─ len(lines) ≤ 5  →  transport.send_response(chat_id, text)  [text message]
  │
  └─ len(lines) > 5  →  text_to_ogg_voice(text)
                           │
                           ├─ primary:  gTTS → MP3 → ffmpeg → OGG/Opus
                           └─ fallback: Coqui XTTS v2 → WAV → ffmpeg → OGG/Opus
                           │
                           ▼
                         voice_transport.send_voice_message(chat_id, ogg_bytes)
```

---

## 12. Autonomous Heartbeat

A background `asyncio.Task` started on application startup polls every 30 seconds and fires at 08:00 daily.

```python
# agent/intelligence/autonomous_heartbeat.py

async def run_forever(self):
    while True:
        now = datetime.now()
        if now.hour == 8 and now.minute == 0:
            await self.send_morning_greeting()
            await asyncio.sleep(60)   # avoid double-trigger within the minute
        await asyncio.sleep(30)       # polling interval
```

### Morning greeting generation

```
08:00:00 — heartbeat triggers
  │
  ▼
QdrantLongTermMemoryStore.scroll_facts(conversation_id, limit=5)
  → last 5 known facts/insights about the user
  │
  ▼
OllamaModelBackend.generate(
    task="proactive_greeting",
    system_prompt="You are SAM, a warm best friend. One sentence, alive and conscious.",
    prompt="8AM greeting for Ismail. Context: [recent facts...]"
)
  │
  ▼
TelegramTransport.send_text(chat_id, greeting)
  → "Morning! Saw you were deep in the AI agent yesterday — 
     any breakthroughs? ☀️"
```

---

## 13. Observability & Tracing

### The tracing contract

All tracer implementations must satisfy the contract defined in `agent/tracing/tracer.py` and frozen by `design/observability_invariants.md`:

```
All Tracer implementations MUST guarantee:
  - No control flow influence
  - No state mutation
  - Non-fatal failures (never raise)
  - Best-effort execution
```

This is not convention — it is enforced by the abstract interface and verified by dedicated tests (`tests/observability/test_tracing_failure_safety.py`, `tests/observability/test_tracing_invariance.py`).

### Tracer backends

| Backend | `TRACER_BACKEND` | Transport | Best for |
|---------|-----------------|-----------|---------|
| `NoOpTracer` | `noop` | — | Development, CI |
| `LangSmithTracer` | `langsmith` | HTTPS (LangSmith API) | Production debugging |
| `OtelTracer` | `otel` | gRPC to OTel collector | Distributed tracing, Jaeger |

### Trace data model

Every agent invocation produces:

```
Trace (trace_id)
  └── Span: agent_request
        ├── Span: router_node          (duration, status)
        ├── Span: state_init_node
        ├── Span: decision_logic_node
        ├── Span: task_preprocessing_node
        ├── Span: memory_access_decision_node
        ├── Span: memory_read_node
        ├── Span: long_term_memory_read_node
        ├── Event: mcp_request_sent    (provider, query)
        ├── Event: mcp_response_received (result_count, chars)
        ├── Span: tool_execution_node
        ├── Span: model_call_node      (model, duration, tool_call detected?)
        ├── Span: memory_write_node
        ├── Span: long_term_memory_write_node
        └── Span: format_response_node (output_chars, truncated?)
```

### Structured logging

Set `LOG_FORMAT=json` to emit structured logs compatible with Loki / CloudWatch / Datadog:

```json
{
  "timestamp": "2026-05-16T13:15:48.334Z",
  "level": "INFO",
  "logger": "agent.langgraph_orchestrator",
  "message": "[LATENCY] model_call_node took 15.581s",
  "trace_id": "f0132022-970e-4ef5-abf4-01839d0b8d96",
  "conversation_id": "telegram_903341171"
}
```

Default (`LOG_FORMAT=text`) is human-readable — identical to pre-existing output:

```
2026-05-16 13:15:48,334 - agent.langgraph_orchestrator - INFO - [LATENCY] model_call_node took 15.581s
```

### Local debug API (development only)

When `LOCAL_OBSERVABILITY_ENABLED=true`, read-only inspection endpoints are available. Require `X-Debug-Token` header if `DEBUG_API_TOKEN` is configured.

| Endpoint | Returns |
|---------|---------|
| `GET /debug/health` | Agent health + config |
| `GET /debug/graph` | Static graph structure |
| `GET /debug/traces?limit=N` | Recent trace metadata |
| `GET /debug/spans?limit=N` | Recent span metadata |
| `GET /debug/memory?limit=N` | Memory operation events |
| `GET /debug/stats` | Store statistics |

---

## 14. Design Principles

These principles are applied consistently throughout the codebase. Understanding them is essential for contributing without breaking existing guarantees.

### P1 — Non-fatal memory and tracing

Every memory operation returns a typed `MemoryReadResponse` or `MemoryWriteResponse` with a `status` field. Every tracer call is wrapped in `try/except`. Neither ever raises an uncaught exception into the graph. The agent continues with degraded context rather than returning a 500.

```python
# agent/memory/base.py — the interface contract:
# "Never raise exceptions. Return typed response with status."
def read(self, request: MemoryReadRequest) -> MemoryReadResponse: ...
def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse: ...
```

### P2 — Swappable backends via abstract interfaces

Every external dependency is hidden behind an abstract base class:

| Interface | Location | Implementations |
|-----------|---------|----------------|
| `ModelBackend` | `inference/base.py` | `OllamaModelBackend`, `StubModelBackend` |
| `MemoryController` | `agent/memory/base.py` | `SQLiteShortTermMemoryStore`, `StubMemoryController` |
| `LongTermMemoryStore` | `agent/memory/long_term_base.py` | `QdrantLongTermMemoryStore`, `StubLongTermMemoryStore` |
| `STTBackend` | `services/stt/base.py` | `WhisperLocalSTTBackend`, `StubSTTBackend` |
| `TTSBackend` | `services/tts/base.py` | `CoquiTTSBackend`, `StubTTSBackend` |
| `Tracer` | `agent/tracing/tracer.py` | `LangSmithTracer`, `OtelTracer`, `NoOpTracer` |

Any backend can be swapped by changing an environment variable — no code changes required.

### P3 — Stub pattern for deterministic testing

Every backend interface has a stub implementation that returns deterministic, configurable responses without any external service. The stub pattern enables:

- Full CI runs with `LLM_BACKEND=stub`, `LTM_BACKEND=stub`, `STT_ENABLED=false`
- Unit tests that isolate individual nodes from all I/O
- Reproducible integration tests that don't depend on Ollama availability

The `StubModelBackend` always returns a fixed success response. The `StubMemoryController` uses an in-memory dict. The `StubLongTermMemoryStore` holds facts in memory.

### P4 — command is the only control flow signal

`decision_logic_node` is the **sole authority** for routing. No other node inspects or modifies the `command` field. Every edge in the graph either follows a fixed sequence or branches based on a value set by `decision_logic_node`. This makes the entire execution path auditable: read `decision_logic_node` and you understand every possible path through the graph.

### P5 — Memory is advisory, never authoritative

Fact retrieved from Qdrant are injected into the model prompt as plain text context. The agent cannot route differently based on memory contents. Memory cannot override guardrails. The LLM decides how (or whether) to use the context it receives.

### P6 — Node single responsibility

Each node has one job. The docstring of every node explicitly lists what it `MUST NOT` do. For example, `tool_execution_node` must not write to `memory_*` fields and must not set `command`. These constraints are tested by `tests/unit/test_tracing_invariants.py`.

---

## 15. Testing Strategy

### Test structure

```
tests/
├── unit/          ← isolated component tests (no external services)
├── integration/   ← full graph execution (stub backends only)
├── transport/     ← Telegram + WhatsApp webhook contract tests
├── mcp/           ← tool execution and guardrail tests
├── observability/ ← tracing invariant tests
└── prompting/     ← prompt builder and budget tests
```

### Running tests

```bash
# All tests (requires deps installed, no external services)
LLM_BACKEND=stub STM_BACKEND=sqlite LTM_BACKEND=stub \
STT_ENABLED=false TTS_ENABLED=false TRACER_BACKEND=noop \
TELEGRAM_BOT_TOKEN=test-token SQLITE_DB_PATH=:memory: \
pytest tests/ -v

# Unit only (fastest)
pytest tests/unit/ -v --tb=short

# Specific category
pytest tests/observability/ -v
pytest tests/integration/ -v --timeout=60
```

### What the tests verify

| Category | Tests | Key invariants |
|----------|-------|---------------|
| `unit/test_langgraph_skeleton.py` | Graph structure, node wiring | All 15 nodes registered, edges match spec |
| `unit/test_deterministic_memory_management.py` | DMA pattern detection | Write/read patterns match expected inputs |
| `unit/test_intelligence_fact_extraction.py` | Regex extraction | 13 patterns, confidence thresholds |
| `unit/test_sqlite_adapter.py` | SQLite CRUD | Upsert semantics, WAL mode, eviction |
| `integration/test_graph_execution.py` | Full graph (stub) | conversation_id/trace_id preserved, final_output present |
| `integration/test_memory_integration.py` | STM read/write cycle | Read authorisation, write authorisation |
| `integration/test_tool_intent_flow.py` | Tool trigger → execution | tool_context injected into second model call |
| `observability/test_tracing_failure_safety.py` | Tracing non-fatal | Agent produces output even when tracer throws |
| `observability/test_tracing_invariance.py` | Output unchanged | With/without tracing, output is identical |
| `transport/telegram/test_telegram_webhook.py` | Dedup + rate limit | Duplicate update_id dropped, flood dropped |

### CI configuration

The GitHub Actions pipeline (`/.github/workflows/ci.yml`) runs all categories with stub backends and no external services. The `build` job (Docker image) only runs after all tests pass.

---

## 16. Known Limitations

### Architectural

| Limitation | Impact | Workaround / Future path |
|-----------|--------|--------------------------|
| Single LLM instance (Ollama) | One request at a time; concurrent Telegram messages queue behind the LLM | Add request queue or multiple Ollama instances |
| Max 1 tool call per turn | Cannot chain tool results (e.g. search → then search again based on result) | Increase `MAX_TOOL_CALLS_PER_TURN` (changes latency profile) |
| SQLite STM — single file, single instance | Cannot scale horizontally; concurrent writes serialize | Replace with Redis for multi-instance deployments |
| Reflection node is fire-and-forget | Insights may not persist if the container shuts down mid-reflection | Add graceful shutdown with `asyncio.shield` |
| `requires_memory_read = True` always (Phase Humanizing) | Every turn reads SQLite even for stateless queries (e.g. "2+2") | Low impact for a personal assistant; can add heuristic bypass |

### Accuracy / Quality

| Limitation | Impact |
|-----------|--------|
| Freshness detection is keyword-based, not semantic | "Tell me about the new Python 3.13 features" won't trigger search (no freshness keyword) |
| Fact extraction is regex-based | Indirect personal facts ("at home I usually..." vs "I live at home") may be missed |
| LTM uses scroll (no semantic query) | Recent facts are retrieved, not the most relevant ones |
| phi3 model quality | Smaller local model; responses may be less coherent than GPT-4 class models |

### Operational

| Limitation | Impact |
|-----------|--------|
| ngrok free tier — static domain but session-dependent | Tunnel must be restarted if the ngrok process dies |
| Ollama model warm-up | First request after container start may time out (model loading); `OLLAMA_KEEP_ALIVE=24h` mitigates this |
| No multi-user support | System prompt and memory scope are designed for one user (Ismail). Multi-tenancy would require parameterized prompts and per-user conversation IDs |

---

## 17. API Reference

### Health

| Method | Path | Returns | Notes |
|--------|------|---------|-------|
| `GET` | `/health/live` | `{status, uptime_seconds, mode, ...}` | Always 200 if process alive |
| `GET` | `/health/ready` | `{status, agent_ready, message, ...}` | 503 if core modules fail to import |
| `GET` | `/health/trace` | `{tracer_backend, enabled, ...}` | Tracer configuration |

### Agent

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/` | — | API info and endpoint list |
| `POST` | `/invoke` | `{"input": "..."}` | `{status, output, conversation_id, trace_id}` |

### Telegram

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/webhook/telegram` | Telegram update receiver — always returns `{"status":"ok"}` |
| `POST` | `/webhook/telegram/voice` | Voice update receiver |
| `GET` | `/webhook/telegram/health` | Transport health check |
| `GET` | `/webhook/telegram/webhook-info` | Current Telegram webhook status |
| `POST` | `/webhook/telegram/set-webhook?webhook_url=...` | Register webhook URL with Telegram API |
| `GET` | `/webhook/telegram/webhook-info` | Current webhook + pending count + last error |

### WhatsApp

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/webhook/whatsapp` | Webhook challenge verification |
| `POST` | `/webhook/whatsapp` | WhatsApp message receiver |

### Debug (requires `LOCAL_OBSERVABILITY_ENABLED=true`)

All endpoints require `X-Debug-Token: <token>` header if `DEBUG_API_TOKEN` is set.

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/debug/health` | Agent health + observability status |
| `GET` | `/debug/graph` | Static graph structure |
| `GET` | `/debug/traces?limit=N` | Recent trace metadata |
| `GET` | `/debug/spans?limit=N` | Recent span metadata |
| `GET` | `/debug/memory?limit=N` | Memory operation events |
| `GET` | `/debug/stats` | Store statistics |

---

## 18. Configuration Reference

All configuration via environment variables. Copy `.env.example` → `.env`.

### Required

| Variable | Description |
|---------|-------------|
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather |

### LLM

| Variable | Default | Options |
|---------|---------|---------|
| `LLM_BACKEND` | `ollama` | `ollama`, `stub` |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Any Ollama base URL |
| `OLLAMA_MODEL` | `phi` | Any model pulled in Ollama (e.g. `phi3`, `llama3`) |

### Short-Term Memory

| Variable | Default | Notes |
|---------|---------|-------|
| `STM_BACKEND` | `sqlite` | `sqlite`, `stub` |
| `SQLITE_DB_PATH` | `/app/data/memory.db` | Use `:memory:` for testing |
| `STM_TTL_SECONDS` | `604800` | 7 days; entries older than this are evicted on write |

### Long-Term Memory

| Variable | Default | Notes |
|---------|---------|-------|
| `LTM_BACKEND` | `qdrant` | `qdrant`, `stub` |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL |
| `QDRANT_API_KEY` | *(empty)* | Optional; omit for local unauthenticated Qdrant |

### Speech

| Variable | Default | Notes |
|---------|---------|-------|
| `STT_ENABLED` | `false` | Enable Whisper voice transcription |
| `STT_BACKEND` | `whisper` | `whisper`, `stub` |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda` |
| `TTS_ENABLED` | `false` | Enable voice replies for long outputs |
| `TTS_BACKEND` | `coqui` | `coqui`, `stub` |

### Web Search (MCP)

| Variable | Notes |
|---------|-------|
| `EXA_API_KEY` | Neural search. Free: 1,000 req/mo. [dashboard.exa.ai](https://dashboard.exa.ai/api-keys) |
| `BRAVE_API_KEY` | Privacy-first search. Free: 2,000 req/mo. [brave.com/search/api](https://brave.com/search/api/) |
| `LINKUP_API_KEY` | Real-time facts. Free tier. [app.linkup.so](https://app.linkup.so) |
| `SEARXNG_URL` | Default: `http://searxng:8080` (internal container — always available, no key needed) |

### Security

| Variable | Default | Notes |
|---------|---------|-------|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins. Set to specific domain in production. |
| `DEBUG_API_TOKEN` | *(empty)* | Required `X-Debug-Token` value for `/debug/*`. Leave empty = open access. |
| `RATE_LIMIT_MAX_CALLS` | `3` | Max Telegram messages per user per window |
| `RATE_LIMIT_WINDOW_S` | `5` | Rate limit window in seconds |

### Observability

| Variable | Default | Notes |
|---------|---------|-------|
| `TRACER_BACKEND` | `noop` | `noop`, `langsmith`, `otel` |
| `LANGCHAIN_API_KEY` | *(empty)* | LangSmith API key |
| `LANGCHAIN_PROJECT` | `SAM-Agent` | LangSmith project name |
| `LANGCHAIN_TRACING_V2` | `true` | Enable LangChain tracing integration |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://otel-collector:4317` | OTel collector gRPC endpoint |
| `LOCAL_OBSERVABILITY_ENABLED` | `false` | Expose `/debug/*` endpoints |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `text` | `text` (human-readable) or `json` (structured, for aggregation) |

---

## 19. Deployment Guide

### Prerequisites

- Docker 24+ and Docker Compose v2
- ngrok account with a static domain
- Telegram Bot Token from @BotFather
- (Optional) API keys for Exa, Brave, or Linkup

### Quick Start (stub backends — no external services needed)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set at minimum: TELEGRAM_BOT_TOKEN

# 2. Start stack with stub LLM (instant, no model download)
LLM_BACKEND=stub LTM_BACKEND=stub docker compose up

# 3. Start ngrok tunnel
./ngrok http 8000 --domain=YOUR-STATIC-DOMAIN.ngrok-free.app

# 4. Register Telegram webhook
curl -X POST \
  "http://localhost:8000/webhook/telegram/set-webhook?webhook_url=https://YOUR-STATIC-DOMAIN.ngrok-free.app/webhook/telegram"

# 5. Verify
curl http://localhost:8000/health/ready
curl http://localhost:8000/webhook/telegram/webhook-info
```

### Production with real LLM

```bash
# 1. Configure .env for production
LLM_BACKEND=ollama
OLLAMA_MODEL=phi3
STM_BACKEND=sqlite
LTM_BACKEND=qdrant
TRACER_BACKEND=langsmith
LANGCHAIN_API_KEY=your_key
STT_ENABLED=false   # or true if voice input needed
TTS_ENABLED=false   # or true if voice replies needed

# 2. Start full stack
docker compose up -d

# 3. Pull the LLM model
docker exec sam-agent-ollama ollama pull phi3

# 4. Verify Ollama loaded the model
docker exec sam-agent-ollama ollama list

# 5. Register webhook
curl -X POST \
  "http://localhost:8000/webhook/telegram/set-webhook?webhook_url=https://YOUR-DOMAIN.ngrok-free.app/webhook/telegram"

# 6. Send a test message via /invoke
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, what is your name?"}'
```

### Docker build targets

```bash
# CPU build (default, ~2 GB)
docker build -f docker/Dockerfile.agent \
  --target final-base \
  -t sam-agent:latest .

# CPU build with Whisper + Coqui (~8 GB)
docker build -f docker/Dockerfile.agent \
  --target final-base \
  --build-arg INSTALL_WHISPER=true \
  --build-arg INSTALL_COQUI=true \
  -t sam-agent:full .

# GPU build — CUDA 12.1+ required (~12 GB)
docker build -f docker/Dockerfile.agent \
  --target final-gpu \
  --build-arg INSTALL_WHISPER=true \
  --build-arg INSTALL_COQUI=true \
  -t sam-agent:gpu .
```

### Health probes

| Probe | Endpoint | Expected |
|-------|---------|---------|
| Liveness | `GET /health/live` | `{"status": "healthy"}` |
| Readiness | `GET /health/ready` | `{"status": "healthy", "agent_ready": true}` |

The Docker HEALTHCHECK uses `/health/live`. Kubernetes readiness should use `/health/ready`.

### Restoring ngrok after restart

```bash
# The static domain is pre-registered — just restart ngrok:
./ngrok http 8000 --domain=YOUR-STATIC-DOMAIN.ngrok-free.app

# Webhook URL is unchanged, so no Telegram re-registration needed.
# Verify pending messages cleared:
curl http://localhost:8000/webhook/telegram/webhook-info | python -m json.tool
# → "pending_update_count" should drop to 0
```

---

## 20. Project Structure

```
SAM-Agent-Telegram/
│
├── agent/                              # Core agent package
│   ├── api.py                          # ★ Production entry point (python -m agent.api)
│   ├── health.py                       # Liveness + readiness health checker
│   ├── langgraph_orchestrator.py       # ★ 15-node LangGraph DAG — core orchestration
│   ├── orchestrator.py                 # Public SAMOrchestrator wrapper
│   ├── state_schema.py                 # ★ AgentState dataclass — central contract
│   ├── memory_nodes.py                 # Memory read/write node wrappers
│   ├── logging_config.py               # Centralised structured logging (text + JSON)
│   │
│   ├── intelligence/                   # Agent intelligence subsystem
│   │   ├── fact_extraction.py          # Personal fact detection (regex + confidence)
│   │   ├── guardrails.py               # Memory write limits per user/conversation
│   │   ├── memory_retrieval.py         # Memory context assembly for prompt injection
│   │   ├── metrics.py                  # Agent performance metrics collection
│   │   ├── tools.py                    # ToolRegistry — register/dispatch tool calls
│   │   └── autonomous_heartbeat.py     # Daily 08:00 personalised greeting service
│   │
│   ├── mcp/                            # Model Context Protocol — web search
│   │   ├── external_client.py          # ★ Multi-provider: Exa → Brave → Linkup → SearXNG
│   │   ├── guardrails.py               # Tool limits (MAX_RESULTS=3, MAX_CHARS=800, TIMEOUT=15s)
│   │   └── connectivity_test.py        # API key validation + Smithery connection setup
│   │
│   ├── memory/                         # Memory backend implementations
│   │   ├── base.py                     # Abstract MemoryController interface
│   │   ├── types.py                    # MemoryReadRequest/Response, MemoryWriteRequest/Response
│   │   ├── sqlite.py                   # ★ SQLite STM — WAL mode, TTL eviction, upsert
│   │   ├── stub.py                     # In-memory stub (testing / CI)
│   │   ├── long_term_base.py           # Abstract LongTermMemoryStore interface
│   │   ├── long_term_qdrant.py         # ★ Qdrant LTM — append-only vector storage
│   │   ├── long_term_stub.py           # In-memory LTM stub (testing / CI)
│   │   ├── long_term_types.py          # LTM request/response types
│   │   └── cognee_adapter.py           # Cognee graph-memory adapter (experimental)
│   │
│   ├── observability/                  # Local dev observability (not for production)
│   │   ├── interface.py                # Read-only inspection interface
│   │   ├── context.py                  # Request-scoped execution context + __deepcopy__
│   │   └── store.py                    # In-memory trace/span/memory event storage
│   │
│   ├── prompting/                      # Prompt engineering
│   │   └── prompt_builder.py           # ★ SYSTEM_PROMPT + REFLECTION_PROMPT + budget logic
│   │
│   ├── tools/                          # Tool implementations
│   │   └── web_search_tool.py          # WebSearchTool — calls MCP external_client
│   │
│   └── tracing/                        # Observability backends
│       ├── tracer.py                   # ★ Abstract Tracer interface + NoOpTracer
│       ├── tracer_factory.py           # Backend selection from TRACER_BACKEND env var
│       ├── langsmith_tracer.py         # LangSmith integration
│       ├── otel_tracer.py              # OpenTelemetry integration (lazy import)
│       ├── langtrace_tracer.py         # Langtrace (placeholder)
│       └── alarms.py                   # Invariant violation detection + alerting
│
├── inference/                          # LLM backend abstraction layer
│   ├── base.py                         # Abstract ModelBackend interface
│   ├── types.py                        # ModelRequest / ModelResponse
│   ├── ollama.py                       # ★ Ollama backend — httpx, 3-retry backoff, tool-call parser
│   └── stub.py                         # Deterministic stub (testing / CI)
│
├── transport/                          # Messaging platform I/O adapters
│   ├── telegram/
│   │   └── transport.py                # NormalizedMessage + Telegram message sender
│   └── whatsapp/
│       ├── webhook.py                  # WhatsApp webhook router
│       ├── normalize.py                # WhatsApp payload → NormalizedMessage
│       ├── security.py                 # HMAC-SHA256 signature verification
│       ├── sender.py                   # WhatsApp message sender
│       └── schemas.py                  # Pydantic payload schemas
│
├── webhook/                            # FastAPI webhook routers
│   ├── telegram.py                     # ★ Telegram text handler — dedup + rate limiting
│   └── telegram_voice.py               # Voice handler — STT + TTS pipeline
│
├── services/                           # External service integrations
│   ├── stt/                            # Speech-to-Text
│   │   ├── base.py                     # Abstract STTBackend (STTRequest/Response)
│   │   ├── whisper.py                  # OpenAI Whisper — local CPU/GPU
│   │   └── stub.py                     # Stub STT (returns fixed text)
│   ├── tts/                            # Text-to-Speech
│   │   ├── base.py                     # Abstract TTSBackend (TTSRequest/Response)
│   │   ├── coqui.py                    # Coqui XTTS v2 — local, voice cloning support
│   │   └── stub.py                     # Stub TTS (returns empty audio)
│   └── audio/
│       └── normalizer.py               # Audio format normalisation utilities
│
├── infra/                              # Infrastructure initialisation
│   ├── config.py                       # InfraConfig — backend factory from environment
│   └── bootstrap.py                    # Singleton infrastructure bootstrapper
│
├── tests/                              # Test suite
│   ├── unit/                           # Component tests — no external services
│   ├── integration/                    # Full graph execution — stub backends
│   ├── transport/                      # Webhook contract tests
│   ├── mcp/                            # Tool execution + guardrail tests
│   ├── observability/                  # Tracing invariant tests
│   ├── prompting/                      # Prompt builder + budget tests
│   ├── services/                       # STT/TTS service tests
│   ├── tools/                          # WebSearchTool tests
│   └── conftest.py                     # Pytest sys.path setup
│
├── evaluation/                         # Offline evaluation framework (not production)
├── experiment_harness/                 # Automated experiment runner (not production)
├── experiments/                        # Experiment definitions and result artifacts
│
├── design/                             # Architecture and design documents
│   └── langgraph_skeleton.md           # Formal graph spec (source of truth for orchestrator)
│
├── scripts/                            # Diagnostic and validation scripts
│   ├── inspect_short_term_memory.py    # Query SQLite STM directly
│   ├── test_agent_endpoints.ps1        # PowerShell API smoke test
│   ├── test_endpoints.ps1              # PowerShell webhook test
│   ├── test_observability.sh           # Shell observability smoke test
│   └── validate_deployment.py          # Python deployment validation
│
├── main.py                             # Development entry point (uvicorn main:app)
├── config.py                           # Root Config class — env var typed access
├── docker-compose.yml                  # ★ 6-service Docker Compose stack
├── docker/Dockerfile.agent             # Multi-stage CPU/GPU image build
├── pyproject.toml                      # Python project + pinned dependency versions
├── uv.lock                             # Pinned dependency lockfile (reproducible builds)
├── otel-collector-config.yaml          # OpenTelemetry collector configuration
├── .env.example                        # ★ Environment variable template
└── .gitignore                          # Git ignore rules
```

★ = most important files to read first when learning the codebase.

---

## 21. Glossary

| Term | Definition |
|------|-----------|
| **AgentState** | The central dataclass that flows through the entire LangGraph graph, carrying all fields needed for an invocation — input, memory flags, model response, tool results, output. |
| **Command** | A string field in `AgentState` (`"preprocess"`, `"call_model"`, `"execute_tool"`, `"memory_write"`, etc.) that `decision_logic_node` sets to control routing. No other node may write it. |
| **DMA** | Deterministic Memory Access — the phase that added rule-based intent detection for memory reads and writes, using precompiled regex patterns rather than LLM classification. |
| **Guardrail** | A hard constraint enforced by a node before executing its operation. Violations are handled gracefully — they never crash the agent. Example: `MAX_TOOL_CALLS_PER_TURN = 1`. |
| **LTM** | Long-Term Memory — cross-session, permanent personal facts stored in Qdrant as vectors. Append-only. Advisory only (never influences routing). |
| **MCP** | Model Context Protocol — the phase that added web search tool execution. Also refers to the tool-calling convention (`[TOOL_CALL]{...}`). |
| **Non-fatal** | A design property: the operation always returns a typed result rather than raising an exception. Memory, tracing, and tool calls are all non-fatal. |
| **Phase** | A named increment of development that added specific capabilities to the agent. Phases are documented in code comments (e.g., `# Phase DMA`). See [Section 3](#3-phase-architecture). |
| **Reflection** | The background `asyncio.Task` that runs 5 seconds after every reply to extract new insights about the user and write them to LTM. Part of Phase Consciousness. |
| **STM** | Short-Term Memory — per-session conversation context stored in SQLite. Evicts entries older than `STM_TTL_SECONDS`. |
| **Stub** | A deterministic, in-memory implementation of a backend interface. Used in CI and unit tests to eliminate all external dependencies. |
| **Tracer** | The observability abstraction (`agent/tracing/tracer.py`). All implementations must be passive: no control flow influence, no state mutation, non-fatal failures. |
| **Two-pass LLM** | When a tool is triggered, the model is called twice: once to decide to search (or pre-search is forced), and once to synthesise the tool results into a reply. |

---

## 22. Contributing Guide

### Before you start

1. Read `design/langgraph_skeleton.md` — the formal graph specification
2. Understand the phase naming convention (Section 3)
3. Run the test suite with stub backends to establish a baseline

### Adding a new LLM backend

1. Create `inference/my_backend.py` implementing `ModelBackend` (from `inference/base.py`)
2. Add `create_my_backend_backend()` to `infra/config.py`
3. Add `"my_backend"` to the `LLMBackendType` literal in `infra/config.py`
4. Add a stub test in `tests/unit/test_infrastructure_integration.py`

### Adding a new web search provider

1. Add a provider method in `agent/mcp/external_client.py` following the cascade pattern
2. Add its env var key (e.g., `MYPROVIDER_API_KEY`) to the provider check
3. Add it to the cascade list in priority order
4. Add test cases in `tests/mcp/test_mcp_schema.py`
5. Document the new key in `.env.example` and [Section 18](#18-configuration-reference)

### Adding a new graph node

1. Add the state fields the node reads/writes to `agent/state_schema.py` with the phase tag
2. Implement the node in `agent/langgraph_orchestrator.py` following the `_wrap_node_execution` pattern
3. Register it in `_build_graph()` with `graph.add_node()`
4. Add edges/conditional edges from/to `decision_logic_node`
5. Add a routing case in `_route_from_decision()` if needed
6. Update `decision_logic_node` with the new command emission logic
7. Document the "MUST NOT" constraints in the node's docstring

### Code review checklist

- [ ] New node follows single-responsibility (one job, explicit MUST NOT list)
- [ ] Memory operations are non-fatal (return typed response, never raise)
- [ ] Tracing calls are wrapped in `try/except Exception: pass`
- [ ] State field ownership is documented in the node docstring
- [ ] Stub implementation exists for any new external dependency
- [ ] Phase tag added to new state fields (e.g., `# Phase MyPhase`)
- [ ] Tests added for the new node or backend
- [ ] `.env.example` updated for any new environment variables

### Running the CI checks locally

```bash
# Lint
ruff check . --select E,F,W,I --ignore E501
black --check --line-length 100 .

# Type check
mypy agent/ inference/ transport/ services/ --ignore-missing-imports

# Full test suite
LLM_BACKEND=stub STM_BACKEND=sqlite LTM_BACKEND=stub \
STT_ENABLED=false TTS_ENABLED=false TRACER_BACKEND=noop \
TELEGRAM_BOT_TOKEN=test-token SQLITE_DB_PATH=:memory: \
pytest tests/ -v --tb=short
```

---

---

## 23. Prompt Engineering & Token Budget

Understanding how the model receives information is critical to tuning SAM's response quality and latency.

### Prompt structure

Every Ollama call assembles the following message list (in `/api/chat` format):

```
[0] role: system
    content: SYSTEM_PROMPT   ← behavioural contract, injected by OllamaModelBackend

[1] role: user
    content:
        [Memory Context block — if retrieved]
        ---
        [Tool Results block — if web search ran]
        ---
        [User message]
        Answer:
```

The system prompt is never embedded in the user message — it is always the `system` role to prevent double-injection when a model backend is changed.

### System prompt design

The `SYSTEM_PROMPT` (defined once in `agent/prompting/prompt_builder.py`, imported by `inference/ollama.py`) encodes SAM's complete behavioural contract in 9 rules:

| Rule | Directive | Engineering reason |
|------|-----------|-------------------|
| IDENTITY | "You are SAM. The user is ISMAIL." | Anchors persona across all contexts |
| FORMAT | No "SAM:" prefix in replies | Prevents transport layer from receiving formatting artifacts |
| FLOW-FIRST | Prioritise conversation transcript above all | Reduces topic drift across multi-turn sessions |
| GROUNDING | Do not speculate | Reduces hallucination rate on personal facts |
| SEARCH FIRST | Must call web_search for real-world data | Forces tool use instead of stale training data |
| BREVITY | Maximum 2 sentences | Enforced in prompt AND by `format_response_node` guardrail (belt + suspenders) |
| STABILITY | Use [CURRENT IDENTITY] to anchor persona | Prevents identity drift in long conversations |
| MEMORY | Weave context organically | Prevents robotic "I remember that..." phrasing |
| CURIOSITY | Ask about Ismail's life once every few turns | Drives proactive relationship building |

### Context budget (from `agent/prompting/prompt_builder.py`)

```python
_MAX_MEMORY_CHARS:       int = 2000   # ≈ 500 tokens  — STM + LTM combined
_MAX_TOOL_CHARS:         int = 1500   # ≈ 375 tokens  — web search results
_MAX_TOTAL_INJECT_CHARS: int = 3000   # hard cap on combined injected context
```

**Priority rule:** when both memory and tool context are present and their sum exceeds `_MAX_TOTAL_INJECT_CHARS`, tool context takes priority and memory is trimmed first. Rationale: tool results answer the immediate query; memory provides background that the model can partially reconstruct from its training.

### Token economics — typical request

```
Component                   Approx. tokens   Notes
─────────────────────────── ──────────────   ─────────────────────────────────
SYSTEM_PROMPT               ~120             Fixed per request
Memory context (STM + LTM)  0–500           Capped at _MAX_MEMORY_CHARS
Tool context (web search)   0–375           Capped at _MAX_TOOL_CHARS (Phase 4)
User message                ~20–80          Typical conversational message
Answer: marker              1
─────────────────────────── ──────────────   ─────────────────────────────────
Input total (no tool)       ~140–700
Input total (with tool)     ~515–1076
─────────────────────────── ──────────────   ─────────────────────────────────
Output (guardrailed)        ≤ 200           MAX_OUTPUT_CHARS=800 ÷ ~4 chars/token
```

### Why context budgets were tightened (Phase 4)

The MCP guardrail constants were reduced in Phase 4 specifically to reduce the second model call latency:

```
Phase 3 → Phase 4
MAX_RESULTS:      5  →  3     (-40% search payload)
MAX_SNIPPET_LEN:  300 → 200   (-33% per snippet)
MAX_TOTAL_CHARS:  1500 → 800  (-47% tool context)
```

Benchmark observation: large tool context was the #2 latency contributor after raw Ollama inference time. Smaller context → faster tokenisation and prefill → measurable reduction in second-pass latency.

### Reflection prompt

The `REFLECTION_PROMPT` is a separate system prompt used only by `reflection_node`. It instructs the model to output a strict JSON array of insight objects — a structured output contract that avoids free-form parsing:

```python
REFLECTION_PROMPT = """You are the inner consciousness of SAM.
...
Output ONLY a JSON list:
[{"fact": "...", "type": "mood|interest|bio|goal", "confidence": 0.0-1.0}]
If nothing new learned, return empty list []."""
```

The strict JSON contract means `reflection_node` can call `json.loads()` directly on the model output rather than parsing prose — reducing the surface area for hallucination.

---

## 24. Performance Profile

Latency numbers observed during live testing (`phi3:latest` on CPU, Docker on Windows host).

### Node latency breakdown

| Node | Typical duration | Bottleneck? |
|------|-----------------|-------------|
| `router_node` | < 1 ms | No |
| `state_init_node` | < 1 ms | No |
| `decision_logic_node` | < 1 ms | No |
| `task_preprocessing_node` | < 1 ms | No |
| `memory_access_decision_node` | < 1 ms | No (regex, no I/O) |
| `fact_extraction_node` | < 5 ms | No (regex) |
| `write_authorization_node` | < 1 ms | No |
| `memory_read_node` (SQLite) | 1–10 ms | No |
| `long_term_memory_read_node` (Qdrant) | 10–30 ms | Minor (network) |
| `tool_execution_node` (web search) | 2,000–8,000 ms | **Yes** — network + provider latency |
| `model_call_node` (Ollama phi3, CPU) | 8,000–30,000 ms | **Yes — primary bottleneck** |
| `memory_write_node` (SQLite) | 5–15 ms | No |
| `long_term_memory_write_node` (Qdrant) | 50–200 ms | Minor |
| `format_response_node` | < 1 ms | No |

### Total request latency — typical paths

```
Path                                    Typical wall time
─────────────────────────────────────── ─────────────────
Text message, no tool (simple query)    10–30 s
Text message + web search               20–45 s
Voice message (Whisper base, CPU)       +3–8 s for STT
```

The HTTP 200 is returned to Telegram **before** this work begins (background task). The user receives the reply after the full latency, but Telegram does not time out because the 200 was already sent.

### Background reflection latency

```
Reflection fires 5s after reply is sent
Ollama call for insight extraction:  8–20 s
Qdrant write (2 facts):              ~100 ms
Total reflection cycle:              ~13–25 s
```

Reflection runs entirely in the background and does not affect the user-facing response time.

### Reducing latency

| Strategy | Impact | Trade-off |
|---------|--------|-----------|
| Use GPU for Ollama (`WHISPER_DEVICE=cuda`) | 5–10× faster model calls | Requires CUDA 12.1+, GPU Docker |
| Use a smaller model (`tiny`, `phi` instead of `phi3`) | 2–3× faster | Lower response quality |
| Disable LTM read (`LTM_BACKEND=stub`) | Save 10–30 ms per turn | No cross-session memory |
| Reduce `OLLAMA_KEEP_ALIVE` to 0 | Not recommended — increases cold-start latency | — |
| Set `OLLAMA_KEEP_ALIVE=24h` (default) | Model stays loaded, eliminates cold-start | Memory usage |
| Reduce freshness keyword set | Fewer forced tool calls | May miss real-time queries |

### Concurrency model

SAM is currently a **single-threaded async server** with a **single Ollama instance**:

- FastAPI (uvicorn) handles concurrent HTTP connections on one event loop
- LLM calls (`model_call_node`) block the event loop until `ainvoke` completes
- Concurrent Telegram messages queue behind the active LLM call
- Rate limiting (3 req/5s/user) bounds the queue growth

For personal-assistant scale (one primary user, occasional concurrent messages) this is acceptable. For multi-user scale, a message queue (Redis + Celery) in front of the agent would decouple webhook receipt from LLM processing.

---

## 25. Security Model

### What is protected

| Surface | Mechanism | Implementation |
|---------|-----------|---------------|
| WhatsApp webhook authenticity | HMAC-SHA256 signature verification | `transport/whatsapp/security.py` |
| Telegram update deduplication | TTLCache on `update_id` (5000 entries, 5 min TTL) | `webhook/telegram.py` |
| Telegram flood protection | Per-user rate limit (3 req / 5 s) | `webhook/telegram.py` |
| CORS origin restriction | `ALLOWED_ORIGINS` env var (default `*` for dev) | `agent/api.py`, `main.py` |
| Debug endpoint access | `X-Debug-Token` header (when `DEBUG_API_TOKEN` set) | `agent/api.py` |
| SQLite data durability | WAL mode + `PRAGMA synchronous=FULL` | `agent/memory/sqlite.py` |
| LTM data integrity | Qdrant append-only — no updates or deletes | `agent/memory/long_term_qdrant.py` |
| Credential isolation | All API keys in `.env` (gitignored); never logged or stored in state | `.gitignore`, `agent/mcp/external_client.py` |
| Non-root container user | `agent` user, uid=1000 | `docker/Dockerfile.agent` |
| Tool call injection | `MCPGuardrails.sanitize_results()` validates URLs (`startswith("http")`) | `agent/mcp/guardrails.py` |

### What is NOT protected (known gaps)

| Surface | Risk | Mitigation |
|---------|------|-----------|
| `/invoke` endpoint | No authentication — any caller can invoke the agent directly | Add API key auth if exposing publicly; currently protected by ngrok being the only entry point |
| `/health/*` endpoints | Publicly readable — reveals backend configuration | Low risk; set `ALLOWED_ORIGINS` if needed |
| Telegram bot token | If leaked, anyone can impersonate the bot or read messages | Rotate immediately via @BotFather; token is gitignored |
| ngrok URL | If the static domain is known, anyone can POST to the webhook | Telegram validates that updates come from Telegram servers; non-Telegram POSTs are handled gracefully |
| Ollama API | Exposed only on internal Docker network (not mapped to host) | Safe as long as Docker network is not bridged to untrusted networks |
| Qdrant API | Exposed on host port 6333 if Docker is running | Set `QDRANT_API_KEY` or firewall the port in production |

### Prompt injection surface

The user's message is injected into the model prompt as-is. A crafted message could attempt to override the system prompt or inject tool call syntax. Current mitigations:

- `MAX_OUTPUT_CHARS = 800` limits any amplified output
- `MAX_TOOL_CALLS_PER_TURN = 1` prevents cascading tool abuse
- `MCPGuardrails.check_tool_call_limit()` is checked in both `decision_logic_node` and `tool_execution_node` (belt + suspenders)
- Tool results are sanitised before injection (`sanitize_results()`)

### Secrets handling

```
.env                 → gitignored, never committed
Config.TELEGRAM_BOT_TOKEN  → never appears in logs
MCP API keys         → never stored in AgentState
LangSmith API key    → passed as env var to container, not logged
```

The `agent/mcp/external_client.py` docstring explicitly states: *"Credentials never logged or stored in state."*

---

## 26. Local Development & Troubleshooting

### Running without Docker

For fast iteration on agent logic without building images:

```bash
# 1. Install dependencies (requires uv)
pip install uv
uv sync

# or with pip directly:
pip install -e ".[dev]"

# 2. Set up minimal environment
export TELEGRAM_BOT_TOKEN=your-token
export LLM_BACKEND=stub          # no Ollama needed
export STM_BACKEND=sqlite
export LTM_BACKEND=stub          # no Qdrant needed
export STT_ENABLED=false
export TTS_ENABLED=false
export TRACER_BACKEND=noop
export SQLITE_DB_PATH=./dev.db

# 3. Run the development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# or the production entry point:
python -m agent.api --port 8000
```

### Running a single service locally

```bash
# Ollama only (if you want a real LLM without full Docker)
docker run -p 11434:11434 -v ollama_data:/var/lib/ollama ollama/ollama
docker exec <container> ollama pull phi3

# Qdrant only
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant
```

### Windows-specific notes

```powershell
# Set env vars in PowerShell
$env:TELEGRAM_BOT_TOKEN = "your-token"
$env:LLM_BACKEND = "stub"

# Run tests
python -m pytest tests/unit/ -v --tb=short

# Line endings: git is configured for CRLF on Windows.
# LF↔CRLF warnings on git operations are expected and harmless.
# To suppress them globally:
git config --global core.autocrlf true
```

### Running the test suite

```bash
# Fastest — unit tests only (no services, ~3s)
pytest tests/unit/ -v --tb=short

# Integration tests (stub backends, no services, ~10s)
LLM_BACKEND=stub STM_BACKEND=sqlite LTM_BACKEND=stub \
STT_ENABLED=false TTS_ENABLED=false TRACER_BACKEND=noop \
TELEGRAM_BOT_TOKEN=test-token SQLITE_DB_PATH=:memory: \
pytest tests/integration/ -v --tb=short

# Specific test category
pytest tests/observability/ -v       # tracing invariants
pytest tests/mcp/ -v                 # tool guardrails
pytest tests/transport/ -v           # webhook contracts
```

---

### Troubleshooting

#### `/health/ready` returns `unhealthy` — `No module named 'opentelemetry'`

**Cause:** A top-level import of `otel_tracer.py` forces `opentelemetry` to load even when `TRACER_BACKEND=langsmith`.

**Fix:** This was resolved in commit `feeaa08` by removing the dead `OtelTracer` import from `langgraph_orchestrator.py`. If you see this error on an older build, rebuild the image.

---

#### Telegram webhook always returns errors — `pending_update_count` stays > 0

**Cause:** ngrok tunnel is down.

**Fix:**
```bash
./ngrok http 8000 --domain=YOUR-STATIC-DOMAIN.ngrok-free.app
# No Telegram re-registration needed if using a static domain.
curl http://localhost:8000/webhook/telegram/webhook-info
# → pending_update_count should drop to 0
```

---

#### Agent crashes on all Telegram messages — `cannot pickle '_thread.lock'`

**Cause:** `dataclasses.asdict(state)` inside `_wrap_node_execution` deepcopies `AgentExecutionContext.telemetry_emitter` (the LangSmith tracer), which contains `threading.Lock` objects.

**Fix:** Resolved in commit `feeaa08`:
- `_snapshot()` function now skips `execution_context` before deepcopying
- `__deepcopy__` added to `AgentExecutionContext`
- `graph.invoke()` → `await graph.ainvoke()` across all async call sites

---

#### Duplicate log lines — every agent log appears twice

**Cause:** The `agent` namespace logger had a `StreamHandler` added while `propagate=True` allowed records to also reach the root handler.

**Fix:** `agent_logger.propagate = False` in `agent/logging_config.py`. Resolved in commit `feeaa08`.

---

#### Ollama returns `model not found`

```bash
# List available models in the container
docker exec sam-agent-ollama ollama list

# Pull the configured model
docker exec sam-agent-ollama ollama pull phi3

# Verify OLLAMA_MODEL in your .env matches the pulled model name exactly
grep OLLAMA_MODEL .env
```

---

#### Qdrant LTM writes fail silently

**Cause:** Qdrant container not running, or the collection doesn't exist yet.

**Fix:**
```bash
# Check Qdrant health
curl http://localhost:6333/healthz

# The collection is created automatically on first write.
# If it fails, check logs:
docker logs sam-agent-qdrant --tail=50
docker logs sam-agent --tail=50 | grep -i qdrant
```

---

#### STT fails — voice messages not transcribed

**Cause:** Whisper model not downloaded, or `STT_ENABLED=false`.

**Fix:**
```bash
# Verify STT is enabled
grep STT_ENABLED .env   # should be "true"

# Whisper downloads on first use — check for download errors:
docker logs sam-agent --tail=100 | grep -i whisper

# If CUDA errors: set WHISPER_DEVICE=cpu in .env
```

---

#### Agent /health/ready fails — `agent_logic_ok: false`

**Cause:** A broken import somewhere in `agent/orchestrator.py` import chain.

**Debug steps:**
```bash
# Enter the container and test imports manually
docker exec -it sam-agent python -c "from agent.orchestrator import SAMOrchestrator; print('OK')"

# If import fails, read the full traceback:
docker exec -it sam-agent python -c "
import traceback
try:
    from agent.orchestrator import SAMOrchestrator
except Exception:
    traceback.print_exc()
"
```

---

## Licence

MIT — see [LICENSE](LICENSE)
