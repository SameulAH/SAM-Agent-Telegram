# Advanced LangGraph Design

## 🏗️ Graph Structure

The agent uses a complex DAG with several specialized pipelines:

### 1. Memory Access Decision (DMA)
- **`memory_access_decision_node`**: Detects intent.
- **`fact_extraction_node`**: Extracts facts (no LLM).
- **`write_authorization_node`**: Validates facts before storage.

### 2. Tool Execution (MCP)
- **`tool_execution_node`**: Handles external tool calls (e.g., search).
- Routes back to `decision_logic_node` to re-invoke the model with tool context.

### 3. Model Call
- **`model_call_node`**: The primary LLM interaction point.

### 4. Memory Write Cycle
- **`memory_write_node`**: SQLite persistence.
- **`long_term_memory_write_node`**: Qdrant persistence.

## 🚦 Invariants
- **Max 1 tool call per turn**: Enforced by `MCPGuardrails`.
- **Deterministic routing**: All branches are based on state variables, not model guesses.
- **Tracing**: Every node execution is wrapped in an OpenTelemetry-compatible tracer.
