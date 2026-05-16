# Agent Core

This package contains the intelligence and orchestration logic for the SAM Agent.

## 🏗️ Sub-packages

- **`intelligence/`**: Fact extraction, guardrails, and tool definitions.
- **`mcp/`**: Model Context Protocol implementation.
- **`prompting/`**: System prompt building and templates.
- **`memory/`**: Short-term and long-term memory backends.

## 🧠 The Orchestrator
`langgraph_orchestrator.py` is the main entry point for agent logic. It implements a stateful graph that coordinates between the user's input, the memory stores, and the LLM.

## 🚦 Guardrails
The agent uses strict guardrails to:
- Truncate long outputs.
- Limit tool execution.
- Validate memory writes.
