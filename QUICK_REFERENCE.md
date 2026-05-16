# SAM-Agent Quick Reference (Advanced)

## 🚀 Common Commands

### Start the Bot (FastAPI)
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Verification
```bash
python verify_skeleton.py
```

### Test Memory
```bash
python verify_memory_system.py
```

## 📂 Key Directories

- `/transport`: Platform-specific messaging logic (Telegram/WhatsApp).
- `/agent/intelligence`: Fact extraction and guardrails.
- `/agent/mcp`: Tool execution logic.
- `/webhook`: FastAPI routers for different platforms.

## 🧠 Memory Intent Detection (DMA)
The agent uses regex patterns in `langgraph_orchestrator.py` to detect if the user is stating a fact (Write Intent) or asking about the past (Read Intent). This saves LLM tokens and reduces latency.

## 🛠️ MCP Tools
Currently supports `web_search`. Adding a tool requires:
1. Defining the tool in `agent/intelligence/tools.py`.
2. Updating the system prompt in `agent/prompting/prompt_builder.py`.
3. Ensuring the orchestrator can parse the tool call.
