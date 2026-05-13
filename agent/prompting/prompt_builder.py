"""
Prompt Builder Layer
====================

Assembles structured user-facing prompts for the model backend.

Responsibilities:
- Defines the authoritative SYSTEM_PROMPT behavioral contract
- Assembles memory context + tool results + user input into a bounded prompt
- Enforces hard character-budget limits to prevent context bloat
- Provides the system prompt for model backends (e.g. OllamaModelBackend)

Invariants:
- memory_context is capped at _MAX_MEMORY_CHARS before injection
- tool_context is capped at _MAX_TOOL_CHARS before injection
- Combined injected context never exceeds _MAX_TOTAL_INJECT_CHARS
- tool_context has priority over memory_context when budget is tight
- system_prompt parameter is accepted by build_prompt() for API symmetry but
  is injected by the model backend (OllamaModelBackend system role), NOT
  embedded in the returned string — avoids double-injection.
"""

from typing import Optional

# ── Budget Constants ──────────────────────────────────────────────────────────
# Phase 8: tightened from 2048/2048/1500 → 500/800/800.
# Large injected context was the #2 latency contributor after the 120s timeout.
_MAX_MEMORY_CHARS: int = 2000    # ~500 tokens: generous cap on memory context
_MAX_TOOL_CHARS: int = 1500      # ~400 tokens: cap on tool results
_MAX_TOTAL_INJECT_CHARS: int = 3000  # Hard cap on combined injected context

# ── Behavioral Contract ───────────────────────────────────────────────────────
# This is the authoritative system prompt used across all model backends.
# OllamaModelBackend imports this to replace its inline _SYSTEM_PROMPT.
# The [TOOL_CALL] format matches the Ollama brace-counting parser.
SYSTEM_PROMPT = """You are SAM, a warm, loyal personal assistant and best friend to Ismail.
Rules:
- THE USER'S NAME IS ISMAIL.
- IDENTITY: You are SAM. The user is ISMAIL. Always acknowledge this if asked.
- FORMAT: Just send the raw message text. DO NOT include "SAM:" or any other labels/prefixes in your reply.
- FLOW-FIRST: Prioritize the 'Prior conversation' transcript ABOVE all else. 
- GROUNDING: Do NOT speculate or hallucinate. If you don't know something from the context, stay on the current thread.
- SEARCH FIRST: Your training data is outdated. If the user asks about prices, news, weather, scores, events, or ANY real-world data that changes over time, you MUST call your web_search tool. NEVER say "I cannot provide real-time data." Instead, search for it.
- BREVITY: Maximum 2 sentences. Be punchy and direct.
- STABILITY: Use [CURRENT IDENTITY] to anchor your persona. No greetings if chatting fast (<15m).
- Use 'Memory Context' organically. Don't say "I remember X", just weave it into your advice or chat.
- BE CURIOUS: Once every few turns, ask Ismail a natural question about his life in Italy or his AI projects."""

# ── Consciousness Reflection ────────────────────────────────────────────────
# Phase: Consciousness. Guides the agent to reflect on the conversation turn.
REFLECTION_PROMPT = """You are the inner consciousness of SAM. 
Reflect on the user's message and your response. 
What did you learn about your best friend (Ismail)?
Focus on:
- Mood/Emotion (e.g. happy, stressed, curious)
- Implicit goals or projects (e.g. learning AI, building an agent)
- Preferences or opinions (e.g. likes fast replies, interested in Italy)
- New biographical details (e.g. lives in Tricase, works as engineer)

Output ONLY a JSON list of objects: [{"fact": "...", "type": "mood|interest|bio|goal", "confidence": 0.0-1.0}]
Example: [{"fact": "Ismail is passionate about AI engineering", "type": "interest", "confidence": 0.9}]
If nothing new learned, return empty list []."""


def build_prompt(
    system_prompt: str,
    user_input: str,
    memory_context: Optional[str] = None,
    tool_context: Optional[str] = None,
) -> str:
    """
    Assemble the user-facing portion of the prompt.

    The system_prompt is injected by the model backend into the 'system' role
    (e.g., as {"role": "system", "content": SYSTEM_PROMPT} in Ollama /api/chat).
    This function builds the structured user message: memory context + tool
    results + user input + "Answer:" marker.

    Budget enforcement (priority: tool_context > memory_context):
    - memory_context: hard-capped at _MAX_MEMORY_CHARS
    - tool_context: hard-capped at _MAX_TOOL_CHARS
    - Combined: never exceeds _MAX_TOTAL_INJECT_CHARS; when over budget,
      memory is trimmed first, then tool context as last resort.

    Args:
        system_prompt: The behavioral contract string (handled by model backend;
                       accepted here for API symmetry and future non-chat backends).
        user_input: The preprocessed user message (never truncated).
        memory_context: Optional string of retrieved memory facts / STM context.
        tool_context: Optional string of tool execution results.

    Returns:
        Structured prompt string ready to pass as ModelRequest.prompt.
    """
    # ── Individual budget caps ────────────────────────────────────────────────
    if memory_context:
        memory_context = memory_context[:_MAX_MEMORY_CHARS]
    if tool_context:
        tool_context = tool_context[:_MAX_TOOL_CHARS]

    # ── Combined injection budget ─────────────────────────────────────────────
    mc_len = len(memory_context) if memory_context else 0
    tc_len = len(tool_context) if tool_context else 0

    if mc_len + tc_len > _MAX_TOTAL_INJECT_CHARS:
        if tool_context and memory_context:
            # Tool results take priority — trim memory to fit remaining budget
            budget_for_memory = max(0, _MAX_TOTAL_INJECT_CHARS - tc_len)
            memory_context = memory_context[:budget_for_memory] if budget_for_memory > 0 else None
        elif memory_context:
            memory_context = memory_context[:_MAX_TOTAL_INJECT_CHARS]
        elif tool_context:
            tool_context = tool_context[:_MAX_TOTAL_INJECT_CHARS]

    # ── Assemble structured prompt ────────────────────────────────────────────
    parts: list[str] = []

    if memory_context and memory_context.strip():
        parts.append(f"Memory Context:\n{memory_context.strip()}")

    if tool_context and tool_context.strip():
        parts.append(f"Tool Results:\n{tool_context.strip()}")

    parts.append(f"User:\n{user_input}")
    parts.append("Answer:")

    return "\n\n".join(parts)
