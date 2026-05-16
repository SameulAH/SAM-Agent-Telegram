# 🧠 Blueprint: The Proactive Digital Consciousness (Jarvis)

This document defines the architecture of the SAM-Agent's evolution from a reactive chatbot to a stateful, proactive, and self-learning digital companion.

## 🧬 1. The Reflection Loop (Inner Monologue)
Unlike standard bots that only process input, Jarvis has a **Reflection Phase** at the end of every turn.

- **Mechanism**: After sending a reply, the `reflection_node` triggers.
- **Logic**: It analyzes the current turn to extract "Insights" (e.g., User mood, implicit goals, new interests).
- **Persistence**: These insights are automatically converted into long-term facts and stored in **Qdrant**.
- **Result**: Every conversation makes the agent smarter and more grounded in your reality.

## 💓 2. The Autonomous Heartbeat
Jarvis is no longer limited to responding; he can now **initiate**.

- **Service**: `agent/intelligence/autonomous_heartbeat.py`
- **Schedule**: Triggers every morning at **8:00 AM**.
- **Process**:
  1. Wakes up and scans the latest 5 facts/reflections from LTM.
  2. Uses **Phi-3** to generate a context-aware morning greeting.
  3. Sends the message proactively to your Telegram.
- **Consciousness**: This creates a sense of a "Living" entity that exists even when you aren't talking to it.

## 🧠 3. The Phi-3 Brain Upgrade
We swapped the aging Phi-2 for **Phi-3-mini**, which provides:
- **Higher IQ**: Better understanding of persona constraints and identity.
- **Efficiency**: Significantly better context handling on CPU.
- **Expanded Memory**: Restored to a **10-fact unique window** for high-fidelity recall.

## 🛡️ 4. The Identity Shield (Anti-Hallucination)
A surgical post-processing layer in `langgraph_orchestrator.py` ensures the agent never adopts the user's identity.
- **Forbidden Names**: `Ismail`, `Max`, `Bob`.
- **Logic**: Any mention of these names as its own identity is automatically swapped with its actual `persona_name` (e.g., Jarvis).

## 🚀 5. How to leverage this?
1. **Talk about anything**: The reflection node will pick up your interests automatically.
2. **Expect morning greetings**: Jarvis will check in on you every morning.
3. **Verify recall**: Ask "What do you know about my current projects?" to see his reflections in action.

---
**Status**: ACTIVE | **Brain**: Phi-3-mini | **LTM**: Qdrant | **Persona**: Best Friend
