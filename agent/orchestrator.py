"""
SAM Agent Orchestrator

Main entry point for the agent control flow. This module is the public API
for invoking the agent.

The actual graph implementation is in langgraph_orchestrator.py, which
implements the exact structure defined in design/langgraph_skeleton.md.
"""

from typing import Optional, Dict, Any
from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference import ModelBackend, StubModelBackend
from inference.ollama import OllamaModelBackend
from config import Config
import logging

logger = logging.getLogger(__name__)


def _default_model_backend() -> ModelBackend:
    """Return the configured model backend based on LLM_BACKEND env var."""
    if Config.LLM_BACKEND == "ollama":
        return OllamaModelBackend(
            model_name=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
        )
    return StubModelBackend()


def _default_stm_store():
    """
    Return a SQLiteShortTermMemoryStore backed by the configured DB path.
    Falls back to StubMemoryController gracefully if SQLite fails.
    """
    try:
        from agent.memory.sqlite import SQLiteShortTermMemoryStore
        db_path = Config.SQLITE_DB_PATH
        store = SQLiteShortTermMemoryStore(db_path=db_path)
        logger.info(f"STM: SQLiteShortTermMemoryStore initialised at {db_path}")
        return store
    except Exception as e:
        from agent.memory.stub import StubMemoryController
        logger.warning(f"STM: falling back to StubMemoryController ({e})")
        return StubMemoryController()


def _default_ltm_store():
    """
    Return a QdrantLongTermMemoryStore when LTM_BACKEND=qdrant (the default),
    otherwise a StubLongTermMemoryStore.

    Delegates to InfraConfig so all env-var reading is in one place.
    Falls back to the stub gracefully if qdrant-client is not installed or
    Qdrant is unreachable at startup.
    """
    try:
        from infra.config import InfraConfig
        return InfraConfig.from_env().create_ltm_backend()
    except Exception:
        from agent.memory.long_term_stub import StubLongTermMemoryStore
        return StubLongTermMemoryStore()


def _default_tracer():
    """Return the configured tracer backend."""
    from agent.tracing.tracer_factory import create_tracer
    return create_tracer()


class SAMOrchestrator:
    """
    Main orchestrator for the SAM agent.
    
    Public API for agent invocation. Delegates to LangGraph implementation.
    """
    
    def __init__(self, model_backend: Optional[ModelBackend] = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            model_backend: ModelBackend instance (uses configured backend by default)
        """
        self.langgraph_orchestrator = SAMAgentOrchestrator(
            model_backend=model_backend or _default_model_backend(),
            memory_controller=_default_stm_store(),
            long_term_memory_store=_default_ltm_store(),
            tracer=_default_tracer(),
        )
    
    async def invoke(self, raw_input: str, conversation_id: Optional[str] = None, trace_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a single agent invocation.
        """
        import asyncio
        
        # Execute main conversation graph
        result = await self.langgraph_orchestrator.invoke(
            raw_input=raw_input,
            conversation_id=conversation_id,
            trace_id=trace_id,
        )
        
        # ── Consciousness Reflection (Background) ─────────────────────────────
        # Trigger deep reflection after a short delay so it doesn't hog CPU during the reply.
        async def delayed_reflect():
            await asyncio.sleep(5)  # Let the response settle
            try:
                await asyncio.to_thread(self.langgraph_orchestrator.reflect, result)
            except Exception as e:
                logger.warning(f"Failed to run reflection: {e}")

        def _on_reflection_done(task: asyncio.Task) -> None:
            """Log any unhandled exception that escaped the reflection coroutine."""
            if task.cancelled():
                logger.debug("Reflection task was cancelled (likely shutdown)")
                return
            exc = task.exception()
            if exc is not None:
                logger.error(
                    "Background reflection task raised an unhandled exception: %s",
                    exc,
                    exc_info=exc,
                )

        try:
            task = asyncio.create_task(delayed_reflect())
            task.add_done_callback(_on_reflection_done)
        except Exception as e:
            logger.warning(f"Failed to start background reflection task: {e}")

        return result
