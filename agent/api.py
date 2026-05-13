"""
Agent API entrypoint for deployment.

Serves:
- /health/live: Liveness probe
- /health/ready: Readiness probe
- /invoke: Agent invocation endpoint (future)

Can be run as a module:
  python -m agent.api
"""

import sys
import os
import logging
import argparse
from typing import Optional


def create_app():
    """Create FastAPI application."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        
        app = FastAPI(
            title="SAM Agent API",
            description="Stateful Agent Model API",
            version="0.0.1"
        )
        
        # Initialize health checker
        from agent.health import (
            initialize_health_checker,
            health_live,
            health_ready
        )
        
        initialize_health_checker()
        
        # Health endpoints
        @app.get("/health/live")
        async def live():
            """Liveness probe."""
            result = await health_live()
            status_code = 200 if result["status"] == "healthy" else 503
            return JSONResponse(content=result, status_code=status_code)
        
        @app.get("/health/ready")
        async def ready():
            """Readiness probe."""
            result = await health_ready()
            status_code = 200 if result["status"] == "healthy" else 503
            return JSONResponse(content=result, status_code=status_code)
        
        @app.get("/health/trace")
        async def trace_health():
            """Trace/observability configuration endpoint (read-only)."""
            try:
                from agent.tracing import get_tracer_config
                config = get_tracer_config()
                return JSONResponse(content=config, status_code=200)
            except Exception as e:
                return JSONResponse(
                    content={"error": str(e), "tracer_backend": "noop"},
                    status_code=500
                )
        
        @app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "SAM Agent API",
                "version": "0.0.1",
                "endpoints": [
                    "/health/live",
                    "/health/ready",
                    "/health/trace",
                    "/invoke",
                    "/webhook/whatsapp",
                    "/webhook/telegram",
                    "/webhook/telegram/voice"
                ]
            }
        
        # ===================================================================
        # WHATSAPP TRANSPORT INTEGRATION (PURE I/O LAYER)
        # ===================================================================
        
        try:
            from transport.whatsapp import router as whatsapp_router
            app.include_router(whatsapp_router)
        except ImportError:
            pass  # WhatsApp transport not available
        
        # ===================================================================
        # TELEGRAM TRANSPORT INTEGRATION (PURE I/O LAYER)
        # ===================================================================
        
        try:
            from webhook.telegram import router as telegram_router
            app.include_router(telegram_router)
            
            # Start Autonomous Heartbeat (Morning Messages)
            @app.on_event("startup")
            async def start_heartbeat():
                import asyncio
                from agent.intelligence.autonomous_heartbeat import AutonomousHeartbeat
                from config import Config
                
                # Admin chat ID from config
                admin_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "903341171")
                heartbeat = AutonomousHeartbeat(admin_chat_id)
                
                startup_logger = logging.getLogger("agent")
                startup_logger.info("Main: Starting Autonomous Heartbeat background service...")
                asyncio.create_task(heartbeat.run_forever())
                
        except ImportError:
            pass  # Telegram text handler not available
        
        try:
            from webhook.telegram_voice import voice_router
            app.include_router(voice_router)
        except ImportError:
            pass  # Telegram voice handler not available
        
        @app.post("/invoke")
        async def invoke(request: dict):
            """Invoke agent with user input."""
            from agent.langgraph_orchestrator import SAMAgentOrchestrator
            from agent.state_schema import AgentState
            from uuid import uuid4
            from datetime import datetime
            import os
            
            try:
                user_input = request.get("input", "")
                
                if not user_input:
                    return JSONResponse(
                        content={"error": "Missing 'input' field"},
                        status_code=400
                    )
                
                # Initialize agent with appropriate backend
                llm_backend = os.getenv("LLM_BACKEND", "stub").lower()
                
                if llm_backend == "ollama":
                    from inference import OllamaModelBackend
                    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    ollama_model = os.getenv("OLLAMA_MODEL", "phi")
                    backend = OllamaModelBackend(model_name=ollama_model, base_url=ollama_url)
                else:
                    # Default to stub
                    from inference import StubModelBackend
                    backend = StubModelBackend()
                
                # Initialize memory backends
                stm_backend = os.getenv("STM_BACKEND", "stub").lower()
                ltm_backend = os.getenv("LTM_BACKEND", "stub").lower()
                
                memory_controller = None
                long_term_memory = None
                
                if stm_backend == "sqlite":
                    from agent.memory import SQLiteShortTermMemoryStore
                    db_path = os.getenv("SQLITE_DB_PATH", os.getenv("DATABASE_PATH", "/app/data/memory.db"))
                    memory_controller = SQLiteShortTermMemoryStore(db_path=db_path)
                else:
                    from agent.memory import StubMemoryController
                    memory_controller = StubMemoryController()
                
                if ltm_backend == "qdrant":
                    from agent.memory import QdrantLongTermMemoryStore
                    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
                    long_term_memory = QdrantLongTermMemoryStore(qdrant_url=qdrant_url)
                else:
                    from agent.memory import StubLongTermMemoryStore
                    long_term_memory = StubLongTermMemoryStore()
                
                agent = SAMAgentOrchestrator(
                    model_backend=backend,
                    memory_controller=memory_controller,
                    long_term_memory_store=long_term_memory
                )
                
                # Create initial state
                state = AgentState(
                    conversation_id=str(uuid4()),
                    trace_id=str(uuid4()),
                    created_at=datetime.now().isoformat(),
                    input_type="text",
                    raw_input=user_input
                )
                
                # Invoke agent (synchronous graph execution)
                result = agent.graph.invoke(state)
                
                # model output source: AgentState.final_output (written by result_handling_node)
                # result dict is the final merged state from the orchestrator
                
                return JSONResponse(
                    content={
                        "status": "success",
                        "input": user_input,
                        "output": result.get("final_output", ""),
                        "conversation_id": result.get("conversation_id", ""),
                        "trace_id": result.get("trace_id", "")
                    },
                    status_code=200
                )
            
            except Exception as e:
                import traceback
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": str(e),
                        "type": type(e).__name__,
                        "traceback": traceback.format_exc()
                    },
                    status_code=500
                )
        
        # ===================================================================
        # OBSERVABILITY ENDPOINTS (LOCAL DEVELOPMENT ONLY)
        # ===================================================================
        
        # Check if observability is enabled
        import os
        observability_enabled = os.getenv("LOCAL_OBSERVABILITY_ENABLED", "false").lower() == "true"
        
        if observability_enabled:
            from agent.observability import LocalObservabilityInterface, get_observability, set_observability, ObservabilityStore
            
            # Initialize global observability on first request
            _observability_instance = [None]  # Use list to allow mutation in nested function
            
            def get_or_create_observability():
                if _observability_instance[0] is None:
                    store = ObservabilityStore()
                    _observability_instance[0] = LocalObservabilityInterface(store=store)
                    set_observability(_observability_instance[0])
                return _observability_instance[0]
            
            @app.get("/debug/health")
            async def debug_health():
                """Get agent health and configuration."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    obs = get_or_create_observability()
                    # Get agent instance from last invocation or global
                    return JSONResponse(
                        content={"status": "healthy", "observability_enabled": True},
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"status": "error", "error": str(e)},
                        status_code=500
                    )
            
            @app.get("/debug/graph")
            async def debug_graph():
                """Get graph structure (static, no execution state)."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    from agent.langgraph_orchestrator import SAMAgentOrchestrator
                    from inference import StubModelBackend
                    
                    # Create a minimal agent to inspect structure
                    agent = SAMAgentOrchestrator(model_backend=StubModelBackend())
                    obs = get_or_create_observability()
                    return JSONResponse(
                        content=obs.get_graph_structure(agent),
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"error": str(e)},
                        status_code=500
                    )
            
            @app.get("/debug/traces")
            async def debug_traces(limit: int = 50):
                """Get recent trace metadata (no content)."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    obs = get_or_create_observability()
                    return JSONResponse(
                        content={
                            "recent_traces": obs.get_recent_traces(limit),
                            "active_traces": obs.get_active_traces(),
                            "limit": limit,
                        },
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"error": str(e)},
                        status_code=500
                    )
            
            @app.get("/debug/spans")
            async def debug_spans(limit: int = 100):
                """Get recent span metadata (no content)."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    obs = get_or_create_observability()
                    return JSONResponse(
                        content={
                            "recent_spans": obs.get_recent_spans(limit),
                            "limit": limit,
                        },
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"error": str(e)},
                        status_code=500
                    )
            
            @app.get("/debug/memory")
            async def debug_memory(limit: int = 100):
                """Get memory operation metadata (no content)."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    obs = get_or_create_observability()
                    return JSONResponse(
                        content={
                            "memory_events": obs.get_memory_events(limit),
                            "limit": limit,
                        },
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"error": str(e)},
                        status_code=500
                    )
            
            @app.get("/debug/stats")
            async def debug_stats():
                """Get observability store statistics."""
                if not observability_enabled:
                    return JSONResponse({"error": "Observability disabled"}, status_code=404)
                
                try:
                    obs = get_or_create_observability()
                    return JSONResponse(
                        content=obs.get_store_stats(),
                        status_code=200
                    )
                except Exception as e:
                    return JSONResponse(
                        content={"error": str(e)},
                        status_code=500
                    )
        
        return app
    
    except ImportError as e:
        print(f"FastAPI not available: {e}", file=sys.stderr)
        print("Install with: pip install fastapi uvicorn", file=sys.stderr)
        return None


def main(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run agent API server."""
    # ── Logging setup (must run before uvicorn reconfigures logging) ──────────
    _log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    _log_fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=_log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Pin a StreamHandler to the 'agent' namespace so [LATENCY]/[DMA] logs
    # survive Uvicorn's dictConfig overwrite of the root logger.
    _agent_logger = logging.getLogger("agent")
    _agent_logger.setLevel(_log_level)
    if not _agent_logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(_log_fmt)
        _agent_logger.addHandler(_h)

    app = create_app()
    
    if app is None:
        print("Failed to create app. Check dependencies.", file=sys.stderr)
        sys.exit(1)
    
    try:
        import uvicorn
        
        print(f"Starting SAM Agent API on {host}:{port}")
        print(f"Health check: http://{host}:{port}/health/live")
        print(f"Readiness check: http://{host}:{port}/health/ready")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    
    except ImportError:
        print("Uvicorn not available. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM Agent API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    
    args = parser.parse_args()
    main(host=args.host, port=args.port, reload=args.reload)
