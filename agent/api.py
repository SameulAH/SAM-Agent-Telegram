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
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse

        app = FastAPI(
            title="SAM Agent API",
            description="Stateful Agent Model API",
            version="0.0.1"
        )

        # CORS — driven by ALLOWED_ORIGINS env var (comma-separated).
        # Defaults to "*" for local dev; restrict in production via env.
        _raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
        _allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
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
                
                # Use ainvoke so the async FastAPI handler doesn't trigger
                # LangGraph's thread-pool pickle path for non-serialisable objects.
                result = await agent.graph.ainvoke(state)
                
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
        observability_enabled = os.getenv("LOCAL_OBSERVABILITY_ENABLED", "false").lower() == "true"
        _debug_token = os.getenv("DEBUG_API_TOKEN", "")

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

            def _verify_debug_token(x_debug_token: str = Header(None)) -> None:
                """Require X-Debug-Token header when DEBUG_API_TOKEN is configured."""
                if _debug_token and x_debug_token != _debug_token:
                    raise HTTPException(status_code=403, detail="Invalid or missing debug token")

            @app.get("/debug/health")
            async def debug_health(x_debug_token: str = Header(None)):
                """Get agent health and configuration."""
                _verify_debug_token(x_debug_token)
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
            async def debug_graph(x_debug_token: str = Header(None)):
                """Get graph structure (static, no execution state)."""
                _verify_debug_token(x_debug_token)
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
            async def debug_traces(limit: int = 50, x_debug_token: str = Header(None)):
                """Get recent trace metadata (no content)."""
                _verify_debug_token(x_debug_token)
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
            async def debug_spans(limit: int = 100, x_debug_token: str = Header(None)):
                """Get recent span metadata (no content)."""
                _verify_debug_token(x_debug_token)
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
            async def debug_memory(limit: int = 100, x_debug_token: str = Header(None)):
                """Get memory operation metadata (no content)."""
                _verify_debug_token(x_debug_token)
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
            async def debug_stats(x_debug_token: str = Header(None)):
                """Get observability store statistics."""
                _verify_debug_token(x_debug_token)
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
    # Configure logging once (format + level from LOG_FORMAT / LOG_LEVEL env vars).
    # Must run before uvicorn reconfigures the root logger.
    from agent.logging_config import configure_logging
    configure_logging()

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
