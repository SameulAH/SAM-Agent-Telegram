"""
FastAPI Application Entry Point

Integrates:
  - Telegram webhook handler
  - Health checks
  - Agent endpoints
  - Middleware for logging & error handling

Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from agent.logging_config import configure_logging
from webhook.telegram import router as telegram_router
from webhook.telegram_voice import voice_router
from config import Config

# Configure logging once at startup (format + level from env vars)
configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: startup and shutdown handlers.
    """
    # Startup
    logger.info("=" * 60)
    logger.info("SAM Agent starting up...")
    logger.info(f"Telegram Bot: @{Config.TELEGRAM_BOT_USERNAME}")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"LLM Backend: {Config.LLM_BACKEND}")
    logger.info("=" * 60)
    
    yield
    
    # Shutdown
    logger.info("SAM Agent shutting down...")


# Create FastAPI app
app = FastAPI(
    title="SAM Agent API",
    description="Secure AI Agent for messaging platforms",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
# Origins are configured via ALLOWED_ORIGINS env var (comma-separated).
# Defaults to "*" for local dev; set a restricted list in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware for logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    logger.debug(f"{request.method} {request.url.path}")
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )


# Include routers
app.include_router(telegram_router)
app.include_router(voice_router)


# Health check endpoints
@app.get("/health/live")
async def health_live():
    """Live health check (Kubernetes liveness probe)."""
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready():
    """Readiness health check (Kubernetes readiness probe)."""
    try:
        # Check configuration
        Config.validate()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "reason": str(e)}


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "SAM Agent API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "telegram_webhook": "POST /webhook/telegram",
            "telegram_voice": "POST /webhook/telegram/voice",
            "telegram_health": "GET /webhook/telegram/health",
            "voice_health": "GET /webhook/telegram/voice/health",
            "health_live": "GET /health/live",
            "health_ready": "GET /health/ready",
        },
    }


@app.get("/config/info")
async def config_info():
    """Get non-sensitive configuration info."""
    return {
        "environment": Config.ENVIRONMENT,
        "llm_backend": Config.LLM_BACKEND,
        "telegram_bot": f"@{Config.TELEGRAM_BOT_USERNAME}",
        "agent_port": Config.AGENT_PORT,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=Config.AGENT_PORT,
        reload=Config.ENVIRONMENT == "development",
    )
