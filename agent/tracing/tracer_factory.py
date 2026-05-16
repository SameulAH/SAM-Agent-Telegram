"""
Tracer factory and initialization logic.

Handles environment-based tracer selection and instantiation.

Implements TRACER_BACKEND setting:
- "noop" (default): No observability
- "langsmith": LangSmith observability
- "langtrace" (future): Langtrace observability
"""

import os
from typing import Optional

from agent.tracing import Tracer, NoOpTracer, LangSmithTracer


def get_tracer_backend() -> str:
    """
    Get the configured tracer backend.

    Environment Variable:
        TRACER_BACKEND: "noop" (default), "langsmith", or "langtrace"

    Returns:
        Backend name (lowercase)
    """
    backend = os.getenv("TRACER_BACKEND", "noop").lower().strip()

    # Validate backend
    valid_backends = {"noop", "langsmith", "langtrace", "otel"}
    if backend not in valid_backends:
        # Unknown backend, default to noop
        return "noop"

    return backend


def create_tracer(observability_sink=None) -> Tracer:
    """
    Create a tracer instance based on environment configuration.

    Args:
        observability_sink: Optional callback for local observability recording

    Returns:
        Tracer instance (never None, defaults to NoOpTracer)

    Environment Variables:
        TRACER_BACKEND: "noop" (default), "langsmith", "langtrace", or "otel"
        OTEL_EXPORTER_OTLP_ENDPOINT: Defaults to http://otel-collector:4317
    """
    backend = get_tracer_backend()

    try:
        # Check for legacy LANGSMITH_ENABLED variable (backward compatibility)
        langsmith_enabled = os.getenv("LANGSMITH_ENABLED", "false").lower() == "true"

        if backend == "langsmith" or (backend == "noop" and langsmith_enabled):
            # Try LangSmith if explicitly selected or if legacy flag is set
            return LangSmithTracer(enabled=True, observability_sink=observability_sink)

        elif backend == "otel":
            from agent.tracing.otel_tracer import OtelTracer
            endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
            return OtelTracer(endpoint=endpoint, observability_sink=observability_sink)

        elif backend == "langtrace":
            # Langtrace backend (currently a no-op placeholder)
            # When implemented, import and instantiate here
            # from agent.tracing import LangtraceTracer
            # return LangtraceTracer(enabled=True, observability_sink=observability_sink)
            #
            # For now, fall through to noop
            return NoOpTracer(observability_sink=observability_sink)

        else:
            # Default to noop
            return NoOpTracer(observability_sink=observability_sink)

    except Exception as e:
        # Tracer initialization failure is non-fatal
        # Always fall back to no-op
        print(f"Warning: Failed to initialize tracer backend '{backend}': {e}")
        return NoOpTracer(observability_sink=observability_sink)


def get_tracer_config() -> dict:
    """
    Get current tracer configuration for health/debug endpoints.

    Returns:
        Dict with tracer status and configuration
    """
    backend = get_tracer_backend()

    config = {
        "tracer_backend": backend,
        "enabled": backend != "noop",
    }

    if backend == "langsmith":
        api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
        config["langsmith_configured"] = bool(api_key)
        config["langsmith_project"] = os.getenv("LANGSMITH_PROJECT", "sam-agent")

    elif backend == "langtrace":
        config["langtrace_configured"] = False  # Not yet implemented
        config["note"] = "Langtrace backend not yet implemented"

    return config
