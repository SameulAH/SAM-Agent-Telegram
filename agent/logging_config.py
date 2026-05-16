"""
Structured logging configuration for SAM Agent.

Usage:
    from agent.logging_config import configure_logging
    configure_logging()  # call once at startup, before any loggers are created

Behaviour:
    - LOG_FORMAT=json  → JSON lines (one object per log record) — recommended for production
    - LOG_FORMAT=text  → Human-readable text (default, same as before)
    - LOG_LEVEL        → DEBUG / INFO / WARNING / ERROR (default INFO)

The JSON formatter adds:
    timestamp, level, logger, message, trace_id, conversation_id (when available)

trace_id / conversation_id are injected via Python's logging.LoggerAdapter or by
including them in the `extra` dict when calling logger.info(..., extra={...}).
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    _CORE_ATTRS = frozenset(
        {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Promote trace context fields if present in the record's extra dict
        for field in ("trace_id", "conversation_id"):
            val = getattr(record, field, None)
            if val:
                payload[field] = val

        # Include any other extra fields that aren't standard LogRecord attrs
        for key, val in record.__dict__.items():
            if key not in self._CORE_ATTRS and not key.startswith("_"):
                payload.setdefault(key, val)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure root logger based on LOG_FORMAT and LOG_LEVEL env vars.

    Call once at application startup. Idempotent — safe to call multiple times.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    if log_format == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Avoid adding duplicate handlers if called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(log_level)

    # Pin a handler on the 'agent' namespace so it survives uvicorn's dictConfig reset.
    # propagate=False prevents records from also being handled by the root logger,
    # which would double-print every agent.* log line.
    agent_logger = logging.getLogger("agent")
    agent_logger.setLevel(log_level)
    agent_logger.propagate = False
    if not agent_logger.handlers:
        agent_handler = logging.StreamHandler(sys.stdout)
        agent_handler.setFormatter(formatter)
        agent_logger.addHandler(agent_handler)
