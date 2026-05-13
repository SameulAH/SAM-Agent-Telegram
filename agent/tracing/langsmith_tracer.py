"""
LangSmith-backed tracing implementation.

Provides production-grade observability via LangSmith.

Frozen constraints:
- Never influences control flow
- Never mutates state
- Failures are silent and non-fatal
- Only allowed data is traced
- No raw prompts or outputs leaked
"""

import os
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from agent.tracing.tracer import Tracer, TraceMetadata, NoOpTracer


class LangSmithTracer(Tracer):
    """
    LangSmith implementation of the Tracer interface.

    Provides production-grade observability while respecting all constraints:
    - No control flow influence
    - No state mutation
    - Silent failures
    - Metadata-only logging

    Safe to fail silently → agent continues unchanged.
    """

    def __init__(self, enabled: bool = True, observability_sink: Optional[Callable] = None):
        """
        Initialize LangSmith tracer.

        Args:
            enabled: Whether to enable LangSmith tracing
            observability_sink: Optional callback for recording to local observability store

        Environment Variables:
            LANGSMITH_API_KEY: LangSmith API key (required if enabled)
            LANGSMITH_PROJECT: LangSmith project name (optional, defaults to "sam-agent")
        """
        super().__init__(observability_sink)
        self._enabled = enabled
        self._langsmith_client = None
        self._project_name = None

        if enabled:
            try:
                from langsmith import Client

                api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
                if not api_key:
                    # No API key, silently disable
                    self._enabled = False
                else:
                    self._langsmith_client = Client(api_key=api_key)
                    self._project_name = os.getenv("LANGSMITH_PROJECT", "sam-agent")
            except (ImportError, Exception):
                # LangSmith unavailable or import failed
                # Silently degrade to no-op (non-fatal)
                self._enabled = False

    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """
        Start a LangSmith trace span.

        Logs structured metadata only (no prompts, outputs, or raw state).

        Best-effort only. Failures are silent.

        Args:
            name: Span name (e.g., "router_node", "model_call")
            metadata: Structural metadata (duration, status, etc.)
            trace_metadata: Trace identity (trace_id, conversation_id)

        Returns:
            Span handle for use in end_span, or None if tracing disabled
        """
        if not self._enabled or self._langsmith_client is None:
            return None

        try:
            # Record to local observability sink if available
            if self.observability_sink:
                try:
                    self.observability_sink(
                        "span_start",
                        {
                            "trace_id": trace_metadata.trace_id,
                            "conversation_id": trace_metadata.conversation_id,
                            "node_name": name,
                        },
                    )
                except Exception:
                    pass  # Observability failure is non-fatal

            # Build LangSmith input with ONLY allowed metadata
            # Safe: structural information only
            trace_input = {
                "node_name": name,
                "trace_id": trace_metadata.trace_id,
                "conversation_id": trace_metadata.conversation_id or "",
            }

            # Filter out any sensitive fields
            safe_metadata = self._filter_safe_metadata(metadata)

            # Create run in LangSmith
            run_id = self._langsmith_client.create_run(
                name=name,
                run_type="chain",
                inputs=trace_input,
                project_name=self._project_name,
                id=None, # Let LangSmith generate or we could use trace_metadata.trace_id if unique per span
                extra=safe_metadata,
                start_time=datetime.utcnow()
            )

            return run_id

        except Exception as e:
            # Tracing failure is non-fatal
            return None

    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """
        End a LangSmith trace span.

        Best-effort only. Failures are silent.

        Args:
            span: Span handle from start_span
            status: "success", "failure", or "skipped"
            metadata: Execution results (duration_ms, error_type, etc.)
        """
        if not self._enabled or span is None:
            return

        try:
            # Record to local observability sink if available
            if self.observability_sink:
                try:
                    self.observability_sink(
                        "span_end",
                        {
                            "trace_id": "unknown",  # span handle is now just an ID or object
                            "span_name": "unknown",
                            "status": status,
                            "duration_ms": metadata.get("duration_ms"),
                        },
                    )
                except Exception:
                    pass  # Observability failure is non-fatal

            # Build safe output (no raw data)
            safe_metadata = self._filter_safe_metadata(metadata)

            # Update run in LangSmith
            self._langsmith_client.update_run(
                run_id=span,  # span is the run_id returned by create_run
                outputs=safe_metadata,
                error=metadata.get("error_message") if status == "failure" else None,
                end_time=datetime.utcnow()
            )

        except Exception:
            # Tracing failure is non-fatal
            pass

    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """
        Record a point-in-time event (e.g., memory operation, invariant check).

        Best-effort only. Failures are silent.

        Args:
            name: Event name (e.g., "memory_write_authorized")
            metadata: Event data
            trace_metadata: Trace identity
        """
        if not self._enabled or self._langsmith_client is None:
            return

        try:
            # Build safe event data
            safe_metadata = self._filter_safe_metadata(metadata)

            event_data = {
                "event_name": name,
                "trace_id": trace_metadata.trace_id,
                "conversation_id": trace_metadata.conversation_id or "",
            }
            event_data.update(safe_metadata)

            # Record as a small run in LangSmith
            self._langsmith_client.create_run(
                name=name,
                run_type="chain",
                inputs=event_data,
                outputs={"status": "recorded"},
                project_name=self._project_name,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )

        except Exception:
            # Tracing failure is non-fatal
            pass

    def is_enabled(self) -> bool:
        """
        Check if LangSmith tracing is enabled.

        Returns:
            True if LangSmith is enabled and available, False otherwise
        """
        return self._enabled

    @staticmethod
    def _filter_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter metadata to include ONLY safe fields.

        Safe fields (allowed to trace):
        - duration_ms, status, node_name
        - error_type, error_message (without traceback)
        - operation (memory operation type)
        - authorized, backend

        Unsafe fields (NEVER traced):
        - raw_input, input, prompt (user messages)
        - output, response (model outputs)
        - memory (stored data)
        - decision (routing logic)

        Args:
            metadata: Unfiltered metadata dict

        Returns:
            Filtered dict with only safe fields
        """
        safe_fields = {
            "duration_ms",
            "status",
            "node_name",
            "error_type",
            "error_message",
            "operation",
            "authorized",
            "backend",
            "total_nodes",
            "node_count",
        }

        filtered = {}
        for key, value in metadata.items():
            if key in safe_fields:
                # Sanitize value if it's a string
                if isinstance(value, str):
                    # Truncate very long strings
                    if len(value) > 256:
                        filtered[key] = value[:256] + "..."
                    else:
                        filtered[key] = value
                elif isinstance(value, (int, float, bool)):
                    filtered[key] = value
                else:
                    # For other types, convert to string and truncate
                    filtered[key] = str(value)[:256]

        return filtered
