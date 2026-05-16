"""
OpenTelemetry implementation of the Tracer interface.

Exports spans to the OpenTelemetry collector.
"""

import logging
from typing import Dict, Any, Optional, Callable
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode

from agent.tracing.tracer import Tracer, TraceMetadata

logger = logging.getLogger(__name__)

class OtelTracer(Tracer):
    """
    OpenTelemetry tracer implementation.
    
    Provides distributed tracing with OTLP export.
    Fail-safe: all operations are wrapped in try-except to never block the agent.
    """

    def __init__(
        self, 
        service_name: str = "sam-agent",
        endpoint: str = "http://otel-collector:4317",
        observability_sink: Optional[Callable] = None
    ):
        """Initialize OTel tracer."""
        super().__init__(observability_sink)
        self._enabled = False
        
        try:
            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)
            
            # Use insecure=True for internal docker communication if not using TLS
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
            span_processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(span_processor)
            
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)
            self._enabled = True
            logger.info(f"OtelTracer initialized, exporting to {endpoint}")
        except Exception as e:
            logger.warning(f"Failed to initialize OtelTracer: {str(e)}. Falling back to no-op.")
            self._enabled = False

    def start_span(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> Optional[Any]:
        """Start an OTel span."""
        if not self._enabled:
            return None
            
        try:
            # Create attributes from metadata and trace_metadata
            attributes = {
                "trace_id": trace_metadata.trace_id,
                "conversation_id": trace_metadata.conversation_id or "",
                **metadata
            }
            
            # We don't use the trace_metadata.trace_id as the actual OTel trace ID here
            # because OTel manages its own ID generation/propagation.
            # However, we attach our internal IDs as attributes for correlation.
            span = self._tracer.start_span(name, attributes=attributes)
            return span
        except Exception:
            return None

    def end_span(self, span: Any, status: str, metadata: Dict[str, Any]) -> None:
        """End an OTel span."""
        if not self._enabled or span is None:
            return
            
        try:
            # Update attributes
            for k, v in metadata.items():
                span.set_attribute(k, v)
                
            # Set OTel status
            if status == "success":
                span.set_status(Status(StatusCode.OK))
            elif status == "error":
                span.set_status(Status(StatusCode.ERROR, description=metadata.get("error", "Unknown error")))
            
            span.end()
        except Exception:
            pass

    def record_event(
        self, name: str, metadata: Dict[str, Any], trace_metadata: TraceMetadata
    ) -> None:
        """Record an event on the current active span or as a standalone point."""
        if not self._enabled:
            return
            
        try:
            current_span = trace.get_current_span()
            attributes = {
                "trace_id": trace_metadata.trace_id,
                "conversation_id": trace_metadata.conversation_id or "",
                **metadata
            }
            
            if current_span and current_span.is_recording():
                current_span.add_event(name, attributes=attributes)
            else:
                # Standalone events are less common in OTel spans, 
                # but we can start a micro-span to record it if needed.
                with self._tracer.start_as_current_span(f"event.{name}", attributes=attributes) as span:
                    pass
        except Exception:
            pass

    def is_enabled(self) -> bool:
        """Return True if OTel is active."""
        return self._enabled
