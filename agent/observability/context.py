"""
Agent Execution Context.

Centralizes telemetry emission and request-scoped metadata for SAM.
Standardizes node health metrics and reasoning spans.
"""

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from uuid import uuid4

@dataclass
class NodeMetric:
    """Metric for a single node execution."""
    node_name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"  # success | error | skipped
    error_type: Optional[str] = None
    duration_ms: Optional[float] = None

@dataclass
class AgentExecutionContext:
    """
    Request-scoped execution context for SAM.
    
    Purpose:
    - Standardize telemetry emission
    - Attach metadata to all spans
    - Provide reasoning-specific spans
    - Track state integrity violations
    """
    trace_id: str
    conversation_id: str
    telemetry_emitter: Any  # Usually an OtelTracer instance
    node_metrics: Dict[str, NodeMetric] = field(default_factory=dict)
    reasoning_events: List[Dict[str, Any]] = field(default_factory=list)

    def start_node(self, node_name: str) -> None:
        """Record start of a node execution."""
        self.node_metrics[node_name] = NodeMetric(
            node_name=node_name,
            start_time=time.time()
        )

    def end_node(self, node_name: str, status: str = "success", error_type: Optional[str] = None) -> None:
        """Record end of a node execution."""
        if node_name in self.node_metrics:
            metric = self.node_metrics[node_name]
            metric.end_time = time.time()
            metric.status = status
            metric.error_type = error_type
            metric.duration_ms = (metric.end_time - metric.start_time) * 1000

    def record_reasoning(self, stage: str, data: Dict[str, Any]) -> None:
        """
        Record a reasoning event (thought, observation, etc).
        
        Args:
            stage: "before_tool" | "after_tool" | "synthesis"
            data: Reasoning data (thought, observation, tool_name, etc)
        """
        event = {
            "stage": stage,
            "timestamp": time.time(),
            **data
        }
        self.reasoning_events.append(event)
        
        # Also emit as a span/event via telemetry
        if hasattr(self.telemetry_emitter, "record_event"):
            from agent.tracing.tracer import TraceMetadata
            self.telemetry_emitter.record_event(
                name=f"reasoning.{stage}",
                metadata=data,
                trace_metadata=TraceMetadata(
                    trace_id=self.trace_id,
                    conversation_id=self.conversation_id
                )
            )

    def __deepcopy__(self, memo: dict) -> "AgentExecutionContext":
        # LangGraph deepcopies the AgentState before each node for retry safety.
        # The telemetry_emitter (tracer) holds threading.Lock objects that cannot
        # be deepcopied. We share the same tracer instance (it is stateless for
        # copy purposes) and deepcopy only the mutable tracking data.
        new = AgentExecutionContext(
            trace_id=self.trace_id,
            conversation_id=self.conversation_id,
            telemetry_emitter=self.telemetry_emitter,  # shared reference, not deepcopied
            node_metrics=copy.deepcopy(self.node_metrics, memo),
            reasoning_events=copy.deepcopy(self.reasoning_events, memo),
        )
        memo[id(self)] = new
        return new

    def record_state_violation(self, node_name: str, field: str, description: str) -> None:
        """Record a state integrity violation."""
        if hasattr(self.telemetry_emitter, "record_event"):
            from agent.tracing.tracer import TraceMetadata
            self.telemetry_emitter.record_event(
                name="state_integrity_violation",
                metadata={
                    "node_name": node_name,
                    "field": field,
                    "description": description
                },
                trace_metadata=TraceMetadata(
                    trace_id=self.trace_id,
                    conversation_id=self.conversation_id
                )
            )
