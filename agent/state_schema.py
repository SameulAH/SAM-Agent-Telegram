"""
Agent state schema and types.

The AgentState is the single source of truth for all agent execution context.
It is mutated only by specific nodes and never by task execution.

Phase 2 Addition: Memory-related fields store pointers and flags, never knowledge.
Phase MCP Addition: Tool execution fields track tool calls and results per turn.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from inference import ModelResponse


@dataclass
class AgentState:
    """
    Complete state schema for the SAM agent.
    
    Invariants:
    - conversation_id and trace_id are immutable once set
    - preprocessing_result, model_response, final_output are written only by specific nodes
    - command is the only control flow signal (decision_logic_node writes it)
    - error_type is set only by error_router_node
    - memory fields store pointers and metadata, never actual knowledge
    - long_term_memory_* fields are advisory only, never influence routing
    """

    # Identity
    conversation_id: str
    trace_id: str
    created_at: str

    # Input / Routing
    input_type: str  # text | audio | image
    raw_input: str

    # Processing
    preprocessing_result: Optional[str] = None

    # Model
    model_response: Optional[ModelResponse] = None
    model_metadata: Optional[Dict[str, Any]] = None

    # Output
    final_output: Optional[str] = None
    error_type: Optional[str] = None
    persona_name: str = "SAM"  # Dynamic agent identity

    # Control
    command: Optional[str] = None  # preprocess | call_model | success | failure
    
    # Memory: Phase 2 Addition (Short-term)
    # These fields store metadata and pointers, never actual memory content
    memory_available: bool = True                    # Is memory service available?
    memory_read_authorized: bool = False             # Does this execution want to read memory?
    memory_write_authorized: bool = False            # Does this execution want to write memory?
    memory_read_result: Optional[Dict[str, Any]] = None  # Data from memory read (if authorized)
    memory_read_status: Optional[str] = None         # Status of memory read (None | success | failed | not_found)
    memory_write_status: Optional[str] = None        # Status of memory write operation

    # Memory: Phase 3.2 Addition (Long-term)
    # Long-term memory is advisory-only, never influences control flow
    long_term_memory_requested: bool = False         # Should we access long-term memory?
    long_term_memory_status: str = "available"       # "available" | "unavailable"
    long_term_memory_read_result: Optional[Dict[str, Any]] = None  # Facts retrieved (if authorized)
    long_term_memory_read_status: Optional[str] = None  # Status of long-term read (None | success | failed | not_found)
    long_term_memory_write_status: Optional[str] = None  # Status of long-term fact write

    # ── Deterministic Memory Intent (Phase DMA) ──────────────────────────────
    # Set by memory_access_decision_node (rule-based, no LLM).
    # These two booleans gate the entire memory pipeline:
    #   requires_memory_write=False AND requires_memory_read=False → direct model path
    requires_memory_write: bool = False  # Declarative fact detected in input
    requires_memory_read: bool = False   # Retrieval reference detected in input
    # Facts produced by fact_extraction_node (Phase DMA)
    extracted_facts: Optional[List[Any]] = None
    # Authorization produced by write_authorization_node (Phase DMA)
    write_authorization_checked: bool = False

    # ── Tool Execution (Phase MCP) ────────────────────────────────────────────
    # These fields track tool calls per turn.
    # Invariants:
    # - tool_executed is True only AFTER tool_execution_node completes
    # - tool_call_count increments by 1 per tool execution (guardrail: max 1)
    # - tool_result stores the raw result dict from the tool
    # - tool_context stores the formatted, injection-safe string for model re-call
    # - Tool node NEVER writes to memory_* fields
    # - Tool node NEVER sets command (decision_logic_node retains authority)
    tool_executed: bool = False                      # Has a tool been executed this turn?
    tool_call_count: int = 0                         # Number of tool calls made (guardrail: max 1)
    tool_call: Optional[Dict[str, Any]] = None       # Pending tool call (set by model_call_node, cleared by tool_execution_node)
    tool_result: Optional[Dict[str, Any]] = None     # Raw result from tool execution
    tool_context: Optional[str] = None               # Formatted tool results for model context
    
    # Reflections: Learning consciousness insights
    reflections: Optional[List[Dict[str, Any]]] = None  # Insights learned about user this turn

    def __post_init__(self):
        """Validate state schema."""
        if not self.conversation_id:
            raise ValueError("conversation_id must not be empty")
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if self.input_type not in ("text", "audio", "image"):
            raise ValueError(f"input_type must be text, audio, or image; got {self.input_type}")
        if not self.raw_input:
            raise ValueError("raw_input must not be empty")
