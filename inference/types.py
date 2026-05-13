from dataclasses import dataclass
from typing import Optional, Dict, Any, Literal

ModelStatus = Literal["success", "recoverable_error", "fatal_error"]


@dataclass
class ModelRequest:
    task: str                  # e.g. "respond", "summarize", "extract"
    prompt: str
    system_prompt: Optional[str] = None
    context: Optional[str] = None
    constraints: Optional[Dict[str, Any]] = None
    # Phase 3: hard cap at 15s (was 30). Prevents hidden 60-120s hangs.
    timeout_s: Optional[int] = 15
    trace_id: Optional[str] = None


@dataclass
class ModelResponse:
    status: ModelStatus
    output: Optional[str] = None
    error_type: Optional[str] = None   # timeout | invalid_output | backend_unavailable
    metadata: Optional[Dict[str, Any]] = None
