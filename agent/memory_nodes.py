"""
Memory node implementations for the LangGraph agent.

These nodes are explicit decision points in the graph.
Memory reads and writes are routed only when authorized by decision_logic_node.

Phase 2: Short-term memory (MemoryController)
Phase 3.2: Long-term memory (LongTermMemoryStore)
Phase PA: Personal assistant fact extraction integrated into LTM write node.
          Personal facts (birthday, preferences, workplace) are extracted from
          raw_input and stored as "persona_fact" / "personal_fact" in Qdrant.
"""

import logging
from typing import Dict, Any, Optional

from agent.state_schema import AgentState
from agent.memory import (
    MemoryController,
    MemoryReadRequest,
    MemoryWriteRequest,
    StubMemoryController,
    LongTermMemoryStore,
    StubLongTermMemoryStore,
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryRetrievalQuery,
)
from agent.intelligence.fact_extraction import (
    FactExtractor,
    FactExtractionRequest,
)

# Get logger for memory operations
logger = logging.getLogger(__name__)


class MemoryNodeManager:
    """
    Manages memory read/write nodes in the graph.
    
    Responsibility:
    - Execute authorized SHORT-TERM memory reads (Phase 2)
    - Execute authorized SHORT-TERM memory writes (Phase 2)
    - Execute authorized LONG-TERM memory reads (Phase 3.2)
    - Execute authorized LONG-TERM memory writes (Phase 3.2)
    - Handle failures gracefully (non-fatal)
    - Never assume memory availability
    """

    def __init__(
        self,
        memory_controller: MemoryController,
        long_term_memory_store: Optional[LongTermMemoryStore] = None,
    ):
        """
        Initialize with memory controllers.
        
        Args:
            memory_controller: MemoryController instance (short-term)
            long_term_memory_store: LongTermMemoryStore instance (optional, Phase 3.2)
        """
        self.memory_controller = memory_controller
        self.long_term_memory_store = long_term_memory_store or StubLongTermMemoryStore()
        self._fact_extractor = FactExtractor()

    def memory_read_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized memory read.
        
        Responsibility:
        - Check authorization
        - Call MemoryController.read()
        - Handle failure gracefully
        
        Rules:
        - Only runs if memory_read_authorized is True
        - Never crashes agent on failure
        - Sets memory_available=False if unavailable
        
        Returns:
            Dict with memory_read_result and memory_available
        """
        # Safety check: should only be called if authorized
        if not state.memory_read_authorized:
            return {
                "memory_read_result": None,
                "memory_available": state.memory_available,
            }

        # Build request
        request = MemoryReadRequest(
            conversation_id=state.conversation_id,
            key="conversation_context",  # Default key for context
            authorized=True,
            reason="agent_requested",
        )

        # Execute read (never crashes)
        try:
            response = self.memory_controller.read(request)

            # Handle response
            if response.status == "success":
                return {
                    "memory_read_result": response.data,
                    "memory_read_status": "success",
                    "memory_available": True,
                }
            elif response.status == "unavailable":
                return {
                    "memory_read_result": None,
                    "memory_read_status": "failed",
                    "memory_available": False,
                }
            else:  # not_found, unauthorized, etc.
                return {
                    "memory_read_result": None,
                    "memory_read_status": "not_found",
                    "memory_available": True,
                }
        except Exception as e:
            # Memory failure is non-fatal
            return {
                "memory_read_result": None,
                "memory_read_status": "failed",
                "memory_available": False,
            }

    def memory_write_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized memory write.
        
        Responsibility:
        - Check authorization
        - Call MemoryController.write()
        - Handle failure gracefully
        
        Rules:
        - Only runs if memory_write_authorized is True
        - Never crashes agent on failure
        - Sets memory_available=False if unavailable
        
        Returns:
            Dict with memory_write_status and memory_available
        """
        # Log entry to memory_write_node
        logger.info(
            f"memory_write_node: START (conversation_id={state.conversation_id}, "
            f"trace_id={state.trace_id}, authorized={state.memory_write_authorized})"
        )
        
        # Safety check: should only be called if authorized
        if not state.memory_write_authorized:
            logger.debug(
                f"memory_write_node: Not authorized for conversation {state.conversation_id}"
            )
            return {
                "memory_write_status": None,
                "memory_available": state.memory_available,
            }

        # Derive final_output: state.final_output is set by format_response_node which
        # runs AFTER memory_write in the new graph flow. Fall back to model_response.output
        # so we always store a non-null answer.
        final_out = state.final_output
        if final_out is None and state.model_response:
            if state.model_response.status == "success" and state.model_response.output:
                final_out = state.model_response.output

        # Phase Rolling Window: Maintain last 5 turns for context continuity
        existing_history = []
        try:
            read_req = MemoryReadRequest(conversation_id=state.conversation_id, key="conversation_context", authorized=True)
            read_resp = self.memory_controller.read(read_req)
            if read_resp.status == "success" and isinstance(read_resp.data, dict):
                # Handle legacy single-turn format or new history format
                if "history" in read_resp.data:
                    existing_history = read_resp.data["history"]
                else:
                    # Upgrade legacy single-turn to history
                    existing_history = [read_resp.data]
        except Exception:
            pass

        # Build current turn
        current_turn = {
            "raw_input": state.raw_input,
            "final_output": final_out,
            "interaction_timestamp": state.created_at,
        }
        
        # Update history (Slide window: keep last 5)
        new_history = existing_history + [current_turn]
        new_history = new_history[-5:]

        # Build request
        request = MemoryWriteRequest(
            conversation_id=state.conversation_id,
            key="conversation_context",
            data={
                "history": new_history,
                "last_turn": current_turn, # Legacy compatibility
            },
            authorized=True,
            reason="agent_storing_rolling_history",
        )

        # Execute write (never crashes)
        try:
            logger.debug(
                f"memory_write_node: Calling write() for {state.conversation_id}"
            )
            response = self.memory_controller.write(request)

            if response.status == "success":
                logger.info(
                    f"memory_write_node: SUCCESS for {state.conversation_id}"
                )
                return {
                    "memory_write_status": "success",
                    "memory_available": True,
                }
            elif response.status == "failed":
                logger.warning(
                    f"memory_write_node: FAILED for {state.conversation_id}: {response.error}"
                )
                return {
                    "memory_write_status": "failed",
                    "memory_available": True,
                }
            else:  # unauthorized, etc.
                logger.warning(
                    f"memory_write_node: {response.status.upper()} for {state.conversation_id}"
                )
                return {
                    "memory_write_status": response.status,
                    "memory_available": True,
                }
        except Exception as e:
            # Memory failure is non-fatal
            logger.error(
                f"memory_write_node: EXCEPTION for {state.conversation_id}: {str(e)}"
            )
            return {
                "memory_write_status": "failed",
                "memory_available": False,
            }
    # ─────────────────────────────────────────────────────
    # LONG-TERM MEMORY NODES (Phase 3.2)
    # ─────────────────────────────────────────────────────

    def long_term_memory_read_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized long-term memory retrieval.
        
        Responsibility:
        - Retrieve facts from LongTermMemoryStore
        - Handle failures gracefully (advisory-only)
        
        Rules:
        - Routing authority rests with decision_logic_node; if we reach this
          node, the read is implicitly authorized.
        - Never crashes agent on failure
        - Sets long_term_memory_status="unavailable" if store unavailable
        - Returns {"facts": []} (empty list sentinel) when no facts found,
          so decision_logic_node can distinguish "not yet read" (None)
          from "read but empty" ({"facts": []}).
        - Always returns Dict (never raises exception)
        
        Returns:
            Dict with long_term_memory_read_result and long_term_memory_status
        """
        # Build retrieval query
        query = LongTermMemoryRetrievalQuery(
            user_id=state.conversation_id,  # Use conversation_id as user_id
            fact_types=None,  # Retrieve all types
            limit=10,  # Default limit
            authorized=True,
        )

        # Execute retrieval (never crashes)
        try:
            response = self.long_term_memory_store.retrieve_facts(query)

            # Handle response
            if response.status == "success":
                facts = response.facts or []
                return {
                    "long_term_memory_read_result": {
                        "facts": [
                            {
                                "fact_type": f.fact_type,
                                "content": f.content if isinstance(f.content, str) else str(f.content),
                                "confidence": f.confidence,
                                "created_at": f.created_at.isoformat() if hasattr(f.created_at, "isoformat") else str(f.created_at) if f.created_at else None,
                            }
                            for f in facts
                        ]
                    },
                    "long_term_memory_read_status": "success",
                    "long_term_memory_status": "available",
                }
            elif response.status == "unavailable":
                return {
                    "long_term_memory_read_result": {"facts": []},  # sentinel: attempted
                    "long_term_memory_read_status": "failed",
                    "long_term_memory_status": "unavailable",
                }
            else:  # not_found, unauthorized, etc.
                return {
                    "long_term_memory_read_result": {"facts": []},  # sentinel: read attempted, no facts
                    "long_term_memory_read_status": "not_found",
                    "long_term_memory_status": "available",
                }
        except Exception as e:
            # Long-term memory failure is non-fatal
            return {
                "long_term_memory_read_result": {"facts": []},  # sentinel: attempted but failed
                "long_term_memory_read_status": "failed",
                "long_term_memory_status": "unavailable",
            }

    def long_term_memory_write_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Execute authorized long-term memory fact writing.
        
        Responsibility:
        - Check authorization
        - Extract personal facts from raw_input via FactExtractor
        - Write extracted persona/personal facts to LongTermMemoryStore
        - Write interaction outcome as fallback fact
        - Handle failures gracefully (advisory-only)
        
        Rules:
        - Only runs if memory_write_authorized is True
        - Never crashes agent on failure
        - Max 3 facts per turn (enforced by FactExtractor)
        - Personal facts have priority over interaction outcome
        - Sets long_term_memory_status="unavailable" if store unavailable
        - Always returns Dict (never raises exception)
        
        Returns:
            Dict with long_term_memory_write_status and long_term_memory_status
        """
        # Safety check: should only be called if authorized
        if not state.memory_write_authorized:
            return {
                "long_term_memory_write_status": None,
                "long_term_memory_status": state.long_term_memory_status,
            }

        written_count = 0
        write_status = "success"

        try:
            # ── Phase PA: Extract personal facts from user input ──────────────
            user_input = state.raw_input or state.preprocessing_result or ""
            if user_input.strip():
                extraction_request = FactExtractionRequest(
                    user_input=user_input,
                    model_response=None,  # Only extract from user statements
                    conversation_turn_id=state.trace_id,
                    conversation_id=state.conversation_id,
                )
                extraction_response = self._fact_extractor.extract(extraction_request)

                for extracted_fact in extraction_response.extracted_facts:
                    fact = MemoryFact(
                        fact_type=extracted_fact.type,  # "persona_fact", "preference", etc.
                        content={"text": extracted_fact.content, "source": "user_statement"},
                        user_id=state.conversation_id,
                        confidence=extracted_fact.confidence,
                        source="user_input_extraction",
                    )
                    write_request = LongTermMemoryWriteRequest(
                        user_id=state.conversation_id,
                        fact=fact,
                        authorized=True,
                        reason="agent_storing_personal_fact",
                    )
                    try:
                        response = self.long_term_memory_store.write_fact(write_request)
                        if response.status == "success":
                            written_count += 1
                        else:
                            logger.debug(
                                f"long_term_memory_write_node: persona fact write status="
                                f"{response.status} for {state.conversation_id}"
                            )
                    except Exception as inner_e:
                        logger.debug(
                            f"long_term_memory_write_node: persona fact write exception: {inner_e}"
                        )

            # ── Interaction outcome (always write if there's output) ───────────
            if state.final_output:
                outcome_fact = MemoryFact(
                    fact_type="interaction_outcome",
                    content={"text": state.final_output, "source": "agent_response"},
                    user_id=state.conversation_id,
                    confidence=0.8,
                    source="agent_interaction",
                )
                outcome_request = LongTermMemoryWriteRequest(
                    user_id=state.conversation_id,
                    fact=outcome_fact,
                    authorized=True,
                    reason="agent_storing_interaction_outcome",
                )
                response = self.long_term_memory_store.write_fact(outcome_request)
                if response.status == "success":
                    written_count += 1
                elif response.status == "failed":
                    write_status = "failed"

            final_status = "success" if written_count > 0 else write_status
            logger.info(
                f"long_term_memory_write_node: wrote {written_count} fact(s) for "
                f"{state.conversation_id}"
            )
            return {
                "long_term_memory_write_status": final_status,
                "long_term_memory_status": "available",
            }

        except Exception as e:
            logger.error(
                f"long_term_memory_write_node: EXCEPTION for {state.conversation_id}: {e}"
            )
            return {
                "long_term_memory_write_status": "failed",
                "long_term_memory_status": "unavailable",
            }

    def _write_long_term_fact(self, user_id: str, text: str, fact_type: str, confidence: float = 0.8) -> bool:
        """Helper to write a single fact to LTM (used by reflection node)."""
        try:
            from agent.memory import MemoryFact, LongTermMemoryWriteRequest
            fact = MemoryFact(
                fact_type=fact_type,
                content={"text": text, "source": "agent_reflection"},
                user_id=user_id,
                confidence=confidence,
                source="reflection_node",
            )
            request = LongTermMemoryWriteRequest(
                user_id=user_id,
                fact=fact,
                authorized=True,
                reason="agent_conscious_reflection",
            )
            response = self.long_term_memory_store.write_fact(request)
            return response.status == "success"
        except Exception as e:
            logger.error(f"MemoryNodeManager: Failed to write reflection fact: {e}")
            return False