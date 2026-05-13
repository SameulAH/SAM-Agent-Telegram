"""
Fact Extraction Layer

Extract structured facts from user input and model responses.
Store in memory with confidence scores and timestamps.

Enforces:
- Confidence threshold (>= 0.7)
- Max 3 facts per turn
- Deterministic schema
- Non-blocking failures

Phase PA (Personal Assistant): Added personal fact patterns for birthday,
preferences, workplace, identity, and location.  Added "persona_fact" type.
"""

import re
from datetime import datetime
from typing import List, Optional, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, validator


class ExtractedFact(BaseModel):
    """
    Structured fact extracted from conversation.
    
    Invariants:
    - confidence ∈ [0.0, 1.0]
    - content length <= 512 chars
    - timestamp is UTC
    """
    
    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    type: Literal["preference", "personal_fact", "persona_fact", "goal", "task"]
    content: str = Field(..., max_length=512)
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_turn_id: str = Field(..., description="Conversation turn ID")
    
    @validator("content")
    def content_not_empty(cls, v):
        """Content must be non-empty and not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()
    
    class Config:
        """Pydantic config."""
        frozen = True  # Immutable


class FactExtractionRequest(BaseModel):
    """Request for fact extraction."""
    
    user_input: Optional[str] = None
    model_response: Optional[str] = None
    conversation_turn_id: str
    conversation_id: str


class FactExtractionResponse(BaseModel):
    """Response from fact extraction."""
    
    extracted_facts: List[ExtractedFact] = Field(default_factory=list)
    extraction_attempted: bool = True
    extraction_error: Optional[str] = None


# ── Personal fact patterns ──────────────────────────────────────────────────
# Each entry: (compiled regex, fact_type, confidence)
# Patterns are applied in order; first match per sentence wins.
_PERSONAL_PATTERNS: List[tuple] = [
    # Birthday / date of birth
    (re.compile(
        r"(?:my\s+birthday\s+is|i\s+was\s+born\s+on|my\s+birth(?:day|date)\s+is)\s+(.+)",
        re.IGNORECASE,
    ), "persona_fact", 0.95),

    # Full name
    (re.compile(
        r"(?:my\s+name\s+is|i\s+am\s+called|call\s+me)\s+([A-Za-z][A-Za-z\s\-']{1,50})",
        re.IGNORECASE,
    ), "persona_fact", 0.95),

    # Workplace / employer
    (re.compile(
        r"(?:i\s+work\s+(?:at|for)|i\s+am\s+employed\s+(?:at|by)|my\s+(?:employer|company|job)\s+is)\s+(.+)",
        re.IGNORECASE,
    ), "persona_fact", 0.90),

    # Location / home
    (re.compile(
        r"(?:i\s+live\s+in|i\s+am\s+(?:based|located)\s+in|i\s+(?:reside|stay)\s+in)\s+(.+)",
        re.IGNORECASE,
    ), "persona_fact", 0.90),

    # Preferences (food, activity, color, …)
    (re.compile(
        r"i\s+(?:prefer|love|like|enjoy|hate|dislike|don't\s+like)\s+(.+)",
        re.IGNORECASE,
    ), "preference", 0.85),

    # Goals
    (re.compile(
        r"i\s+(?:want\s+to|would\s+like\s+to|am\s+trying\s+to|aim\s+to|plan\s+to)\s+(.+)",
        re.IGNORECASE,
    ), "goal", 0.80),

    # Tasks / reminders
    (re.compile(
        r"(?:remind\s+me\s+to|i\s+need\s+to|i\s+have\s+to|don't\s+forget\s+to)\s+(.+)",
        re.IGNORECASE,
    ), "task", 0.80),

    # Personal descriptors ("I am a doctor", "I am 30 years old")
    (re.compile(
        r"i\s+am\s+(?:a\s+|an\s+)?([A-Za-z][A-Za-z\s\-']{2,60})",
        re.IGNORECASE,
    ), "personal_fact", 0.75),

    # Agent Persona/Identity (User renaming the agent)
    (re.compile(
        r"(?:your\s+name\s+is|you\s+are\s+called|i\s+will\s+call\s+you)\s+([A-Za-z][A-Za-z\s\-']{1,50})",
        re.IGNORECASE,
    ), "persona_fact", 0.95),
]

# Sentences to skip (too short or meta-conversational)
_MIN_CONTENT_LEN = 5
_SKIP_PATTERNS = re.compile(
    r"^(ok|okay|sure|yes|no|hi|hello|thanks|thank\s+you|got\s+it|alright)\b",
    re.IGNORECASE,
)


class FactExtractor:
    """
    Extract facts from conversation turns.
    
    Guardrails:
    - Only extract: preferences, personal facts, persona_facts, goals, tasks
    - Confidence threshold: >= 0.7
    - Max 3 facts per turn
    - Reject vague or empty content
    - Non-blocking failures (exceptions don't break flow)
    """
    
    MIN_CONFIDENCE = 0.7
    MAX_FACTS_PER_TURN = 3
    
    def extract(
        self,
        request: FactExtractionRequest,
    ) -> FactExtractionResponse:
        """
        Extract facts from user input and model response.
        
        Returns facts that pass confidence threshold.
        Never raises; exceptions are caught and logged.
        
        Args:
            request: FactExtractionRequest
        
        Returns:
            FactExtractionResponse with extracted facts (or empty if none)
        """
        
        try:
            facts: List[ExtractedFact] = []
            
            # Extract from user input
            if request.user_input:
                user_facts = self._extract_from_text(
                    request.user_input,
                    request.conversation_turn_id,
                    source="user_input"
                )
                facts.extend(user_facts)
            
            # Extract from model response
            if request.model_response:
                response_facts = self._extract_from_text(
                    request.model_response,
                    request.conversation_turn_id,
                    source="model_response"
                )
                facts.extend(response_facts)
            
            # Apply confidence threshold and limit
            filtered_facts = [
                f for f in facts
                if f.confidence >= self.MIN_CONFIDENCE
            ][:self.MAX_FACTS_PER_TURN]
            
            return FactExtractionResponse(
                extracted_facts=filtered_facts,
                extraction_attempted=True,
                extraction_error=None,
            )
        
        except Exception as e:
            # Non-blocking failure
            return FactExtractionResponse(
                extracted_facts=[],
                extraction_attempted=False,
                extraction_error=str(e),
            )
    
    def _extract_from_text(
        self,
        text: str,
        turn_id: str,
        source: str,
    ) -> List[ExtractedFact]:
        """
        Extract facts from a text block using deterministic pattern matching.

        Patterns (in priority order):
        - Birthday / date of birth → persona_fact (confidence 0.95)
        - Full name → persona_fact (confidence 0.95)
        - Workplace / employer → persona_fact (confidence 0.90)
        - Location / home → persona_fact (confidence 0.90)
        - Preferences → preference (confidence 0.85)
        - Goals ("I want to…") → goal (confidence 0.80)
        - Tasks / reminders → task (confidence 0.80)
        - Personal descriptors ("I am a…") → personal_fact (confidence 0.75)

        Args:
            text: Text to extract from
            turn_id: Conversation turn ID
            source: "user_input" or "model_response"
        
        Returns:
            List of ExtractedFact (may be empty)
        """
        facts: List[ExtractedFact] = []
        seen_contents: set = set()

        # Split on sentence boundaries for finer-grained matching
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        # Also try the whole text as one unit
        candidates = sentences + [text.strip()]

        for sentence in candidates:
            sentence = sentence.strip()
            if len(sentence) < _MIN_CONTENT_LEN:
                continue
            if _SKIP_PATTERNS.match(sentence):
                continue

            for pattern, fact_type, confidence in _PERSONAL_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    # Extract the captured content (first group)
                    captured = match.group(1).strip().rstrip(".!?,;")
                    if len(captured) < _MIN_CONTENT_LEN:
                        continue

                    # Build human-readable content string
                    # Include the trigger phrase for context
                    full_sentence = sentence.strip().rstrip(".!?")
                    if len(full_sentence) > 512:
                        full_sentence = full_sentence[:509] + "..."

                    # Dedup by content
                    if full_sentence.lower() in seen_contents:
                        continue
                    seen_contents.add(full_sentence.lower())

                    try:
                        fact = ExtractedFact(
                            type=fact_type,
                            content=full_sentence,
                            confidence=confidence,
                            source_turn_id=turn_id,
                        )
                        facts.append(fact)
                    except Exception:
                        # Skip invalid facts silently
                        pass

                    # Only match the highest-priority pattern per sentence
                    break

            if len(facts) >= self.MAX_FACTS_PER_TURN:
                break

        return facts


def get_fact_extractor() -> "FactExtractor":
    """Get singleton fact extractor instance."""
    global _fact_extractor
    if _fact_extractor is None:
        _fact_extractor = FactExtractor()
    return _fact_extractor


_fact_extractor: Optional[FactExtractor] = None
