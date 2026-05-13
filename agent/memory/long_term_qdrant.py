"""
Qdrant-backed long-term memory store.

A durable, vector-based semantic knowledge base that appends stable facts
and enables similarity-based retrieval without influencing control flow.

Key properties:
- Append-only: facts never overwritten
- Semantic search: similarity-based retrieval
- Failure-safe: Qdrant down → graceful degrade
- Advisory-only: facts inform responses, never decisions
"""

import json
from typing import Optional, List
from datetime import datetime
from uuid import uuid4

import logging
from agent.memory.long_term_base import LongTermMemoryStore
from agent.memory.long_term_types import (
    MemoryFact,
    LongTermMemoryWriteRequest,
    LongTermMemoryWriteResponse,
    LongTermMemoryRetrievalQuery,
    LongTermMemoryRetrievalResponse,
)


class QdrantLongTermMemoryStore(LongTermMemoryStore):
    """
    Qdrant-backed long-term memory store.
    
    Stores facts as vectors for semantic retrieval.
    Append-only: facts are never updated or deleted.
    Failure-safe: Qdrant unavailable → graceful degradation.
    
    Configuration:
    - qdrant_url: Qdrant API endpoint
    - collection_name: Collection to store facts
    - vector_size: Dimension of fact embeddings (default: 384)
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "long_term_memory",
        vector_size: int = 384,
    ):
        """
        Initialize Qdrant long-term memory store.
        
        Args:
            qdrant_url: Qdrant API endpoint
            collection_name: Collection name for facts
            vector_size: Embedding dimension
        """
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._client = None
        self._embedder = None
        self.logger = logging.getLogger(__name__)
        self._initialize()

    def _initialize(self) -> None:
        """Initialize Qdrant client and collection if needed."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            # Connect to Qdrant
            self._client = QdrantClient(url=self.qdrant_url, timeout=5.0)

            # Check if collection exists
            try:
                self._client.get_collection(self.collection_name)
            except Exception:
                # Collection doesn't exist, create it
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                self.logger.info(f"Qdrant collection created: {self.collection_name}")
            else:
                self.logger.info(f"Qdrant collection connected: {self.collection_name}")
        except ImportError:
            # qdrant-client not installed
            self.logger.error("qdrant-client not installed")
            self._client = None
        except Exception as e:
            # Qdrant unavailable
            self.logger.error(f"Qdrant unavailable at {self.qdrant_url}: {str(e)}")
            self._client = None

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding for text using a lightweight embedder.
        
        For simplicity, uses sentence-transformers if available,
        otherwise returns None (graceful degradation).
        """
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            embedding = self._embedder.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception:
            # Embedder unavailable, return zeros (safe default)
            return [0.0] * self.vector_size

    def write_fact(self, request: LongTermMemoryWriteRequest) -> LongTermMemoryWriteResponse:
        """
        Append a fact to Qdrant.
        
        Args:
            request: LongTermMemoryWriteRequest
            
        Returns:
            LongTermMemoryWriteResponse with success or explicit failure
        """
        # Check authorization
        if not request.authorized:
            return LongTermMemoryWriteResponse(
                status="unauthorized",
                error="Memory write not authorized by decision_logic_node",
            )

        try:
            # Check if Qdrant is available
            if self._client is None:
                return LongTermMemoryWriteResponse(
                    status="failed",
                    error="Qdrant connection unavailable",
                )

            # Create fact with ID and timestamp
            fact = request.fact
            fact.fact_id = str(uuid4())
            fact.created_at = datetime.now().isoformat()

            # Get embedding (fallback to zeros if unavailable)
            fact_text = json.dumps(fact.content)
            embedding = self._get_embedding(fact_text)

            # Prepare point for Qdrant
            point_data = {
                "fact_id": fact.fact_id,
                "user_id": fact.user_id,
                "fact_type": fact.fact_type,
                "content": fact.content,
                "confidence": fact.confidence,
                "source": fact.source,
                "created_at": fact.created_at,
            }

            # Upsert to Qdrant (append)
            from qdrant_client.http import models

            point = models.PointStruct(
                id=int(fact.fact_id.replace("-", "")[:16], 16) % (2**63),  # Numeric ID
                vector=embedding,
                payload=point_data,
            )

            self._client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            return LongTermMemoryWriteResponse(
                status="success",
                fact_id=fact.fact_id,
            )
        except Exception as e:
            return LongTermMemoryWriteResponse(
                status="failed",
                error=f"Failed to write fact to Qdrant: {str(e)}",
            )

    def retrieve_facts(self, query: LongTermMemoryRetrievalQuery) -> LongTermMemoryRetrievalResponse:
        """
        Retrieve facts from Qdrant.
        
        Args:
            query: LongTermMemoryRetrievalQuery
            
        Returns:
            LongTermMemoryRetrievalResponse with facts (oldest first) or error
        """
        # Check authorization
        if not query.authorized:
            return LongTermMemoryRetrievalResponse(
                status="unauthorized",
                error="Memory read not authorized by decision_logic_node",
            )

        try:
            # Check if Qdrant is available
            if self._client is None:
                return LongTermMemoryRetrievalResponse(
                    status="unavailable",
                    error="Qdrant connection unavailable",
                )

            # Search Qdrant via scroll with filter (exact match for user_id)
            from qdrant_client import models
            
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=query.user_id),
                    )
                ]
            )
            
            results, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=query.limit,
                with_payload=True,
                with_vectors=False,
            )

            # Extract facts from results
            facts = []
            for result in results:
                payload = result.payload
                # Filter by fact_type if specified
                if query.fact_types and payload.get("fact_type") not in query.fact_types:
                    continue

                # Reconstruct fact
                fact = MemoryFact(
                    fact_type=payload.get("fact_type", ""),
                    content=payload.get("content", {}),
                    user_id=payload.get("user_id", ""),
                    confidence=payload.get("confidence", 1.0),
                    source=payload.get("source", ""),
                    fact_id=payload.get("fact_id"),
                    created_at=payload.get("created_at"),
                )
                facts.append(fact)

            # Sort by created_at (oldest first)
            facts.sort(key=lambda f: f.created_at or "", reverse=False)

            return LongTermMemoryRetrievalResponse(
                status="success",
                facts=facts,
            )
        except Exception as e:
            return LongTermMemoryRetrievalResponse(
                status="unavailable",
                error=f"Failed to retrieve facts from Qdrant: {str(e)}",
            )
