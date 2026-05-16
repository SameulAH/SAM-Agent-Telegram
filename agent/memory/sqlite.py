"""
SQLite-backed short-term memory controller.

A durable, session-scoped scratchpad that supports conversation continuity
without ever influencing control flow, decisions, or authority.

Key properties:
- Implements exactly the same interface as StubMemoryController
- Can be swapped without changing any agent code
- Returns typed responses (never raises uncaught exceptions)
- Session-scoped: stores recent context per conversation
- Non-fatal: failures degrade gracefully, never block execution

Data stored:
- conversation_id (session identifier)
- turn information (context for conversation)
- message metadata (timestamps, participants)

Data NEVER stored:
- long-term facts
- preferences
- embeddings
- model outputs
- decision outputs
- control signals
"""

import sqlite3
import json
import logging
import os
from pathlib import Path
from typing import Optional
from agent.memory.base import MemoryController
from agent.memory.types import MemoryReadRequest, MemoryReadResponse, MemoryWriteRequest, MemoryWriteResponse

# Get logger for memory operations
logger = logging.getLogger(__name__)

# Evict STM entries not touched within this many seconds.
# Configurable via STM_TTL_SECONDS env var; defaults to 7 days.
_STM_TTL_SECONDS: int = int(os.getenv("STM_TTL_SECONDS", str(7 * 24 * 3600)))


class SQLiteShortTermMemoryStore(MemoryController):
    """
    SQLite-backed memory for short-term session context.
    
    Pure plumbing: SQLite is an implementation detail.
    Agent logic is completely decoupled from storage mechanics.
    
    Design:
    - One table: short_term_memory
    - Columns: conversation_id, key, data (JSON), created_at, updated_at
    - Index: (conversation_id, key) for fast lookups
    - No schema assumptions in agent code
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite short-term memory store.
        
        Args:
            db_path: Path to SQLite database file.
                    If None, uses ':memory:' (in-memory, useful for testing).
        """
        self.db_path = db_path or ":memory:"
        self._initialize_db()

    def _initialize_db(self) -> None:
        """
        Initialize SQLite database and schema.
        
        Called once at startup. If database already exists, this is a no-op.
        Enables WAL mode for durability and concurrency.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable WAL mode for better concurrency and crash recovery
            if self.db_path != ":memory:":
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=FULL")

            # Create short_term_memory table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS short_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(conversation_id, key)
                )
            """)

            # Create index for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_key 
                ON short_term_memory(conversation_id, key)
            """)

            conn.commit()
            conn.close()
            
            logger.debug(f"SQLite memory initialized: {self.db_path}")
        except Exception as e:
            # Log initialization error but don't raise
            # Database will be marked unavailable on first operation
            logger.error(f"Failed to initialize SQLite memory: {str(e)}")
            pass

    def read(self, request: MemoryReadRequest) -> MemoryReadResponse:
        """
        Read derived facts from short-term memory.
        
        Args:
            request: MemoryReadRequest with conversation_id, key, authorized flag
            
        Returns:
            MemoryReadResponse with status and data (if successful) or error
            
        Never raises exceptions. All failures are returned as response status.
        """
        # Check authorization
        if not request.authorized:
            logger.debug(f"Memory read unauthorized: {request.conversation_id}")
            return MemoryReadResponse(
                status="unauthorized",
                error="Memory read not authorized by decision_logic_node",
            )

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT data, created_at FROM short_term_memory
                WHERE conversation_id = ? AND key = ?
                """,
                (request.conversation_id, request.key),
            )

            row = cursor.fetchone()
            conn.close()

            if row is None:
                logger.debug(f"Memory key not found: {request.conversation_id}, key={request.key}")
                return MemoryReadResponse(
                    status="not_found",
                    error=f"Key '{request.key}' not found in conversation memory",
                )

            # Parse JSON data
            try:
                data = json.loads(row[0])
                # Inject database timestamp into data for temporal awareness (Phase Humanizing)
                if isinstance(data, dict):
                    data["created_at"] = row[1]
                
                logger.debug(f"Memory read successful: {request.conversation_id}, key={request.key}")
                return MemoryReadResponse(status="success", data=data)
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted memory data: {request.conversation_id}, {str(e)}")
                return MemoryReadResponse(
                    status="failed",
                    error=f"Corrupted memory data: {str(e)}",
                )

        except sqlite3.OperationalError as e:
            # Database locked, file missing, corrupted, etc.
            logger.error(f"SQLite operational error during read: {str(e)}")
            return MemoryReadResponse(
                status="unavailable",
                error=f"Memory unavailable: {str(e)}",
            )
        except Exception as e:
            # Unexpected error
            return MemoryReadResponse(
                status="unavailable",
                error=f"Memory read failed: {str(e)}",
            )

    def write(self, request: MemoryWriteRequest) -> MemoryWriteResponse:
        """
        Write derived facts to short-term memory.
        
        Args:
            request: MemoryWriteRequest with conversation_id, key, data, authorized flag
            
        Returns:
            MemoryWriteResponse with success status or explicit failure
            
        Never raises exceptions. All failures are returned as response status.
        """
        # Check authorization
        if not request.authorized:
            logger.debug(f"Memory write unauthorized: {request.conversation_id}")
            return MemoryWriteResponse(
                status="unauthorized",
                error="Memory write not authorized by decision_logic_node",
            )

        try:
            # Validate data is JSON-serializable
            data_json = json.dumps(request.data)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Insert or replace (upsert)
            cursor.execute(
                """
                INSERT INTO short_term_memory (conversation_id, key, data)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id, key)
                DO UPDATE SET data = excluded.data, updated_at = CURRENT_TIMESTAMP
                """,
                (request.conversation_id, request.key, data_json),
            )

            # Evict entries older than STM_TTL_SECONDS (non-fatal, inside same txn)
            self._evict_old_entries(conn)

            # Explicit commit for durability
            conn.commit()
            conn.close()
            
            logger.info(f"Memory write successful: conversation_id={request.conversation_id}, key={request.key}")
            return MemoryWriteResponse(status="success")

        except sqlite3.OperationalError as e:
            # Database locked, file missing, corrupted, etc.
            logger.error(f"SQLite operational error during write: {request.conversation_id}, {str(e)}")
            return MemoryWriteResponse(
                status="failed",
                error=f"Memory unavailable: {str(e)}",
            )
        except (json.JSONDecodeError, TypeError) as e:
            # Data not JSON-serializable
            logger.error(f"Data not JSON-serializable: {request.conversation_id}, {str(e)}")
            return MemoryWriteResponse(
                status="failed",
                error=f"Data not JSON-serializable: {str(e)}",
            )
        except Exception as e:
            # Unexpected error
            logger.error(f"Memory write failed unexpectedly: {request.conversation_id}, {str(e)}")
            return MemoryWriteResponse(
                status="failed",
                error=f"Memory write failed: {str(e)}",
            )

    def _evict_old_entries(self, conn: sqlite3.Connection) -> None:
        """Delete STM rows not updated within _STM_TTL_SECONDS.

        Called inside an existing write transaction so no extra commit is needed.
        Non-fatal: silently skipped on any error.
        """
        try:
            conn.execute(
                """
                DELETE FROM short_term_memory
                WHERE (strftime('%s', 'now') - strftime('%s', updated_at)) > ?
                """,
                (_STM_TTL_SECONDS,),
            )
        except Exception as e:
            logger.debug("STM eviction skipped: %s", e)

    def clear_conversation(self, conversation_id: str) -> bool:
        """
        Clear all memory for a conversation (session end).
        
        Useful for cleaning up old sessions.
        
        Args:
            conversation_id: Conversation to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM short_term_memory WHERE conversation_id = ?",
                (conversation_id,),
            )
            conn.commit()
            conn.close()
            return True
        except Exception:
            return False
