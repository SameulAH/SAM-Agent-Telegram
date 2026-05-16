"""
Telegram Webhook Handler

Receives Telegram updates via webhook and processes them through the agent.

Security:
  - Verify Telegram signature (optional, add later if needed)
  - Rate limiting (can be added via middleware)
  - Error recovery (non-blocking failures)

Update Flow:
  webhook → parse_update → normalize → agent → send_response
"""

import logging
import os
import time
from typing import Dict, Optional, Any
from datetime import datetime

from cachetools import TTLCache
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks

from config import Config
from transport.telegram.transport import (
    create_telegram_transport,
    NormalizedMessage,
)

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/webhook", tags=["webhook"])

# TTL cache for update_id deduplication — 5-minute window, max 5 000 entries.
# Prevents duplicate processing when Telegram retries a slow webhook.
_processed_updates: TTLCache = TTLCache(maxsize=5000, ttl=300)

# ── Per-user rate limiting ─────────────────────────────────────────────────────
# Allow at most RATE_LIMIT_MAX_CALLS requests per user within RATE_LIMIT_WINDOW_S seconds.
# Configurable via env vars; defaults are generous for a personal assistant.
_RATE_LIMIT_WINDOW_S: int = int(os.getenv("RATE_LIMIT_WINDOW_S", "5"))
_RATE_LIMIT_MAX_CALLS: int = int(os.getenv("RATE_LIMIT_MAX_CALLS", "3"))

# Maps user_id → list of timestamps of recent requests (within the window)
_user_request_times: TTLCache = TTLCache(maxsize=2000, ttl=_RATE_LIMIT_WINDOW_S)


def _is_rate_limited(user_id: int) -> bool:
    """Return True if this user has exceeded the per-window call limit."""
    now = time.monotonic()
    timestamps: list = _user_request_times.get(user_id, [])
    # Prune timestamps outside the current window
    timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW_S]
    if len(timestamps) >= _RATE_LIMIT_MAX_CALLS:
        return True
    timestamps.append(now)
    _user_request_times[user_id] = timestamps
    return False

# Telegram update structure
class TelegramUser(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    username: Optional[str] = None


class TelegramChat(BaseModel):
    id: int
    type: str  # private, group, supergroup, channel
    title: Optional[str] = None
    username: Optional[str] = None


class TelegramVoice(BaseModel):
    """Voice object from Telegram."""
    file_id: str
    file_unique_id: str
    duration: int  # seconds
    mime_type: Optional[str] = None
    file_size: Optional[int] = None


class TelegramMessage(BaseModel):
    message_id: int
    date: int
    chat: TelegramChat
    from_: TelegramUser = Field(..., alias="from")
    text: Optional[str] = None
    voice: Optional[TelegramVoice] = None  # Support for voice messages
    
    class Config:
        populate_by_name = True


class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[TelegramMessage] = None


# Storage for transport (initialized once)
_transport = None


def get_telegram_transport():
    """Get or create Telegram transport (singleton)."""
    global _transport
    if _transport is None:
        _transport = create_telegram_transport()
    return _transport


@router.post("/telegram")
async def telegram_webhook(update: TelegramUpdate, background_tasks: BackgroundTasks):
    """
    Receive Telegram webhook updates.

    Idempotency guarantee: updates with a previously-seen update_id are
    silently acknowledged (HTTP 200) without any processing so that Telegram
    retries never produce duplicate messages.

    Heavy work (STT, agent.invoke, TTS) is offloaded to a BackgroundTask so
    the handler returns HTTP 200 in milliseconds — well inside Telegram's
    10-second timeout.
    """
    # ── 1. Deduplication ──────────────────────────────────────────────────────
    if update.update_id in _processed_updates:
        logger.debug(f"Ignoring duplicate update_id={update.update_id}")
        return {"status": "ok"}
    _processed_updates[update.update_id] = True
    logger.info(f"Processing update_id={update.update_id}")

    try:
        # ── 2. Validate payload ───────────────────────────────────────────────
        if not update.message:
            logger.warning(f"Update {update.update_id} has no message")
            return {"status": "ok"}

        # ── 3. Per-user rate limiting ─────────────────────────────────────────
        user_id = update.message.from_.id
        if _is_rate_limited(user_id):
            logger.warning(
                f"Rate limit exceeded for user_id={user_id}, update_id={update.update_id}"
            )
            # Always return 200 to Telegram — dropping silently is intentional
            # (sending a reply here would itself spam the user under flood conditions)
            return {"status": "ok"}

        msg = update.message

        # ── 4. Route voice messages to the voice handler ──────────────────────
        if msg.voice:
            from webhook.telegram_voice import handle_voice_message
            logger.info("Voice message detected, forwarding to voice handler")
            return await handle_voice_message(update, background_tasks)

        # ── 5. Skip non-text messages ─────────────────────────────────────────
        if not msg.text:
            logger.debug(f"Message {msg.message_id} has no text or voice, skipping")
            return {"status": "ok"}

        logger.info(
            f"Received message from {msg.from_.username or msg.from_.first_name}: "
            f"{msg.text[:50]}"
        )

        # ── 6. Offload to background — return 200 immediately ─────────────────
        background_tasks.add_task(
            _process_text_async,
            chat_id=msg.chat.id,
            user_id=msg.from_.id,
            text=msg.text,
            user_name=msg.from_.username or msg.from_.first_name,
            timestamp=datetime.fromtimestamp(msg.date),
        )
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Error handling Telegram update: {str(e)}", exc_info=True)
        return {"status": "ok"}  # Always 200 so Telegram stops retrying


async def _process_text_async(
    chat_id: int,
    user_id: int,
    text: str,
    user_name: str,
    timestamp: datetime,
) -> None:
    """
    Background task: run agent pipeline and send the reply for a text message.

    Runs *after* the webhook handler has already returned HTTP 200, so any
    latency here (LLM, tool calls, TTS) never risks a Telegram retry.
    """
    from webhook.telegram_voice import clean_agent_response, text_to_ogg_voice

    transport = get_telegram_transport()

    try:
        normalized = NormalizedMessage(
            platform="telegram",
            user_id=str(user_id),
            chat_id=str(chat_id),
            content=text,
            timestamp=timestamp,
            user_name=user_name,
        )

        response_text = await process_message_through_agent(normalized)

        if response_text:
            response_text = clean_agent_response(response_text)

        if not response_text:
            response_text = "Sorry, I couldn't process that."

        # Long replies → send as voice message; short replies → send as text
        lines = [l for l in response_text.splitlines() if l.strip()]
        if len(lines) > 5:
            import asyncio
            loop = asyncio.get_event_loop()
            ogg_bytes = await loop.run_in_executor(None, text_to_ogg_voice, response_text)
            from webhook.telegram_voice import get_voice_transport
            voice_transport = get_voice_transport()
            if ogg_bytes:
                await voice_transport.send_voice_message(chat_id, ogg_bytes)
                logger.info(f"Sent long voice reply to chat {chat_id}")
                return
            # TTS failed — fall through to plain text

        transport.send_response(chat_id, response_text)
        logger.info(f"Sent text reply to chat {chat_id}")

    except Exception as e:
        logger.error(f"Text processing failed for chat {chat_id}: {e}", exc_info=True)
        try:
            transport.send_response(chat_id, "Sorry, something went wrong.")
        except Exception:
            pass



@router.get("/telegram/health")
async def telegram_health():
    """Health check for Telegram webhook."""
    try:
        transport = get_telegram_transport()
        return {
            "status": "ok",
            "bot": Config.TELEGRAM_BOT_USERNAME,
            "token_loaded": bool(Config.TELEGRAM_BOT_TOKEN),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_message_through_agent(message: NormalizedMessage) -> str:
    """
    Process a normalized message through the agent.
    
    Args:
        message: Normalized message from transport layer
        
    Returns:
        Response text to send back to user
    """
    try:
        from agent.orchestrator import SAMOrchestrator
        import uuid
        
        agent = SAMOrchestrator()
        
        # Generate conversation and trace IDs
        conversation_id = f"telegram_{message.chat_id}"
        trace_id = str(uuid.uuid4())
        
        # Invoke agent (already in async context)
        result = await agent.invoke(
            raw_input=message.content,
            conversation_id=conversation_id,
            trace_id=trace_id,
        )
        
        # Agent may return "output" or "final_output" depending on state serialization
        raw = result.get("output") or result.get("final_output") or ""
        from webhook.telegram_voice import clean_agent_response
        response = clean_agent_response(raw) or "Sorry, I couldn't process that."
        logger.info(f"Agent result keys: {list(result.keys())}, output: {response[:80] if response else None}")
        return response
    
    except Exception as e:
        logger.error(f"Agent invocation failed: {e}", exc_info=True)
        return f"Echo: {message.content}"


@router.post("/telegram/set-webhook")
async def set_telegram_webhook(webhook_url: str):
    """
    Configure Telegram webhook (admin endpoint).
    
    Args:
        webhook_url: Full webhook URL (e.g., https://example.com/webhook/telegram)
        
    Returns:
        Result from Telegram API
        
    Note:
        This is an admin endpoint. In production, secure this with authentication.
    """
    try:
        transport = get_telegram_transport()
        
        # Call Telegram API to set webhook
        import requests
        
        response = requests.post(
            f"{transport.api_url}/setWebhook",
            json={"url": webhook_url},
            timeout=10,
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to set webhook: {response.text}")
            raise HTTPException(status_code=400, detail=response.text)
        
        result = response.json()
        logger.info(f"Webhook set successfully: {result}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error setting webhook: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telegram/webhook-info")
async def get_webhook_info():
    """
    Get current webhook configuration.
    
    Returns:
        Webhook info from Telegram API
    """
    try:
        transport = get_telegram_transport()
        
        import requests
        
        response = requests.get(
            f"{transport.api_url}/getWebhookInfo",
            timeout=10,
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=response.text)
        
        return response.json()
    
    except Exception as e:
        logger.error(f"Error getting webhook info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
