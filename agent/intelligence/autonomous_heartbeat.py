import asyncio
import logging
import random
from datetime import datetime
from typing import Optional

from transport.telegram.transport import create_telegram_transport
from agent.memory import MemoryController, LongTermMemoryStore
from agent.memory.long_term_qdrant import QdrantLongTermMemoryStore
from agent.memory.sqlite import SQLiteShortTermMemoryStore
from inference.ollama import OllamaModelBackend
from inference.types import ModelRequest
from config import Config

logger = logging.getLogger(__name__)

class AutonomousHeartbeat:
    """
    The agent's 'Heartbeat' service.
    Handles proactive messaging (e.g. Good Morning messages)
    and background reflection cleanup.
    """
    
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.transport = create_telegram_transport()
        
        # Initialize dependencies for proactive thought
        self.ltm = QdrantLongTermMemoryStore(
            qdrant_url=Config.QDRANT_URL,
            collection_name=Config.LTM_COLLECTION
        )
        self.model = OllamaModelBackend(
            model_name=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL
        )

    async def send_morning_greeting(self):
        """
        Wakes up, reads LTM, and sends a personalized morning message.
        """
        logger.info(f"AutonomousHeartbeat: Triggering morning greeting for {self.chat_id}")
        
        # 1. Retrieve recent context about the user
        facts = []
        try:
            # Get latest 5 facts/reflections
            fact_results = self.ltm.scroll_facts(
                conversation_id=f"telegram_{self.chat_id}",
                limit=5
            )
            facts = [f.content.get("text", "") for f in fact_results]
        except Exception as e:
            logger.error(f"Heartbeat: Failed to read LTM: {e}")

        # 2. Generate a personalized greeting using the LLM
        prompt = f"It is 8:00 AM. Generate a warm, best-friend style morning greeting for Ismail."
        if facts:
            prompt += f"\nRecent context about Ismail:\n- " + "\n- ".join(facts)
        
        system_prompt = (
            "You are SAM, a warm, loyal best friend. "
            "Generate a one-sentence morning greeting that feels alive and conscious. "
            "Reference a recent interest if possible."
        )

        try:
            response = self.model.generate(ModelRequest(
                task="proactive_greeting",
                prompt=prompt,
                system_prompt=system_prompt,
                timeout_s=30
            ))
            
            message = response.output if response.status == "success" else "Good morning Ismail! Thinking of you and our chat yesterday. Hope you have an amazing day! ☀️"
            
            # 3. Send via Telegram
            self.transport.send_text(self.chat_id, message)
            logger.info(f"Heartbeat: Sent morning greeting to {self.chat_id}")
            
        except Exception as e:
            logger.error(f"Heartbeat: Failed to generate/send greeting: {e}")

    async def run_forever(self):
        """
        Simple loop to simulate a daily heartbeat.
        In a real prod environment, we'd use APScheduler, but for this 
        MVP consciousness, a sleep loop is more robust within the container.
        """
        logger.info("AutonomousHeartbeat: Service started")
        while True:
            now = datetime.now()
            # Trigger between 8:00 and 8:01 AM
            if now.hour == 8 and now.minute == 0:
                await self.send_morning_greeting()
                # Sleep 60s to avoid double-trigger
                await asyncio.sleep(60)
            
            # Check every minute
            await asyncio.sleep(30)

if __name__ == "__main__":
    # Test script
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    # Use the test chat ID from environment
    test_chat_id = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "903341171")
    heartbeat = AutonomousHeartbeat(test_chat_id)
    
    # Run once for testing
    asyncio.run(heartbeat.send_morning_greeting())
