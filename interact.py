import asyncio
import sys
import os
from uuid import uuid4

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from agent.langgraph_orchestrator import SAMAgentOrchestrator
from inference.stub import StubModelBackend
from agent.memory.stub import StubMemoryController
from agent.memory.long_term_stub import StubLongTermMemoryStore

async def main():
    print("SAM-Agent Telegram CLI Tester")
    print("==============================")
    print("Type 'exit' or 'quit' to stop.")
    
    # Initialize components
    model = StubModelBackend()
    memory = StubMemoryController()
    ltm = StubLongTermMemoryStore()
    orchestrator = SAMAgentOrchestrator(
        model_backend=model,
        memory_controller=memory,
        long_term_memory_store=ltm
    )
    
    conversation_id = str(uuid4())
    
    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            break
            
        if user_input.lower() in ["exit", "quit"]:
            break
            
        print("Agent is thinking...")
        
        # Invoke the orchestrator
        response = await orchestrator.invoke(
            raw_input=user_input,
            conversation_id=conversation_id
        )
        
        # The response can be the formatted dict or the state object
        output = response.get("output") or response.get("final_output")
        print(f"\nSAM: {output}")
        print(f"--- [Status: {response.get('status')}, Trace ID: {response.get('trace_id')}] ---")

if __name__ == "__main__":
    # Ensure Windows handles the loop correctly
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
