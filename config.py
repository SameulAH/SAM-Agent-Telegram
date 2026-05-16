"""
Configuration management for SAM Agent.

Loads environment variables from .env file and provides typed access to configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)


class Config:
    """Configuration class for SAM Agent."""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "")
    
    # Agent API Configuration
    AGENT_PORT = int(os.getenv("AGENT_PORT", "8000"))
    
    # LLM Backend Configuration
    LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")
    
    # Environment
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # LangSmith (Optional)
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "")
    
    # Database
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./memory.db")
    
    # Long-term Memory (Qdrant)
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    LTM_COLLECTION = os.getenv("LTM_COLLECTION", "long_term_memory")

    # Security
    # Comma-separated list of allowed CORS origins.
    # Set to "*" only for local dev; restrict in production.
    # Example: ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
    ALLOWED_ORIGINS: list = [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
        if o.strip()
    ]

    # Token required in X-Debug-Token header to access /debug/* endpoints.
    # Leave empty to disable all debug endpoints regardless of LOCAL_OBSERVABILITY_ENABLED.
    DEBUG_API_TOKEN: str = os.getenv("DEBUG_API_TOKEN", "")

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is set."""
        required = ["TELEGRAM_BOT_TOKEN"]
        missing = [key for key in required if not getattr(cls, key)]
        
        if missing:
            print(f"⚠️  Missing required environment variables: {', '.join(missing)}")
            print(f"   Please set them in .env file")
            return False
        
        return True


if __name__ == "__main__":
    # Test configuration loading
    print("Configuration loaded:")
    print(f"  Telegram Bot Token: {'✓ Set' if Config.TELEGRAM_BOT_TOKEN else '✗ Missing'}")
    print(f"  Telegram Bot Username: {Config.TELEGRAM_BOT_USERNAME}")
    print(f"  Agent Port: {Config.AGENT_PORT}")
    print(f"  LLM Backend: {Config.LLM_BACKEND}")
    print(f"  Environment: {Config.ENVIRONMENT}")
    print(f"\n  Validation: {'✓ PASSED' if Config.validate() else '✗ FAILED'}")
