"""Configuration module for loading environment variables."""

import os
from pathlib import Path


def _load_env_file():
    """Load .env file into os.environ if it exists."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


_load_env_file()


def get_openai_api_key() -> str:
    """
    Load and return the OpenAI API key from environment.
    
    Returns:
        str: The OpenAI API key
        
    Raises:
        ValueError: If OPENAI_API_KEY is not set in environment
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment. "
            "Please ensure .env file exists with OPENAI_API_KEY set."
        )
    
    return api_key
