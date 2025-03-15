"""
Configuration module for the RAG application.
Loads settings from environment variables or a .env file.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Get the project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Load environment variables from .env file if it exists
dotenv_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded environment from {dotenv_path}")

class Config:
    """Configuration class for the RAG application"""

    # Default configurations with absolute paths
    DEFAULT_CONFIG = {
        # LLM settings
        "LLM_MODEL": "gemma2:2b",                              # Default LLM model
        "LLM_TEMPERATURE": 0.7,                               # Default temperature
        "LLM_TOP_K": 40,                                      # Default top_k
        "LLM_TOP_P": 0.9,                                     # Default top_p

        # Vector DB settings - using absolute paths
        "VECTOR_DB_PATH": os.path.join(PROJECT_ROOT, "data", "chroma_db"),  # Default path for vector DB
        "VECTOR_COLLECTION": "mountains_data",                 # Default collection name

        # Data settings
        "MOUNTAIN_CSV_PATH": os.path.join(PROJECT_ROOT, "data", "mountain.csv"),  # Path to mountain CSV

        # API settings
        "API_HOST": "0.0.0.0",                                # Default host
        "API_PORT": 8000,                                     # Default port
        "CORS_ORIGINS": "*",                                  # Default CORS origins

        # Embedding settings
        "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2"  # Default embedding model
    }

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from environment variables or default config

        Args:
            key: The configuration key to get
            default: Optional default value if not found

        Returns:
            The configuration value
        """
        # If default is None, use the DEFAULT_CONFIG value
        if default is None and key in cls.DEFAULT_CONFIG:
            default = cls.DEFAULT_CONFIG[key]

        # Get from environment variable
        return os.environ.get(key, default)