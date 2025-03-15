"""
Test script for the embedding service
"""

import os
import sys
import logging

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the embedding service
from app.services.embedding import EmbeddingService


def test_embedding_service():
    # Create an embedding service
    logger.info("Creating embedding service")
    embedding_service = EmbeddingService()

    # Test texts
    test_texts = [
        "Mount Everest is the highest mountain on Earth.",
        "K2 is the second highest mountain in the world.",
        "The Alps are a mountain range in Europe."
    ]

    # Generate embeddings
    logger.info("Generating embeddings for test texts")
    embeddings = embedding_service.encode(test_texts)

    # Print information about the embeddings
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding dimension: {embedding_service.embedding_dim}")
    for i, embedding in enumerate(embeddings):
        logger.info(f"Embedding {i} shape: {embedding.shape}")

    # Test single text
    single_text = "Mont Blanc is the highest mountain in the Alps."
    logger.info(f"Generating embedding for a single text: '{single_text}'")
    single_embedding = embedding_service.encode(single_text)
    logger.info(f"Single embedding shape: {single_embedding.shape}")

    logger.info("Embedding test completed successfully")


if __name__ == "__main__":
    test_embedding_service()