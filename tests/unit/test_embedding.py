"""
Test script for the embedding service
"""

import os
import sys
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the embedding service singleton
from app.services.embedding import embedding_service


def test_embedding_service():
    """Test the embedding service"""

    logger.info(f"Testing embedding service with model: {embedding_service.model}")
    logger.info(f"Embedding dimension: {embedding_service.embedding_dim}")

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
    for i, embedding in enumerate(embeddings):
        logger.info(f"Embedding {i} shape: {embedding.shape}")

    # Test single text
    single_text = "Mont Blanc is the highest mountain in the Alps."
    logger.info(f"Generating embedding for a single text: '{single_text}'")
    single_embedding = embedding_service.encode(single_text)
    logger.info(f"Single embedding shape: {single_embedding.shape}")

    # Test similarity between embeddings
    logger.info("Testing similarity between embeddings")

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    logger.info("Similarity matrix:")
    for i in range(len(test_texts)):
        for j in range(len(test_texts)):
            logger.info(f"Similarity between text {i+1} and text {j+1}: {similarity_matrix[i][j]:.4f}")

    # Test similarity with the single text
    similarities = cosine_similarity([single_embedding], embeddings)[0]
    logger.info("Similarities with single text:")
    for i, sim in enumerate(similarities):
        logger.info(f"Similarity with text {i+1}: {sim:.4f}")

    # Find most similar text
    most_similar_idx = np.argmax(similarities)
    logger.info(f"Most similar text to '{single_text}' is: '{test_texts[most_similar_idx]}'")
    logger.info(f"Similarity score: {similarities[most_similar_idx]:.4f}")

    logger.info("Embedding test completed successfully")


if __name__ == "__main__":
    test_embedding_service()