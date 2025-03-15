"""
Test script for the vector store module
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

# Import the vector store service
from app.services.vector_store import VectorStore


def test_vector_store():
    # Create a vector store
    logger.info("Creating vector store")
    store = VectorStore(
        collection_name="mountains_data",
        persist_directory=os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")
    )

    # Get collection stats
    logger.info("Getting collection stats")
    stats = store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")

    # Test search
    test_queries = [
        "What is the height of Mount Everest?",
        "Tell me about mountains in the Alps",
        "Which is the highest mountain in South America?"
    ]

    for query in test_queries:
        logger.info(f"Searching for: '{query}'")
        results = store.search(query, top_k=2)
        logger.info(f"Found {len(results)} results")

        for i, result in enumerate(results):
            logger.info(f"Result {i + 1}:")
            logger.info(f"  Document: {result['text'][:100]}...")
            logger.info(f"  Score: {result['score']}")
            if 'metadata' in result:
                logger.info(f"  Mountain: {result['metadata'].get('mountain', 'Unknown')}")
                logger.info(f"  Height: {result['metadata'].get('height_m', 'Unknown')} m")

    logger.info("Vector store test completed successfully")


if __name__ == "__main__":
    test_vector_store()