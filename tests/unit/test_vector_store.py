"""
Test script for the vector store module
"""

import os
import sys
import logging
import argparse

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the vector store service - singleton version
from app.services.vector_store import vector_store


def test_vector_store(query=None):
    """Test the vector store service with sample queries or a specific query"""

    # Get collection stats
    logger.info("Getting collection stats")
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")

    # Check if the collection is empty
    if stats.get("document_count", 0) == 0:
        logger.warning("The collection is empty. Please run the data preparation script first.")
        return

    # Use provided query or sample queries
    test_queries = [query] if query else [
        "What is the height of Mount Everest?",
        "Tell me about mountains in the Alps",
        "Which is the highest mountain in South America?"
    ]

    for query in test_queries:
        logger.info(f"Searching for: '{query}'")
        results = vector_store.search(query, top_k=2)
        logger.info(f"Found {len(results)} results")

        for i, result in enumerate(results):
            logger.info(f"Result {i + 1}:")
            logger.info(f"  Document: {result['text'][:100]}...")
            logger.info(f"  Score: {result.get('score', 'N/A')}")
            if 'metadata' in result:
                logger.info(f"  Mountain: {result['metadata'].get('mountain', 'Unknown')}")
                logger.info(f"  Height: {result['metadata'].get('height_m', 'Unknown')} m")
                logger.info(f"  Location: {result['metadata'].get('location', 'Unknown')}")

    logger.info("Vector store test completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the vector store with a query')
    parser.add_argument('--query', type=str, help='Optional query to test')
    args = parser.parse_args()

    test_vector_store(args.query)