"""
End-to-end test script for the complete RAG system.
This script tests the entire pipeline: loading data, searching for relevant documents,
and generating a response with the LLM.
"""

import os
import sys
import logging
import time
import argparse
from pprint import pprint

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# First, check if database contains mountains
def check_db():
    """Check the ChromaDB database for collections and documents"""
    from app.services.vector_store import vector_store

    logger.info("=" * 50)
    logger.info("CHECKING DATABASE")
    logger.info("=" * 50)

    # Get collection stats
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")

    # Check if the collection is empty
    if stats.get('document_count', 0) == 0:
        logger.warning("The collection is empty! You need to load data first.")
        logger.info("Run the data preparation script: python scripts/prepare_data.py")
        return False
    else:
        logger.info(f"Collection contains {stats.get('document_count')} documents - good!")
        return True


# Test a query with the full RAG pipeline
def test_query(query, use_rag=True):
    """Test the complete RAG pipeline with a query"""
    if use_rag:
        from app.services.rag import rag_pipeline

        logger.info("=" * 50)
        logger.info(f"TESTING RAG PIPELINE with query: '{query}'")
        logger.info("=" * 50)

        # Start timer
        start_time = time.time()

        # Execute the RAG pipeline
        response, sources, processing_time = rag_pipeline(query, top_k=3)

        # Log results
        logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
        logger.info(f"Found {len(sources)} relevant documents:")

        # Display sources if any
        for i, source in enumerate(sources):
            logger.info(f"SOURCE {i + 1}:")
            logger.info(f"  Text: {source['text'][:100]}...")
            if 'metadata' in source:
                logger.info(f"  Metadata: {source['metadata']}")
            logger.info(f"  Relevance score: {source.get('score', 'N/A')}")
            logger.info("-" * 30)

        # Display response
        logger.info("\nRESPONSE:")
        logger.info(f"{response}")

        return response, sources
    else:
        from app.services.llm import query_llm_directly

        logger.info("=" * 50)
        logger.info(f"TESTING DIRECT LLM QUERY with query: '{query}'")
        logger.info("=" * 50)

        # Execute direct LLM query
        response, processing_time = query_llm_directly(query)

        # Log results
        logger.info(f"Query completed in {processing_time:.2f} seconds")
        logger.info("\nRESPONSE:")
        logger.info(f"{response}")

        return response, None


# Main function to run the test
def main():
    parser = argparse.ArgumentParser(description='Test the complete RAG system')
    parser.add_argument('--query', type=str, default="What is the height of Mount Everest?",
                        help='Query to test')
    parser.add_argument('--no-rag', action='store_true',
                        help='Use direct LLM query without RAG')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check the database without querying')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Run data preparation script if database is empty')
    args = parser.parse_args()

    # Check the database
    db_ok = check_db()

    if not db_ok and args.prepare_data:
        logger.info("Automatically running data preparation script...")
        try:
            import scripts.prepare_data
            scripts.prepare_data.main()
            db_ok = check_db()  # Check again after preparation
        except Exception as e:
            logger.error(f"Failed to prepare data: {str(e)}")

    if args.check_only:
        return

    if not db_ok:
        logger.warning("Proceeding with query even though database may be empty...")

    # Test the query
    test_query(args.query, not args.no_rag)


if __name__ == "__main__":
    main()