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
import asyncio
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
def test_query(query, use_rag=True, temperature=0.7, top_k=3):
    """Test the complete RAG pipeline with a query"""
    if use_rag:
        from app.services.rag import rag_pipeline

        logger.info("=" * 50)
        logger.info(f"TESTING RAG PIPELINE with query: '{query}'")
        logger.info(f"Parameters: top_k={top_k}, temperature={temperature}")
        logger.info("=" * 50)

        # Start timer
        start_time = time.time()

        # Execute the RAG pipeline
        response, sources, processing_time = rag_pipeline(
            query=query,
            top_k=top_k,
            temperature=temperature
        )

        # Log results
        logger.info(f"Pipeline completed in {processing_time:.2f} seconds")
        logger.info(f"Found {len(sources)} relevant documents:")

        # Display sources if any
        for i, source in enumerate(sources):
            logger.info(f"SOURCE {i + 1}:")
            logger.info(f"  Text: {source['text'][:100]}...")
            if 'metadata' in source and source['metadata']:
                logger.info(f"  Mountain: {source['metadata'].get('mountain', 'Unknown')}")
                logger.info(f"  Height: {source['metadata'].get('height_m', 'Unknown')} m / {source['metadata'].get('height_ft', 'Unknown')} ft")
                logger.info(f"  Location: {source['metadata'].get('location', 'Unknown')}")
            logger.info(f"  Relevance score: {source.get('score', 'N/A')}")
            logger.info("-" * 30)

        # Display response
        logger.info("\nRESPONSE:")
        logger.info(f"{response}")

        return response, sources
    else:
        from app.services.llm import query_llm_with_fallback

        logger.info("=" * 50)
        logger.info(f"TESTING DIRECT LLM QUERY with query: '{query}'")
        logger.info(f"Parameters: temperature={temperature}")
        logger.info("=" * 50)

        prompt = f"Question: {query}\nAnswer:"

        # Start timer
        start_time = time.time()

        # Execute direct LLM query
        response = query_llm_with_fallback(
            prompt=prompt,
            temperature=temperature
        )

        processing_time = time.time() - start_time

        # Log results
        logger.info(f"Query completed in {processing_time:.2f} seconds")
        logger.info("\nRESPONSE:")
        logger.info(f"{response}")

        return response, None


async def test_query_stream(query, use_rag=True, temperature=0.7, top_k=3):
    """Test streaming query with the RAG pipeline"""
    if use_rag:
        from app.services.rag import rag_pipeline_stream

        logger.info("=" * 50)
        logger.info(f"TESTING RAG PIPELINE STREAM with query: '{query}'")
        logger.info(f"Parameters: top_k={top_k}, temperature={temperature}")
        logger.info("=" * 50)

        # Start timer
        start_time = time.time()

        # Initialize stream
        token_generator = await rag_pipeline_stream(
            query=query,
            top_k=top_k,
            temperature=temperature
        )

        # Collect tokens
        tokens = []
        print("Streaming response: ", end="", flush=True)

        async for token in token_generator:
            tokens.append(token)
            print(token, end="", flush=True)

        print("\n")  # Add newline after tokens

        full_response = "".join(tokens)
        processing_time = time.time() - start_time

        # Log results
        logger.info(f"Streaming completed in {processing_time:.2f} seconds")
        logger.info(f"Total tokens received: {len(tokens)}")

        return full_response

    else:
        from app.services.llm import query_llm_stream

        logger.info("=" * 50)
        logger.info(f"TESTING DIRECT LLM STREAM with query: '{query}'")
        logger.info(f"Parameters: temperature={temperature}")
        logger.info("=" * 50)

        # Start timer
        start_time = time.time()

        # Collect tokens
        tokens = []
        print("Streaming response: ", end="", flush=True)

        async for token in query_llm_stream(
            query=query,
            temperature=temperature
        ):
            tokens.append(token)
            print(token, end="", flush=True)

        print("\n")  # Add newline after tokens

        full_response = "".join(tokens)
        processing_time = time.time() - start_time

        # Log results
        logger.info(f"Streaming completed in {processing_time:.2f} seconds")
        logger.info(f"Total tokens received: {len(tokens)}")

        return full_response


# Main function to run the test
def main():
    parser = argparse.ArgumentParser(description='Test the complete RAG system')
    parser.add_argument('--query', type=str, default="What is the height of Mount Everest?",
                        help='Query to test')
    parser.add_argument('--no-rag', action='store_true',
                        help='Use direct LLM query without RAG')
    parser.add_argument('--stream', action='store_true',
                        help='Use streaming response')
    parser.add_argument('--check-only', action='store_true',
                        help='Only check the database without querying')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Run data preparation script if database is empty')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for LLM generation (0.1-2.0)')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of documents to retrieve')
    args = parser.parse_args()

    # Check the database
    db_ok = check_db()

    if args.check_only:
        return

    if not db_ok and args.stream and not args.no_rag:
        logger.warning("Database appears to be empty. RAG may not work correctly.")

    # Test the query
    if args.stream:
        asyncio.run(test_query_stream(
            query=args.query,
            use_rag=not args.no_rag,
            temperature=args.temperature,
            top_k=args.top_k
        ))
    else:
        test_query(
            query=args.query,
            use_rag=not args.no_rag,
            temperature=args.temperature,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()