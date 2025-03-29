"""
Test script for the LLM module
"""

import os
import sys
import logging
import time
import argparse
import asyncio
from dotenv import load_dotenv, find_dotenv

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the LLM functions and config
from app.services.llm import query_llm_with_fallback, query_llm_with_ollama_api, query_llm_stream
from app.core.config import Config

async def test_streaming(query, model_name=None, temperature=0.7):
    """Test the streaming functionality"""
    logger.info(f"Testing streaming with query: '{query}'")

    response_parts = []
    async for token in query_llm_stream(
        query=query,
        temperature=temperature,
        max_tokens=None
    ):
        response_parts.append(token)
        # Print just a dot to show progress without cluttering the console
        print(".", end="", flush=True)

    print()  # New line after dots
    full_response = "".join(response_parts)
    logger.info(f"Full streamed response: {full_response[:200]}..." if len(full_response) > 200 else f"Full streamed response: {full_response}")

def test_llm(model_name=None, temperature=0.7):
    """
    Test the LLM module with a specific model

    Args:
        model_name: Name of the model to test (if None, uses config)
        temperature: Temperature setting for generation
    """
    # Load default model if none specified
    if model_name is None:
        model_name = Config.get("LLM_MODEL", "gemma2:2b")

    logger.info(f"Testing with model: {model_name} (temperature: {temperature})")

    # Test direct query
    test_query = "What are mountains?"
    prompt = f"Question: {test_query}\nAnswer:"

    logger.info("Testing direct LLM query")
    logger.info(f"Query: '{test_query}'")

    start_time = time.time()
    response = query_llm_with_fallback(
        prompt=prompt,
        model_name=model_name,
        temperature=temperature
    )
    duration = time.time() - start_time

    logger.info(f"Response received in {duration:.2f} seconds")
    logger.info(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

    # Test with context
    logger.info("\nTesting LLM with context")
    context = """
    Mount Everest is a mountain with a height of 8848 meters (29029 feet). 
    It is part of the Himalayas range and is located in Nepal/China.
    K2 is a mountain with a height of 8612 meters (28255 feet).
    It is part of the Karakoram range and is located in Pakistan/China.
    """

    prompt_with_context = f"""Context:
{context}

Question: What is the height difference between Mount Everest and K2?

Answer:"""

    logger.info("Sending prompt with context")
    logger.info("Question : What is the height difference between Mount Everest and K2?")
    start_time = time.time()
    response = query_llm_with_fallback(
        prompt=prompt_with_context,
        model_name=model_name,
        temperature=temperature
    )
    duration = time.time() - start_time

    logger.info(f"Response received in {duration:.2f} seconds")
    logger.info(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

    # Test with streaming
    logger.info("\nTesting streaming functionality")
    asyncio.run(test_streaming("What makes mountains different from hills?", model_name, temperature))

    logger.info(f"LLM test completed successfully for model: {model_name}")

    return duration

def test_all_models():
    """Test all available models and compare their performance"""
    models = ["gemma2:2b", "mistral:latest"]
    results = {}

    for model in models:
        logger.info(f"\n{'='*50}\nTesting model: {model}\n{'='*50}")

        # Set environment variable
        os.environ["LLM_MODEL"] = model

        # Test the model
        duration = test_llm(model)
        results[model] = duration

    # Compare results
    logger.info("\n" + "="*50)
    logger.info("Performance comparison:")
    for model, duration in results.items():
        logger.info(f"Model: {model} - Response time: {duration:.2f} seconds")
    logger.info("="*50)

if __name__ == "__main__":
    # Load environment variables
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description='Test LLM module')
    parser.add_argument('--model', help='Specific model to test')
    parser.add_argument('--all', action='store_true', help='Test all available models')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature setting (0.1-2.0)')
    parser.add_argument('--query', type=str, help='Specific query to test')
    args = parser.parse_args()

    if args.query:
        logger.info(f"Testing specific query: {args.query}")
        if args.model:
            os.environ["LLM_MODEL"] = args.model
        asyncio.run(test_streaming(args.query, args.model, args.temperature))
    elif args.all:
        test_all_models()
    else:
        test_llm(args.model, args.temperature)