"""
Test script for the LLM module
"""

import os
import sys
import logging
import time
import argparse
from dotenv import load_dotenv, set_key

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import the LLM functions and config
from app.services.llm import query_llm_with_ollama, query_llm_directly
from app.core.config import Config

def test_llm(model_name=None):
    """
    Test the LLM module with a specific model

    Args:
        model_name: Name of the model to test (if None, uses config)
    """
    if model_name:
        logger.info(f"Testing with model: {model_name}")
    else:
        model_name = Config.get("LLM_MODEL")
        logger.info(f"Testing with default model from config: {model_name}")

    # Test direct query
    test_query = "What are mountains?"

    logger.info("Testing direct LLM query")
    logger.info(f"Query: '{test_query}'")

    start_time = time.time()
    response, duration = query_llm_directly(test_query)
    total_time = time.time() - start_time

    logger.info(f"Response received in {duration:.2f} seconds (total: {total_time:.2f}s)")
    logger.info(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

    # Test with context
    logger.info("\nTesting LLM with context")
    context = """
    Mount Everest is a mountain with a height of 8848 meters (29029 feet). 
    It is part of the Himalayas range and is located in Nepal/China.
    K2 is a mountain with a height of 8612 meters (28255 feet).
    It is part of the Karakoram range and is located in Pakistan/China.
    """

    prompt = f"""Context:
{context}

Question: What is the height difference between Mount Everest and K2?

Answer:"""

    logger.info("Sending prompt with context to Ollama")
    logger.info("Question : What is the height difference between Mount Everest and K2?")
    start_time = time.time()
    response = query_llm_with_ollama(prompt, model_name=model_name)
    duration = time.time() - start_time

    logger.info(f"Response received in {duration:.2f} seconds")
    logger.info(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

    logger.info(f"LLM test completed successfully for model: {model_name}")

    return duration

def test_all_models():
    """Test all available models and compare their performance"""
    models = ["gemma2:2b", "mistral:latest"]
    results = {}

    for model in models:
        logger.info(f"\n{'='*50}\nTesting model: {model}\n{'='*50}")
        # Update the .env file
        dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        os.environ["LLM_MODEL"] = model
        if os.path.exists(dotenv_path):
            set_key(dotenv_path, "LLM_MODEL", model)
            load_dotenv(dotenv_path, override=True)

        # Reload configuration
        Config.get_all()

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
    parser = argparse.ArgumentParser(description='Test LLM module')
    parser.add_argument('--model', help='Specific model to test')
    parser.add_argument('--all', action='store_true', help='Test all available models')
    args = parser.parse_args()

    if args.all:
        test_all_models()
    elif args.model:
        test_llm(args.model)
    else:
        test_llm()