"""
RAG (Retrieval Augmented Generation) service.
Combines document retrieval and LLM generation for question answering.
"""

import time
import logging
from typing import Tuple, List, Dict, Any, AsyncGenerator

from app.services.vector_store import vector_store
from app.services.llm import query_llm_with_fallback, query_llm_with_ollama_stream
from app.core.config import Config

logger = logging.getLogger(__name__)

def rag_pipeline(
        query: str,
        top_k: int = 3,
        temperature: float = None,
        max_tokens: int = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Execute the RAG pipeline - retrieval + LLM generation

    Args:
        query: The user query
        top_k: Number of documents to retrieve
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple containing (response_text, sources, time_taken)
    """
    temp_value = temperature if temperature is not None else float(Config.get("LLM_TEMPERATURE"))

    start_time = time.time()
    logger.info(f"Starting RAG pipeline for query: {query}")

    try:
        # Retrieve relevant documents
        docs = vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(docs)} documents for context")

        for i, doc in enumerate(docs):
            logger.debug(f"Document {i + 1}: score={doc.get('score', 'N/A')}")

        # Create prompt with context
        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Generate response with LLM
        response = query_llm_with_fallback(
            prompt=prompt,
            temperature=temp_value,
            max_tokens=max_tokens
        )

        time_taken = time.time() - start_time
        logger.info(f"RAG pipeline completed in {time_taken:.2f} seconds")

        return response, docs, time_taken

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        time_taken = time.time() - start_time
        return f"Error: {str(e)}", [], time_taken


async def rag_pipeline_stream(
        query: str,
        top_k: int = None,
        temperature: float = None,
        max_tokens: int = None
) -> AsyncGenerator[str, Any]:
    """
    Execute the RAG pipeline with streaming response

    Args:
        query: The user query
        top_k: Number of documents to retrieve
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens to generate

    Returns:
        A stream of tokens generated by the LLM
    """
    logger.info(f"Starting streaming RAG pipeline for query: {query}")

    temp_value = temperature if temperature is not None else float(Config.get("LLM_TEMPERATURE"))
    top_k_value = top_k if top_k is not None else float(Config.get("LLM_TOP_K"))

    try:
        docs = vector_store.search(query, top_k=top_k_value)
        logger.info(f"Retrieved {len(docs)} documents for context")

        return generate_tokens(query, docs, temp_value, max_tokens)

    except Exception as e:
        logger.error(f"Error setting up RAG pipeline stream: {str(e)}")

        async def error_generator():
            yield f"Error: {str(e)}"

        return error_generator()

async def rag_pipeline_stream_docs(
        query: str,
        docs: List[Dict[str, Any]],
        temperature: float = None,
        max_tokens: int = None
) -> AsyncGenerator[str, None]:
    """
    Stream response for a RAG pipeline with pre-retrieved documents

    Args:
        query: The user query
        docs: Pre-retrieved documents for context
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens to generate

    Yields:
        Generated tokens
    """
    logger.info(f"Streaming RAG response with {len(docs)} pre-retrieved documents")

    temp_value = temperature if temperature is not None else float(Config.get("LLM_TEMPERATURE"))

    return generate_tokens(query, docs, temp_value, max_tokens)

async def generate_tokens(
    query: str,
    docs: list[Dict[str, Any]],
    temperature: float = None,
    max_tokens: int = None):
    try:
        temp_value = temperature if temperature is not None else float(Config.get("LLM_TEMPERATURE"))

        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        async for token in query_llm_with_ollama_stream(
                prompt=prompt,
                temperature=temp_value,
                max_tokens=max_tokens
        ):
            yield token
    except Exception as e:
        logger.error(f"Error streaming from LLM: {str(e)}")
        yield f"Error: {str(e)}"