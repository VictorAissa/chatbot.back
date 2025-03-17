"""
RAG (Retrieval Augmented Generation) service.
Combines document retrieval and LLM generation for question answering.
"""

import time
import logging
from typing import Tuple, List, Dict, Any, AsyncGenerator

# Import services
from app.services.vector_store import vector_store
from app.services.llm import query_llm_with_ollama, query_llm_with_ollama_stream

logger = logging.getLogger(__name__)


def generate_prompt_from_context(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Generate a prompt for the LLM based on the retrieved context documents

    Args:
        query: The user query
        contexts: List of context documents with their metadata

    Returns:
        str: The formatted prompt
    """
    # Build context string from the documents
    context_parts = []
    for doc in contexts:
        # Extract mountain metadata if available
        metadata = doc.get('metadata', {})
        if metadata:
            mountain_name = metadata.get('mountain', 'Unknown Mountain')
            context_parts.append(f"Mountain: {mountain_name}\n{doc['text']}")
        else:
            context_parts.append(doc['text'])

    context_text = "\n\n".join(context_parts)

    # Construct the prompt
    prompt = f"""Context:
{context_text}

Question: {query}

Instructions:
- Answer the question based on the provided context about mountains
- Be concise and accurate with mountain heights, locations, and other facts
- If the context doesn't contain enough information to answer the question, say so
- If the question asks for a comparison between mountains, use the exact figures from the context
- Format your answer in a clear, readable way

Answer:"""

    return prompt


def rag_pipeline(
        query: str,
        top_k: int = 3,
        temperature: float = 0.7,
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
    start_time = time.time()
    logger.info(f"Starting RAG pipeline for query: {query}")

    try:
        # Step 1: Retrieve relevant documents
        docs = vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(docs)} documents for context")

        # Log the document scores for analysis
        for i, doc in enumerate(docs):
            logger.debug(f"Document {i + 1}: score={doc.get('score', 'N/A')}")

        # Step 2: Create prompt with context
        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Step 3: Generate response with LLM
        response = query_llm_with_ollama(
            prompt=prompt,
            temperature=temperature,
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
        top_k: int = 3,
        temperature: float = 0.7,
        max_tokens: int = None
) -> Tuple[AsyncGenerator[str, None], List[Dict[str, Any]]]:
    """
    Execute the RAG pipeline with streaming response

    Args:
        query: The user query
        top_k: Number of documents to retrieve
        temperature: Temperature for LLM generation
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple containing (token_generator, sources)
    """
    logger.info(f"Starting streaming RAG pipeline for query: {query}")

    try:
        # Step 1: Retrieve relevant documents
        docs = vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(docs)} documents for context")

        # Create a function that will yield tokens
        async def generate_tokens():
            try:
                # Step 2: Create prompt with context
                context = "\n\n".join([doc["text"] for doc in docs])
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

                # Step 3: Stream response from LLM
                async for token in query_llm_with_ollama_stream(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                ):
                    yield token
            except Exception as e:
                logger.error(f"Error streaming from LLM: {str(e)}")
                yield f"Error: {str(e)}"

        return generate_tokens(), docs

    except Exception as e:
        logger.error(f"Error setting up RAG pipeline stream: {str(e)}")

        # Return an error generator and empty docs
        async def error_generator():
            yield f"Error: {str(e)}"

        return error_generator(), []


async def rag_pipeline_stream_docs(
        query: str,
        docs: List[Dict[str, Any]],
        temperature: float = 0.7,
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

    try:
        # Create prompt with context
        context = "\n\n".join([doc["text"] for doc in docs])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        # Stream response from LLM
        async for token in query_llm_with_ollama_stream(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
        ):
            yield token

    except Exception as e:
        logger.error(f"Error in RAG streaming with pre-retrieved docs: {str(e)}")
        yield f"Error: {str(e)}"