"""
RAG (Retrieval Augmented Generation) service.
Combines document retrieval and LLM generation for question answering.
"""

import time
import logging
from typing import Tuple, List, Dict, Any, Optional

# Import services
from app.services.vector_store import vector_store
from app.services.llm import query_llm_with_ollama
from app.core.config import Config

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
        max_tokens: Optional[int] = None
) -> Tuple[str, List[Dict[str, Any]], float]:
    """
    Complete RAG pipeline that retrieves relevant documents and generates a response

    Args:
        query: The user query
        top_k: Number of documents to retrieve
        temperature: Temperature for the LLM
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple[str, List[Dict], float]: The response, sources used, and processing time
    """
    start_time = time.time()
    logger.info(f"Starting RAG pipeline for query: {query}")

    try:
        # 1. Retrieve relevant documents from vector store
        relevant_docs = vector_store.search(query, top_k=top_k)
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")

        # 2. Generate prompt with retrieved context
        prompt = generate_prompt_from_context(query, relevant_docs)
        logger.info(f"Generated prompt with {len(prompt)} characters")

        # 3. Get model from config
        model_name = Config.get("LLM_MODEL")
        logger.info(f"Using LLM model: {model_name}")

        # 4. Generate response with LLM
        logger.info("Generating response with LLM")
        response = query_llm_with_ollama(
            prompt=prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 5. Calculate processing time
        time_taken = time.time() - start_time
        logger.info(f"RAG pipeline completed in {time_taken:.2f} seconds")

        return response, relevant_docs, time_taken

    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        time_taken = time.time() - start_time
        return f"Sorry, an error occurred: {str(e)}", [], time_taken