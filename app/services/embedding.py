"""
Embedding service for generating text embeddings using Sentence Transformers.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding service with a Sentence Transformer model

        Args:
            model_name: Name of the Sentence Transformer model to use
        """
        logger.info(f"Initializing EmbeddingService with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for one or more texts

        Args:
            texts: A text string or list of text strings to encode
            batch_size: Batch size for encoding

        Returns:
            numpy.ndarray: Generated embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Handle empty or None texts
        texts = [text if text is not None else "" for text in texts]

        try:
            embeddings = self.model.encode(texts, batch_size=batch_size)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise


# Create a singleton instance
embedding_service = EmbeddingService()