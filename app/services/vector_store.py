"""
Vector store module for interacting with ChromaDB.
"""

import chromadb
import os
import logging
from typing import List, Dict, Any

# Import embedding service and configuration
from app.services.embedding import embedding_service
from app.core.config import Config

logger = logging.getLogger(__name__)

class VectorStore:
    """Interface for working with ChromaDB vector store"""

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        """
        Initialize connection to ChromaDB

        Args:
            collection_name: Name of the collection to use
            persist_directory: Path where ChromaDB data is persisted
        """
        # Use provided parameters or defaults from config
        self.collection_name = collection_name or Config.get("VECTOR_COLLECTION")
        self.persist_directory = persist_directory or Config.get("VECTOR_DB_PATH")

        # Fallback to hardcoded defaults if still None
        if self.collection_name is None:
            self.collection_name = "mountains_data"
            logger.warning(f"No collection name found in config, using default: {self.collection_name}")

        if self.persist_directory is None:
            self.persist_directory = "data/chroma_db"
            logger.warning(f"No persist directory found in config, using default: {self.persist_directory}")

        # Create persistence directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(f"Initializing ChromaDB with path: {self.persist_directory}")

        try:
            # Create ChromaDB client with persistence
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Try to get existing collection or create a new one
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Using existing collection: {self.collection_name}")
            except Exception as e:
                logger.info(f"Collection not found, creating new collection: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
                )

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for documents most similar to the query

        Args:
            query: The query text
            top_k: Number of results to return

        Returns:
            List[Dict]: List of most relevant documents with their metadata
        """
        # Generate embedding for the query
        query_embedding = embedding_service.encode(query)

        try:
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=top_k
            )

            # Format results
            documents = []
            for i in range(len(results['ids'][0])):
                doc = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'score': float(results['distances'][0][i]) if 'distances' in results else None
                }

                # Add metadata if available
                if 'metadatas' in results and results['metadatas'][0]:
                    doc['metadata'] = results['metadatas'][0][i]

                documents.append(doc)

            logger.info(f"Found {len(documents)} relevant documents for query")
            return documents

        except Exception as e:
            logger.error(f"Error searching in ChromaDB: {str(e)}")
            raise

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return statistics about the current collection

        Returns:
            Dict: Collection statistics
        """
        try:
            # Get all IDs to count the number of documents
            all_ids = self.collection.get()['ids']
            count = len(all_ids)

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "collection_name": self.collection_name,
                "error": str(e),
                "persist_directory": self.persist_directory
            }

# Create a singleton instance
vector_store = VectorStore()