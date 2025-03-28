"""
Data preparation script for the mountain dataset.
Loads a CSV file of mountains, generates embeddings, and stores them in ChromaDB.
"""

import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import argparse

# Add parent directory to path to import app modules if running standalone
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from app.core.config import Config, PROJECT_ROOT
    logger.info("Using app config for paths")
except ImportError:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logger.info(f"App config not available, using manual PROJECT_ROOT: {PROJECT_ROOT}")

def load_mountains_data(csv_path):
    """
    Load mountains dataset from CSV file
    """
    logger.info(f"Loading mountains data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} mountain records")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {str(e)}")
        raise

def prepare_mountains_for_embeddings(df):
    """
    Prepare mountain data for embedding generation by creating descriptive texts
    """
    logger.info("Preparing mountain data for embeddings")

    # Create descriptive texts for each mountain
    texts = []
    for _, row in df.iterrows():
        text = f"{row['Mountain']} is a mountain with a height of {row['Metres']} meters ({row['Feet']} feet). "
        if pd.notna(row['Range']) and row['Range'] != 'nan' and row['Range'].strip() != "":
            text += f"It is part of the {row['Range']} range and "
        else:
            text += "Its range is unknown and "
        if pd.notna(row['Location']) and row['Location'] != 'nan' and row['Location'].strip() != "":
            text += f"is located in {row['Location']}."
        else:
            text += "its location is unknown."
        texts.append(text)

    # Use mountain names as titles
    titles = df['Mountain'].tolist()

    # Add mountain data as metadata
    metadata = []
    for _, row in df.iterrows():
        meta = {
            "mountain": row['Mountain'],
            "height_m": row['Metres'],
            "height_ft": row['Feet'],
            "range": row['Range'],
            "location": row['Location']
        }
        metadata.append(meta)

    return texts, titles, metadata

def create_vector_store(texts, metadata, collection_name="mountains_data", persist_directory="../data/chroma_db"):
    """
    Create embeddings and store them in ChromaDB
    """
    logger.info("Creating embeddings and storing in ChromaDB")

    os.makedirs(persist_directory, exist_ok=True)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    logger.info(f"Generating embeddings for {len(texts)} mountains")
    embeddings = model.encode(texts, show_progress_bar=True)

    logger.info(f"Initializing ChromaDB with persistence at: {persist_directory}")
    chroma_client = chromadb.PersistentClient(path=persist_directory)

    # Check if collection exists and delete if necessary - compatible with ChromaDB v0.6.0+
    try:
        collection_names = chroma_client.list_collections()
        if collection_name in [coll for coll in collection_names]:
            logger.info(f"Collection {collection_name} already exists, recreating it")
            chroma_client.delete_collection(name=collection_name)
    except Exception as e:
        logger.error(f"Error checking existing collections: {str(e)}")
        # Attempt to continue by assuming we need to recreate the collection
        try:
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Attempted to delete existing collection: {collection_name}")
        except Exception:
            # Ignore if the collection doesn't exist
            pass

    # Create collection
    try:
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    except Exception as e:
        logger.error(f"Failed to create collection, trying again: {str(e)}")
        # Last attempt: Force recreate collection
        chroma_client.delete_collection(name=collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Add data to ChromaDB
    logger.info(f"Adding {len(texts)} mountains to ChromaDB")
    collection.add(
        ids=[f"mountain_{i}" for i in range(len(texts))],
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        documents=texts
    )

    logger.info(f"Successfully stored {len(texts)} mountains in ChromaDB")
    return collection

def main(args=None):
    parser = argparse.ArgumentParser(description='Prepare mountain data and load into ChromaDB')

    # Use Config for default paths if available
    try:
        default_csv_path = Config.get("MOUNTAIN_CSV_PATH", os.path.join(PROJECT_ROOT, 'scripts', 'mountain.csv'))
        default_db_path = Config.get("VECTOR_DB_PATH", os.path.join(PROJECT_ROOT, 'data', 'chroma_db'))
        default_collection = Config.get("VECTOR_COLLECTION", 'mountains_data')
    except (NameError, AttributeError):
        # Fallback to hardcoded defaults
        default_csv_path = os.path.join(PROJECT_ROOT, 'scripts', 'mountain.csv')
        default_db_path = os.path.join(PROJECT_ROOT, 'data', 'chroma_db')
        default_collection = 'mountains_data'

    parser.add_argument('--csv_path', default=default_csv_path, help='Path to mountains CSV file')
    parser.add_argument('--collection', default=default_collection, help='Name for ChromaDB collection')
    parser.add_argument('--persist_dir', default=default_db_path, help='Directory for ChromaDB persistence')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    os.makedirs(args.persist_dir, exist_ok=True)

    logger.info(f"Loading data from: {args.csv_path}")
    df = load_mountains_data(args.csv_path)

    texts, titles, metadata = prepare_mountains_for_embeddings(df)

    create_vector_store(
        texts=texts,
        metadata=metadata,
        collection_name=args.collection,
        persist_directory=args.persist_dir
    )

    logger.info("Mountain data preparation completed successfully")

if __name__ == "__main__":
    main()