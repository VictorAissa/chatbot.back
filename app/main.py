"""
Main FastAPI application for the Mountains RAG Chatbot.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
import importlib.util
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import configuration
from app.core.config import Config, PROJECT_ROOT
from app.services.vector_store import vector_store


# Ensure data directories exist with proper structure
def ensure_data_structure():
    """Ensure the data directory structure is properly set up"""
    # Ensure main data directory exists
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Ensuring data directory exists: {data_dir}")

    # Ensure ChromaDB directory exists
    chroma_dir = os.path.join(data_dir, "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)

    # Check if we need to consolidate data from tests/data or app/data
    test_data_dir = os.path.join(PROJECT_ROOT, "tests", "data")
    app_data_dir = os.path.join(PROJECT_ROOT, "app", "data")

    # Check for mountain.csv
    mountain_csv = os.path.join(data_dir, "mountain.csv")
    if not os.path.exists(mountain_csv):
        logger.warning(f"mountain.csv not found in main data directory: {mountain_csv}")

        # Look in tests/data
        test_csv = os.path.join(test_data_dir, "mountain.csv")
        if os.path.exists(test_csv):
            logger.info(f"Found mountain.csv in tests/data, copying to main data directory")
            shutil.copy2(test_csv, mountain_csv)

        # Look in app/data
        app_csv = os.path.join(app_data_dir, "mountain.csv")
        if os.path.exists(app_csv):
            logger.info(f"Found mountain.csv in app/data, copying to main data directory")
            shutil.copy2(app_csv, mountain_csv)

    # Check for ChromaDB data
    test_chroma = os.path.join(test_data_dir, "chroma_db")
    app_chroma = os.path.join(app_data_dir, "chroma_db")

    # If main ChromaDB is empty but test ChromaDB exists and has data
    if os.path.exists(test_chroma) and os.listdir(test_chroma):
        if not os.listdir(chroma_dir):
            logger.info(f"Found ChromaDB data in tests/data, copying to main data directory")
            shutil.rmtree(chroma_dir)  # Remove empty directory
            shutil.copytree(test_chroma, chroma_dir)

    # If main ChromaDB is empty but app ChromaDB exists and has data
    if os.path.exists(app_chroma) and os.listdir(app_chroma):
        if not os.listdir(chroma_dir):
            logger.info(f"Found ChromaDB data in app/data, copying to main data directory")
            shutil.rmtree(chroma_dir)  # Remove empty directory
            shutil.copytree(app_chroma, chroma_dir)

# Ensure proper data structure before anything else
ensure_data_structure()

# Now import vector_store after data structure is set up
from app.services.vector_store import vector_store as vector_store_instance

# Function to check DB state
def check_db():
    """Check the database state before starting the API"""
    global vector_store_instance
    stats = vector_store_instance.get_collection_stats()
    doc_count = stats.get("document_count", 0)

    logger.info(f"Database check: Collection '{stats.get('collection_name')}' contains {doc_count} documents")

    if doc_count == 0:
        logger.warning("WARNING: The vector database is empty. Prepare data before using RAG functionality.")
        logger.warning("Run 'python scripts/prepare_data.py' to populate the database")

        # Automatically run data preparation script if requested
        if os.environ.get("AUTO_PREPARE_DATA", "").lower() in ("true", "1", "yes"):
            try:
                logger.info("Attempting to automatically prepare data...")
                # If we have a CSV file but no database, prepare the data
                csv_path = Config.get("MOUNTAIN_CSV_PATH")

                if os.path.exists(csv_path):
                    logger.info(f"Found mountain data CSV at: {csv_path}")

                    # Get the path to the prepare_data.py script
                    script_path = os.path.join(PROJECT_ROOT, "scripts", "prepare_data.py")

                    # Import and run the prepare_data module
                    if os.path.exists(script_path):
                        spec = importlib.util.spec_from_file_location("prepare_data", script_path)
                        prepare_data = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(prepare_data)

                        # Override the default CSV path
                        prepare_data.main(["--csv_path", csv_path])

                        # Check DB state again and reload vector_store
                        if "app.services.vector_store" in sys.modules:
                            del sys.modules["app.services.vector_store"]
                        from app.services.vector_store import vector_store
                        stats = vector_store.get_collection_stats()
                        logger.info(f"After data preparation: Collection contains {stats.get('document_count', 0)} documents")
                    else:
                        logger.error(f"Data preparation script not found at {script_path}")
                else:
                    logger.error(f"Mountain data CSV not found at {csv_path}")
            except Exception as e:
                logger.error(f"Failed to prepare data: {str(e)}")

    return doc_count > 0

# Check DB state
db_has_documents = check_db()

# Create FastAPI app
app = FastAPI(
    title="Mountains RAG Chatbot API",
    description="An API for querying information about mountains using RAG",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include our router
from app.api.routes import router
app.include_router(router, prefix="/api")

# Add startup event handler
@app.on_event("startup")
async def startup_event():
    """Run when the API starts up"""
    logger.info("Starting Mountain RAG API")
    if not db_has_documents:
        logger.warning("API started but the vector database is empty - RAG functionality will be limited")
    else:
        logger.info("Vector database is populated and ready")

# Add a root endpoint that redirects to docs
@app.get("/")
async def root_redirect():
    """Redirect root to API endpoint"""
    return {
        "message": "Welcome to the Mountains RAG API",
        "docs_url": "/docs",
        "db_status": "populated" if db_has_documents else "empty"
    }

if __name__ == "__main__":
    # Start the server if run directly
    import uvicorn
    host = Config.get("API_HOST")
    port = int(Config.get("API_PORT"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)