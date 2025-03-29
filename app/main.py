"""
Main FastAPI application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys
import importlib.util
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from app.core.config import Config, PROJECT_ROOT

cors_origins_str = Config.get("CORS_ORIGINS", "*")
if cors_origins_str == "*":
    origins = ["*"]
else:
    origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]


def ensure_data_structure():
    """Ensure the data directory structure is properly set up"""
    # Ensure main data directory exists
    data_dir = os.path.join(PROJECT_ROOT, "data")
    scripts_dir = os.path.join(PROJECT_ROOT, "scripts")
    os.makedirs(data_dir, exist_ok=True)
    logger.info(f"Ensuring data directory exists: {data_dir}")

    # Ensure ChromaDB directory exists
    chroma_dir = os.path.join(data_dir, "chroma_db")
    os.makedirs(chroma_dir, exist_ok=True)

    # Check for mountain.csv
    mountain_csv = os.path.join(scripts_dir, "mountain.csv")
    if not os.path.exists(mountain_csv):
        logger.warning(f"mountain.csv not found in main data directory: {mountain_csv}")

ensure_data_structure()

from app.services.vector_store import vector_store as vector_store_instance

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
                csv_path = Config.get("MOUNTAIN_CSV_PATH")

                if os.path.exists(csv_path):
                    logger.info(f"Found mountain data CSV at: {csv_path}")

                    script_path = os.path.join(PROJECT_ROOT, "scripts", "prepare_data.py")

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length"]
)

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

@app.get("/")
async def root_redirect():
    """Redirect root to API endpoint"""
    return {
        "message": "Welcome to the Mountains RAG API",
        "docs_url": "/docs",
        "db_status": "populated" if db_has_documents else "empty"
    }

if __name__ == "__main__":
    import uvicorn
    host = Config.get("API_HOST")
    port = int(Config.get("API_PORT"))
    uvicorn.run("app.main:app", host=host, port=port, reload=True)