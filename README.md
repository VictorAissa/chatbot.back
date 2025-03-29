# Mountain RAG Chatbot

A Retrieval Augmented Generation (RAG) chatbot application for answering questions about mountains, built with FastAPI, ChromaDB, and Ollama.

## System Architecture

- **Backend**: FastAPI application
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Runtime**: Ollama (running Gemma 2)

## Data Preparation

The system includes a data preparation pipeline to load mountain information into the vector database:

### Data Preparation Script (`prepare_data.py`)

This script processes mountain data and creates the vector database:

1. **Data Loading**: Loads mountain information from a CSV file
2. **Text Preparation**: Converts mountain records into descriptive texts
   - Each mountain is described with its height, range, and location
   - Example: "Mount Everest is a mountain with a height of 8848.0 meters (29029 feet). It is part of the Himalayas range and is located in Nepal/China."
3. **Embedding Generation**: Creates vector embeddings using Sentence Transformers
4. **Vector Database Creation**: Stores embeddings and metadata in ChromaDB
   - Mountain information is stored as metadata for easy retrieval
   - Embeddings enable semantic search for relevant mountains

### Usage

You can populate the database by running:

```bash
python scripts/prepare_data.py
```

## API Routes (`routes.py`)

The API routes expose endpoints for interacting with the system:

- `/chat`: Standard endpoint returning complete responses
- `/chat/stream`: Streaming endpoint for real-time token generation
- `/`: Health check endpoint providing system status

## Request Flow

### Standard Request Flow (with RAG)

1. Client sends query to `/chat` endpoint with RAG enabled
2. Vector store searches for relevant documents using query embeddings
3. Top k documents are retrieved and combined into context
4. LLM generates a response based on the context and query
5. Response, sources, and timing information are returned to client

### Streaming Request Flow

1. Client establishes SSE connection to `/chat/stream` endpoint
2. Vector store retrieves relevant documents (if RAG is enabled)
3. Context is built from retrieved documents
4. LLM generates tokens one at a time
5. Tokens are streamed to client as they are generated
6. Client receives and displays tokens incrementally

### Direct LLM Query (without RAG)

1. Client sends query with `use_rag: false`
2. Query is sent directly to LLM without document retrieval
3. LLM generates response based only on its pre-trained knowledge
4. Response is returned to client (either complete or streaming)

## Configuration Parameters

### RAG Options

- `use_rag`: Enable/disable RAG functionality (boolean)
- `top_k`: Number of documents to retrieve (integer, typically 3-5)

### LLM Generation Options

- `temperature`: Controls randomness in responses (float, 0.1-2.0)
  - Lower values (0.1-0.7): More deterministic, focused responses
  - Higher values (1.0-2.0): More creative, diverse responses
- `max_tokens`: Maximum tokens to generate (optional integer)


## Usage

### Prerequisites

- Python 3.10+ installed
- Python package installer (pip)
- [Ollama](https://ollama.ai) installed (used to run the Gemma 2 model locally)
- 4GB+ of RAM recommended for model execution

### Installation

1. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
      ```
   ```shell
   # On Windows
   venv\Scripts\activate
   ```
   
   ```shell
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the Gemma 2 model in Ollama:
   ```bash
   ollama pull gemma2:2b
   ```

### Preparing the Data

Before using the chatbot, you need to prepare the vector database with mountain data:

```bash
python scripts/prepare_data.py
```

### Testing the System

Run a quick system test to make sure everything is working:

```bash
python tests/test_system.py --query "What is the height of Mount Everest?" --prepare-data
```

You can test with different parameters:

```bash
# Test with a different query
python tests/test_system.py --query "Tell me about mountains in Europe"

# Test direct LLM query without RAG
python tests/test_system.py --query "What is a mountain?" --no-rag

# Test with streaming response (tokens appear one by one)
python tests/test_system.py --query "Compare the Alps and the Himalayas" --stream

# Customize generation parameters
python tests/test_system.py --query "Which mountain is the hardest to climb?" --temperature 1.2 --top-k 5
```

### Running the API Server

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The server will start at http://localhost:8000

### Using the API via Swagger UI

FastAPI provides an interactive Swagger UI for testing the API:
http://localhost:8000/docs

### Troubleshooting

- **Ollama issues**: Make sure Ollama is running with `ollama serve`
- **Empty responses**: Check if the database was properly populated with `python tests/test_system.py --check-only`
- **Slow responses**: Lower the `top_k` value or consider using a more powerful machine
