"""
API routes.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
import logging
import os
import json
import asyncio

from app.api.models import ChatRequest, ChatResponse, HealthResponse, DocumentInfo
from app.core.config import Config
from app.services.rag import rag_pipeline, rag_pipeline_stream_docs
from app.services.llm import query_llm_directly, query_llm_stream
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["Mountain RAG API"])

cors_origins_str = Config.get("CORS_ORIGINS", "*").strip()

@router.get("/", response_model=HealthResponse)
def read_root():
    """Root endpoint to check if the API is running"""
    stats = vector_store.get_collection_stats()
    return {
        "status": "ok",
        "version": "0.1.0",
        "collection_stats": stats
    }

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with or without RAG context
    """
    try:
        logger.info(f"Received query: {request.query}")

        # Check if database has documents before using RAG
        stats = vector_store.get_collection_stats()
        doc_count = stats.get("document_count", 0)

        if request.use_rag and doc_count == 0:
            logger.warning("RAG requested but vector database is empty. Falling back to direct LLM query.")
            logger.warning("Run 'python scripts/prepare_data.py' to populate the database")
            response, time_taken = query_llm_directly(
                query=request.query,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            sources_info = None

        elif request.use_rag:
            # Use RAG pipeline
            response, sources, time_taken = rag_pipeline(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )

            # Check if we got any relevant documents
            if not sources or len(sources) == 0:
                logger.warning(f"RAG found no relevant documents for query: {request.query}")
                # Optional: Add a note to the response
                if "context does not contain" not in response.lower():
                    response = f"{response}\n\nNote: No relevant documents were found in the database to supplement this answer."

            # Convert sources to DocumentInfo model
            sources_info = [
                DocumentInfo(
                    id=source.get('id', ''),
                    text=source.get('text', ''),
                    score=source.get('score'),
                    metadata=source.get('metadata')
                ) for source in sources
            ] if sources else None

        else:
            # Query LLM directly without RAG
            logger.info("Using direct LLM query (RAG disabled)")
            response, time_taken = query_llm_directly(
                query=request.query,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            sources_info = None

        return ChatResponse(
            response=response,
            sources=sources_info,
            processing_time=time_taken
        )

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/prepare-data")
async def prepare_data(background_tasks: BackgroundTasks):
    """
    Endpoint to trigger data preparation in the background
    """
    try:
        # Get the path to the prepare_data.py script
        script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts", "prepare_data.py")

        if not os.path.exists(script_path):
            raise HTTPException(status_code=404, detail=f"Data preparation script not found")

        # Define a function to run in background
        def run_data_preparation():
            try:
                logger.info("Starting data preparation in background...")
                # Import and run the prepare_data module
                import importlib.util
                spec = importlib.util.spec_from_file_location("prepare_data", script_path)
                prepare_data = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(prepare_data)
                prepare_data.main()
                logger.info("Data preparation completed successfully")
            except Exception as e:
                logger.error(f"Data preparation failed: {str(e)}")

        # Add the task to run in background
        background_tasks.add_task(run_data_preparation)

        return {"status": "started", "message": "Data preparation started in background"}

    except Exception as e:
        logger.error(f"Failed to start data preparation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.route("/chat/stream", methods=["GET", "POST"])
async def stream_chat_endpoint(request: Request):
    try:
        if request.method == "GET":
            query = request.query_params.get("query", "")
            use_rag = request.query_params.get("use_rag", "true").lower() == "true"
            top_k = int(request.query_params.get("top_k", "3"))
            temperature = float(request.query_params.get("temperature", "0.7"))
        else:  # POST
            data = await request.json()
            query = data.get("query", "")
            use_rag = data.get("use_rag", True)
            top_k = data.get("top_k", 3)
            temperature = data.get("temperature", 0.7)

        async def generate_sse_stream():
            try:
                yield "data: {\"type\": \"start\"}\n\n"

                if use_rag:
                    docs = vector_store.search(query, top_k=top_k)
#                    sources_info = [
#                        {
#                            "id": doc.get('id', ''),
#                            "text": doc.get('text', ''),
#                            "score": doc.get('score'),
#                            "metadata": doc.get('metadata')
#                        } for doc in docs
#                    ]
#                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources_info})}\n\n"

                    async for token in rag_pipeline_stream_docs(
                            query=query,
                            docs=docs,
                            temperature=temperature,
                            max_tokens=None
                    ):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        # Delay ti avoid buffering problems
                        await asyncio.sleep(0.01)
                else:
                    async for token in query_llm_stream(
                            query=query,
                            temperature=temperature,
                            max_tokens=None
                    ):
                        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                        await asyncio.sleep(0.01)

                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error(f"Error in stream: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"

        request_origin = request.headers.get("Origin")

        allowed_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

        if cors_origins_str == "*":
            response_origin = "*"
        elif request_origin and request_origin in allowed_origins:
            response_origin = request_origin
        else:
            response_origin = allowed_origins[0] if allowed_origins else "*"

        return StreamingResponse(
            generate_sse_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": response_origin,
                "Content-Encoding": "identity"
            }
        )

    except Exception as e:
        logger.error(f"Error setting up chat stream: {str(e)}")

        async def error_stream():
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*"
            }
        )

logger = logging.getLogger(__name__)