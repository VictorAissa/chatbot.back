"""
Pydantic models for the API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    """Model for chat request"""
    query: str = Field(..., description="User query")
    use_rag: bool = Field(True, description="Whether to use RAG or query LLM directly")
    top_k: int = Field(3, description="Number of documents to retrieve")
    temperature: float = Field(0.7, description="Temperature for the LLM (0.0 to 1.0)")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")

class DocumentInfo(BaseModel):
    """Model for document information"""
    id: str = Field(..., description="Document ID")
    text: str = Field(..., description="Document text")
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Document metadata")

class ChatResponse(BaseModel):
    """Model for chat response"""
    response: str = Field(..., description="LLM response")
    sources: Optional[List[DocumentInfo]] = Field(None, description="Sources used for RAG")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    collection_stats: Dict[str, Any] = Field(..., description="Vector collection statistics")