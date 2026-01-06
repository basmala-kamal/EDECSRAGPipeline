from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    filename: str = Field(..., description="Original filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    message: str = Field(..., description="Success message")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    document_id: Optional[str] = Field(None, description="Optional document ID to filter search")


class SourceChunk(BaseModel):
    chunk_id: str
    similarity_score: float
    text: str


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer from the RAG system")
    source_chunks: List[SourceChunk] = Field(..., description="Source chunks used for generation")
    document_ids: List[str] = Field(..., description="Document IDs from which context was retrieved")


class HealthResponse(BaseModel):
    status: str
    message: str
