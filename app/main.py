from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
from uuid import uuid4
import os

from app.config import get_settings, Settings
from app.models.schemas import (
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse
)
from app.services.document_service import DocumentService


# Initialize FastAPI application
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API using FastAPI, ChromaDB, and Gemini",
    version="1.0.0"
)

# Create temp directory for uploads
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_document_service(settings: Settings = Depends(get_settings)) -> DocumentService:
    return DocumentService(settings)


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        message="RAG API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="All systems operational"
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service)
):
    # Validate file format
    allowed_extensions = {'.pdf', '.txt'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )

    # Save uploaded file temporarily
    temp_filename = f"{uuid4()}{file_extension}"
    temp_file_path = UPLOAD_DIR / temp_filename

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process document
        document_id, num_chunks = document_service.process_document(
            file_path=str(temp_file_path),
            filename=file.filename
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_created=num_chunks,
            message="Document uploaded and processed successfully"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path.exists():
            temp_file_path.unlink()


@app.post("/documents/query", response_model=QueryResponse)
async def query_documents(
    query: QueryRequest,
    document_service: DocumentService = Depends(get_document_service)
):
    try:
        response = document_service.query_documents(
            question=query.question,
            document_id=query.document_id
        )
        return response

    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    settings = get_settings()

    # Check if API key is configured
    if not settings.gemini_api_key:
        print("WARNING API Key")
    else:
        print("Gemini API Key configured")

    print(f"ChromaDB path: {settings.chromadb_path}")
    print(f"Chunk size: {settings.chunk_size}, Overlap: {settings.chunk_overlap}")
    print(f"Top-K retrieval: {settings.top_k}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
