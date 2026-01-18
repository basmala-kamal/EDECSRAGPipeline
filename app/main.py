from typing import List
from app.models.schemas import BulkUploadResponse, BulkUploadResult
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

# Bulk upload endpoint (moved below app initialization)
@app.post("/documents/bulk_upload", response_model=BulkUploadResponse)
async def bulk_upload_documents(
    files: List[UploadFile] = File(...),
    document_service: DocumentService = Depends(get_document_service)
):
    allowed_extensions = {'.pdf', '.txt'}
    results = []
    for file in files:
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in allowed_extensions:
            results.append(BulkUploadResult(
                document_id=None,
                filename=file.filename,
                chunks_created=None,
                message="Failed",
                error=f"Unsupported file format. Allowed: {allowed_extensions}"
            ))
            continue

        temp_filename = f"{uuid4()}{file_extension}"
        temp_file_path = UPLOAD_DIR / temp_filename
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            document_id, num_chunks = document_service.process_document(
                file_path=str(temp_file_path),
                filename=file.filename
            )
            results.append(BulkUploadResult(
                document_id=document_id,
                filename=file.filename,
                chunks_created=num_chunks,
                message="Document uploaded and processed successfully",
                error=None
            ))
        except Exception as e:
            results.append(BulkUploadResult(
                document_id=None,
                filename=file.filename,
                chunks_created=None,
                message="Failed",
                error=str(e)
            ))
        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()
    return BulkUploadResponse(results=results)


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
    import zipfile
    import tarfile
    import gzip
    import tempfile
    allowed_extensions = {'.pdf', '.txt'}
    compressed_extensions = {'.zip', '.tar', '.tar.gz', '.tgz', '.gz'}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension in compressed_extensions:
        with tempfile.TemporaryDirectory() as extract_dir:
            temp_path = UPLOAD_DIR / f"{uuid4()}{file_extension}"
            try:
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                # Handle .zip
                if file_extension == '.zip':
                    with zipfile.ZipFile(temp_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                # Handle .tar, .tar.gz, .tgz
                elif file_extension in {'.tar', '.tar.gz', '.tgz'}:
                    with tarfile.open(temp_path, 'r:*') as tar_ref:
                        tar_ref.extractall(extract_dir)
                # Handle .gz (single file)
                elif file_extension == '.gz':
                    # Assume the .gz contains a single file
                    import shutil as sh
                    base_name = Path(file.filename).stem
                    out_path = os.path.join(extract_dir, base_name)
                    with gzip.open(temp_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
                        sh.copyfileobj(f_in, f_out)
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported compressed file type: {file_extension}")

                results = []
                for root, _, files in os.walk(extract_dir):
                    for fname in files:
                        ext = Path(fname).suffix.lower()
                        if ext not in allowed_extensions:
                            results.append({
                                "filename": fname,
                                "message": f"Skipped unsupported file type: {ext}"
                            })
                            continue
                        fpath = os.path.join(root, fname)
                        try:
                            document_id, num_chunks = document_service.process_document(
                                file_path=fpath,
                                filename=fname
                            )
                            results.append({
                                "document_id": document_id,
                                "filename": fname,
                                "chunks_created": num_chunks,
                                "message": "Document uploaded and processed successfully"
                            })
                        except Exception as e:
                            results.append({
                                "filename": fname,
                                "message": f"Failed: {str(e)}"
                            })
                return JSONResponse(content={"results": results})
            finally:
                if temp_path.exists():
                    temp_path.unlink()
    elif file_extension in allowed_extensions:
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
            if temp_file_path.exists():
                temp_file_path.unlink()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions | compressed_extensions}"
        )


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
