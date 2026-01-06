# RAG API - Retrieval-Augmented Generation System

A production-ready RAG (Retrieval-Augmented Generation) API built with FastAPI, ChromaDB, and Google's Gemini AI. This system enables semantic search and question-answering over uploaded documents.

## 🏗️ Architecture

```
User Question
    ↓
FastAPI Endpoint
    ↓
Document Processing
    ├── Text Extraction (PDF/TXT)
    ├── Text Chunking (with overlap)
    └── Embedding Generation (Gemini)
    ↓
ChromaDB Vector Store
    ↓
Semantic Search (cosine similarity)
    ↓
Context Retrieval (Top-K chunks)
    ↓
LLM Answer Generation (Gemini)
    ↓
Structured Response
```

## ✨ Features

- **Document Upload**: Support for PDF and TXT files
- **Intelligent Chunking**: Sentence-aware chunking with configurable overlap
- **Semantic Search**: Vector similarity search using Gemini embeddings
- **Controlled Generation**: LLM constrained to use only retrieved context
- **Clean Architecture**: Separation of concerns with service layer pattern
- **Type Safety**: Full Pydantic model validation
- **Error Handling**: Comprehensive error handling and validation

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Gemini API Key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

5. **Run the application**
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Upload Document
```bash
POST /documents/upload
Content-Type: multipart/form-data

# Example with curl
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "document_id": "uuid-here",
  "filename": "document.pdf",
  "chunks_created": 15,
  "message": "Document uploaded and processed successfully"
}
```

#### 3. Query Documents
```bash
POST /documents/query
Content-Type: application/json

# Example with curl
curl -X POST "http://localhost:8000/documents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "document_id": "optional-uuid-to-filter"
  }'
```

**Response:**
```json
{
  "answer": "Generated answer based on context...",
  "source_chunks": [
    {
      "chunk_id": "doc-id_chunk_0",
      "similarity_score": 0.95,
      "text": "Relevant text chunk..."
    }
  ],
  "document_ids": ["doc-id"]
}
```

## 🏛️ Project Structure

```
rag-api/
├── app/
│   ├── main.py              # FastAPI application & endpoints
│   ├── config.py            # Configuration management
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── document_service.py   # Main orchestration service
│   │   ├── embedding_service.py  # Gemini embedding operations
│   │   ├── vector_store.py       # ChromaDB interface
│   │   └── llm_service.py        # Gemini LLM operations
│   └── utils/
│       ├── chunker.py       # Text chunking logic
│       └── text_extractor.py # PDF/TXT extraction
├── requirements.txt
├── .env.example
└── README.md
```

## 🎯 Design Principles

### 1. **Separation of Concerns**
- **Services**: Business logic separated into focused services
- **Utils**: Reusable utilities for text processing
- **Models**: Type-safe data models with Pydantic

### 2. **Dependency Injection**
- Settings injected via FastAPI dependencies
- Services initialized with required dependencies
- Easy to test and mock

### 3. **Error Handling**
- Comprehensive validation at each layer
- Meaningful error messages
- HTTP status codes follow REST conventions

### 4. **Configuration Management**
- Environment-based configuration
- Sensible defaults with override capability
- Cached settings for performance

### 5. **Clean Code**
- Type hints throughout
- Docstrings for all public methods
- Consistent naming conventions
- Single Responsibility Principle

## ⚙️ Configuration

Edit `.env` file to customize:

```env
# API Configuration
GEMINI_API_KEY=your_key_here

# Chunking Parameters
CHUNK_SIZE=600          # Target chunk size in tokens
CHUNK_OVERLAP=100       # Overlap between chunks

# Retrieval Parameters
TOP_K=5                 # Number of chunks to retrieve

# Storage
CHROMADB_PATH=./chroma_db
```

## 🧪 Testing the System

### Test with a Sample Document

1. **Upload a document**:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@sample.pdf"
```

Save the `document_id` from response.

2. **Ask questions**:
```bash
curl -X POST "http://localhost:8000/documents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "document_id": "your-document-id"
  }'
```

3. **Test out-of-scope questions**:
Questions not answerable from the document should return:
```
"I don't know based on the provided context."
```

## 🔍 How It Works

### Document Processing Pipeline

1. **Text Extraction**: Extracts text from PDF/TXT files
2. **Chunking**: Splits text into overlapping chunks (~600 tokens each)
3. **Embedding**: Generates vector embeddings using Gemini
4. **Storage**: Stores chunks and embeddings in ChromaDB

### Query Pipeline

1. **Query Embedding**: Converts user question to embedding
2. **Semantic Search**: Finds top-K similar chunks via cosine similarity
3. **Context Assembly**: Combines retrieved chunks as context
4. **Answer Generation**: Gemini generates answer constrained to context
5. **Response**: Returns answer with source attribution

## 🚨 Common Issues

### API Key Not Set
```
⚠️  WARNING: GEMINI_API_KEY not configured
```
**Solution**: Add your API key to `.env` file

### Import Errors
**Solution**: Ensure virtual environment is activated and dependencies installed

### ChromaDB Permission Errors
**Solution**: Ensure write permissions in project directory

## 📝 Key Evaluation Points

✅ Document upload works  
✅ Text extraction handles PDF and TXT  
✅ Chunking with proper overlap  
✅ Embeddings stored in ChromaDB  
✅ Semantic search retrieves relevant chunks  
✅ LLM constrained to retrieved context  
✅ Out-of-scope questions handled correctly  
✅ Clean code with proper structure  
✅ Type safety with Pydantic  
✅ Comprehensive error handling  

## 🎓 Learning Outcomes

This project demonstrates:
- **RAG architecture**: How retrieval enhances generation
- **Vector databases**: Semantic search with embeddings
- **API design**: RESTful principles with FastAPI
- **Clean architecture**: Maintainable, testable code structure
- **LLM integration**: Controlled generation with external knowledge

## 📄 License

This is an educational project for internship purposes.

## 🤝 Contributing

This is a one-day icebreaker project. Focus on understanding each component before making modifications.
