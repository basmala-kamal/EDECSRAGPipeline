from typing import List, Tuple
from pathlib import Path
from uuid import uuid4
from app.config import Settings
from app.utils.text_extractor import TextExtractor
from app.utils.resume_semantic_chunker import ResumeSemanticChunker  # NEW IMPORT
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.models.schemas import QueryResponse, SourceChunk


class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_extractor = TextExtractor()
        self.chunker = ResumeSemanticChunker(  # NEW: Semantic chunker
            max_chunk_tokens=settings.chunk_size,
            overlap=settings.chunk_overlap
        )
        self.embedding_service = EmbeddingService(settings)
        self.vector_store = VectorStore(settings)
        self.llm_service = LLMService(settings)
    
    def process_document(self, file_path: str, filename: str) -> Tuple[str, int]:
        # Extract file extension
        file_extension = Path(filename).suffix
        
        # Extract text
        text = self.text_extractor.extract_text(file_path, file_extension)
        
        # Semantic resume chunking
        semantic_chunks = self.chunker.chunk_resume(text)
        
        if not semantic_chunks:
            raise ValueError("No chunks generated from document")
        
        print(f"Generated {len(semantic_chunks)} semantic chunks")
        for i, chunk in enumerate(semantic_chunks[:3]): 
            print(f"  - [{chunk['headline']}] {chunk['content'][:100]}...")
        
        # Prepare texts for embeddings (headline + content)
        chunks = [f"{chunk['headline']}: {chunk['content']}" for chunk in semantic_chunks]
        
        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_batch(chunks)
        
        # Generate unique document ID
        document_id = str(uuid4())
        
        # Enhanced metadata with semantic info
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "headline": semantic_chunks[i]['headline'],
                "chunk_type": semantic_chunks[i]['chunk_type'],
                "confidence": semantic_chunks[i].get('confidence', 'unknown'),
                "content_preview": semantic_chunks[i]['content'][:500],
                "full_content_length": len(semantic_chunks[i]['content'])
            }
            for i in range(len(chunks))
        ]
        
        # Store in vector database
        chunk_ids = self.vector_store.add_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings,
            metadatas=metadatas  # Enhanced metadata
        )
        
        return document_id, len(semantic_chunks)
    
    def query_documents(self, question: str, document_id: str = None) -> QueryResponse:
        # Generate query embedding
        try:
            query_embedding = self.embedding_service.generate_query_embedding(question)
        except Exception as e:
            raise RuntimeError(f"Query embedding failed: {str(e)}")
        
        # Retrieve relevant chunks with metadata
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
            document_id=document_id
        )
        
        chunk_ids, chunk_texts, similarity_scores, metadatas = search_results
        
        if not chunk_texts:
            return QueryResponse(
                answer="I don't know based on the provided context.",
                source_chunks=[],
                document_ids=[]
            )
        
        source_chunks = [
            SourceChunk(
                chunk_id=chunk_ids[i],
                similarity_score=round(similarity_scores[i], 4),
                text=chunk_texts[i][:500],
                headline=metadatas[i].get('headline', 'CONTENT'),
                chunk_type=metadatas[i].get('chunk_type', 'unknown'),
                confidence=metadatas[i].get('confidence', 'low')
            )
            for i in range(len(chunk_texts))
        ]
        
        # Generate answer using LLM
        answer = self.llm_service.generate_answer(question, source_chunks)
        
        # Extract document IDs
        doc_ids = list(set([m.get('document_id', 'unknown') for m in metadatas]))
        
        return QueryResponse(
            answer=answer,
            source_chunks=source_chunks,
            document_ids=doc_ids
        )
