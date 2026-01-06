from typing import List, Tuple
from pathlib import Path
from uuid import uuid4
from app.config import Settings
from app.utils.text_extractor import TextExtractor
from app.utils.chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.models.schemas import QueryResponse, SourceChunk


class DocumentService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.text_extractor = TextExtractor()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
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

        # Chunk text
        chunks = self.chunker.chunk_text(text)

        if not chunks:
            raise ValueError("No chunks generated from document")

        # Generate embeddings
        embeddings = self.embedding_service.generate_embeddings_batch(chunks)

        # Generate unique document ID
        document_id = str(uuid4())

        # Store in vector database
        chunk_ids = self.vector_store.add_chunks(
            document_id=document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        return document_id, len(chunks)

    def query_documents(self, question: str, document_id: str = None) -> QueryResponse:
        # Generate query embedding
        query_embedding = self.embedding_service.generate_query_embedding(question)

        # Retrieve relevant chunks
        chunk_ids, chunk_texts, similarity_scores = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.settings.top_k,
            document_id=document_id
        )

        if not chunk_texts:
            return QueryResponse(
                answer="I don't know based on the provided context.",
                source_chunks=[],
                document_ids=[]
            )

        # Generate answer using LLM
        answer = self.llm_service.generate_answer(question, chunk_texts)

        # Prepare source chunks
        source_chunks = [
            SourceChunk(
                chunk_id=chunk_id,
                similarity_score=round(score, 4),
                text=text[:500]  # Truncate for response size
            )
            for chunk_id, score, text in zip(chunk_ids, similarity_scores, chunk_texts)
        ]

        # Extract unique document IDs
        doc_ids = list(set([cid.split('_chunk_')[0] for cid in chunk_ids]))

        return QueryResponse(
            answer=answer,
            source_chunks=source_chunks,
            document_ids=doc_ids
        )
