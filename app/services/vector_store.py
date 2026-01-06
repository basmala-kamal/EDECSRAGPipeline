import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
from app.config import Settings


class VectorStore:
   
    def __init__(self, settings: Settings):
      
        self.settings = settings
        self.client = chromadb.PersistentClient(
            path=settings.chromadb_path,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]]
    ) -> List[str]:
       
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        # Generate unique chunk IDs
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]

        # Prepare metadata
        metadatas = [
            {
                "document_id": document_id,
                "chunk_index": i,
                "chunk_text": chunk[:1000]  # Store truncated text in metadata
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return chunk_ids

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> Tuple[List[str], List[str], List[float]]:
       
        # Prepare query filter
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter
        )

        # Extract results
        chunk_ids = results['ids'][0] if results['ids'] else []
        chunk_texts = results['documents'][0] if results['documents'] else []
        distances = results['distances'][0] if results['distances'] else []
        similarity_scores = [1 - dist for dist in distances]

        return chunk_ids, chunk_texts, similarity_scores

    def delete_document(self, document_id: str) -> None:
        # Get all chunks for document
        results = self.collection.get(
            where={"document_id": document_id}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])

    def get_collection_stats(self) -> Dict:
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.settings.collection_name
        }
