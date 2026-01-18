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
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
        )
    
    def add_chunks(
        self,
        document_id: str,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None  # Enhanced semantic metadata
    ) -> List[str]:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Generate unique chunk IDs
        chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        
        # Default metadata if none provided
        if metadatas is None:
            metadatas = [{"chunk_index": i} for i in range(len(chunks))]
        
        # Ensure all required fields exist
        enhanced_metadatas = []
        for i, meta in enumerate(metadatas):
            enhanced_meta = {
                "document_id": document_id,
                "chunk_index": i,
                **meta  # Merge with semantic metadata
            }
            enhanced_metadatas.append(enhanced_meta)
        
        # Add to ChromaDB
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=enhanced_metadatas
        )
        
        print(f"Stored {len(chunk_ids)} semantic chunks")
        return chunk_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        document_id: Optional[str] = None
    ) -> Tuple[List[str], List[str], List[float], List[Dict]]:
        # Prepare filter
        where_filter = None
        if document_id:
            where_filter = {"document_id": document_id}
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )
        
        # Extract results safely
        chunk_ids = results['ids'][0] if results.get('ids') else []
        chunk_texts = results['documents'][0] if results.get('documents') else []
        distances = results['distances'][0] if results.get('distances') else []
        metadatas = results['metadatas'][0] if results.get('metadatas') else []
        
        # Convert distances to similarity (cosine: 1 - distance)
        similarity_scores = [round(1 - dist, 4) for dist in distances]
        
        print(f"Found {len(chunk_ids)} relevant chunks")
        for i, meta in enumerate(metadatas[:2]):  # Log top 2
            print(f"  - [{meta.get('headline', 'N/A')}] score: {similarity_scores[i]}")
        
        return chunk_ids, chunk_texts, similarity_scores, metadatas
    
    def delete_document(self, document_id: str) -> None:
        results = self.collection.get(where={"document_id": document_id})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} chunks for document {document_id}")
    
    def get_collection_stats(self) -> Dict:
        stats = self.collection.get()
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.settings.collection_name,
            "unique_documents": len(set(stats.get('metadatas', [])).get('document_id', []))
        }
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        results = self.collection.get(where={"document_id": document_id})
        chunks = []
        for i in range(len(results['ids'])):
            chunk = {
                'id': results['ids'][i],
                'text': results['documents'][i],
                'metadata': results['metadatas'][i]
            }
            chunks.append(chunk)
        return chunks
