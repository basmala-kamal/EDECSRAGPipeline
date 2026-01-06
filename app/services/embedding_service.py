import google.generativeai as genai
from typing import List
from app.config import Settings


class EmbeddingService:
 
    def __init__(self, settings: Settings):

        self.settings = settings
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = settings.gemini_embedding_model

    def generate_embedding(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    def generate_query_embedding(self, query: str) -> List[float]:
       
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {str(e)}")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
       
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings
