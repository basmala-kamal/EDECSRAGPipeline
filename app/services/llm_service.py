import google.generativeai as genai
from typing import List
from app.config import Settings


class LLMService:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_generation_model)

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        # Construct context from chunks
        context = "\n\n".join([f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])

        # Create controlled prompt
        prompt = f"""Use ONLY the context below to answer the question.
If the answer is not present in the context, respond with "I don't know based on the provided context."
Do not use external knowledge or make assumptions.

Context:
{context}

Question:
{question}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
