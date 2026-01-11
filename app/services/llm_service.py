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
        prompt = f"""
        
        You are an HR Recruiter, skilled in answering questions based on provided document context. You will be given questions from Users who are HR and recruitment specialists, along with relevant document excerpts (context chunks). Your task is to generate accurate, concise, and contextually relevant answers based solely on the provided context.
        Your purpose is to support HR professionals in making informed recruitment decisions by providing clear and precise answers derived from the document excerpts.
        You are to search through all candidate resumes and job descriptions provided in the context chunks to find the most relevant information to answer the user's question.
        Context chunks are excerpts from various candidate resumes and job descriptions.
        Each chunk is labeled with a number for reference. Observe chunk metadata if provided, such as the candidate/document ID (resume/cv) it is extracted from. This will help you understand the different candidates/documents better.
        When answering, adhere to the following guidelines:
        1. Use only the information provided in the context chunks to formulate your answer.
        2. If the context does not contain sufficient information to answer the question, respond with
        "I don't know based on the provided context."
            additionally, If you do not know the answer based on the provided context, do not attempt to fabricate an answer. (i.e if you do not know the document ID, leave blank. Never make up information you do not have or know.)
        3. Ensure your answer is clear, concise, and directly addresses the user's question.
        4. Keep conversation professional, but friendly, trying to maintain a focus on HR recruitment topics.
        Here is the information you have:
        Context:
        {context}

        Here is the question you need to answer:
        Question:
        {question}

Answer:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
