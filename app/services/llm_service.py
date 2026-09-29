from google import genai

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class LLMService:

    def __init__(self):
        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        self.model = GEMINI_MODEL

    def generate_answer(
        self,
        question: str,
        context: str,
        history: str = "",
    ) -> str:

        prompt = f"""
    You are an AI assistant that answers questions
    using the provided document context.

    Rules:

    1. Use the document context to answer.
    2. Use conversation history to understand
    references such as "it", "that", or "the previous policy".
    3. Do not invent information.
    4. If the answer is not available in the
    provided context, say that the information
    was not found in the documents.
    5. Give a clear and concise answer.

    CONVERSATION HISTORY:

    {history}

    DOCUMENT CONTEXT:

    {context}

    CURRENT USER QUESTION:

    {question}

    ANSWER:
    """

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text