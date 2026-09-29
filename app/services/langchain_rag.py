from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

from app.core.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)

from app.services.langchain_retriever import (
    LangChainRetriever,
)


class LangChainRAGService:

    def __init__(self):

        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

        self.retriever_service = (
            LangChainRetriever()
        )

        # Query rewriting
        self.rewrite_prompt = ChatPromptTemplate.from_template(
            """
Given the conversation history and the latest user question,
rewrite the latest question into a standalone search query.

Rules:

1. Resolve references such as "it", "they", "that",
   and "the previous policy".
2. Preserve the user's original meaning.
3. Do not answer the question.
4. Return only the rewritten search query.

CONVERSATION HISTORY:

{history}

LATEST QUESTION:

{question}

STANDALONE SEARCH QUERY:
"""
        )

        self.rewrite_chain = (
            self.rewrite_prompt | self.llm
        )

        # Answer generation
        self.prompt = ChatPromptTemplate.from_template(
    """
You are an AI assistant that answers questions
using the provided document context.

Rules:

1. Answer ONLY using the provided document context.
2. Use conversation history to understand references.
3. Do not invent information.
4. Every factual statement must have a source citation.
5. Cite sources using exactly this format:
   [Source 1]
   [Source 2]
6. Only cite sources that actually support the statement.
7. If multiple sources support a statement, cite all relevant sources.
8. If the answer cannot be found in the context, say:
   "The information was not found in the provided documents."
9. Do not create citations that do not exist.
10. Keep the answer clear and concise.

CONVERSATION HISTORY:

{history}

DOCUMENT CONTEXT:

{context}

CURRENT QUESTION:

{question}

ANSWER:
"""
)

        self.chain = self.prompt | self.llm

    def rewrite_question(
        self,
        question: str,
        history: str,
    ) -> str:

        response = self.rewrite_chain.invoke(
            {
                "question": question,
                "history": history,
            }
        )

        return self.extract_text(
                response.content
            ).strip()

    def retrieve(
        self,
        question: str,
        history: str = "",
        top_k: int = 5,
        document_id: int | None = None,
    ):

        if history.strip():

            search_query = self.rewrite_question(
                question=question,
                history=history,
            )

        else:

            search_query = question

        return self.retriever_service.retrieve(
            query=search_query,
            top_k=top_k,
            document_id=document_id,
        )

    def build_context(self, documents):

        context_parts = []

        for index, document in enumerate(documents, start=1):

            metadata = document.metadata

            context_parts.append(
                f"""
    [Source {index}]

    Document: {metadata.get("filename", "Unknown")}
    Page: {metadata.get("page_number", "Unknown")}
    Chunk ID: {metadata.get("chunk_id", "Unknown")}

    Content:
    {document.page_content}
    """
            )

        return "\n\n---\n\n".join(context_parts)

    def build_sources(
        self,
        documents,
    ):

        sources = []

        for index, document in enumerate(documents, start=1):

            metadata = document.metadata

            sources.append(
                {
                    "source_id": f"Source {index}",
                    "chunk_id": metadata.get(
                        "chunk_id"
                    ),
                    "document_id": metadata.get(
                        "document_id"
                    ),
                    "filename": metadata.get(
                        "filename"
                    ),
                    "page_number": metadata.get(
                        "page_number"
                    ),
                    "score": metadata.get("score"),
                }
            )

        return sources

    def generate(self, question: str, context: str):
        prompt = f"""
    You are a document question-answering assistant.

    Answer the user's question using ONLY the information provided
    in the retrieved document context.

    Rules:
    1. Do not use outside knowledge.
    2. Do not invent or assume information.
    3. If the answer is not present in the context, say:
    "The requested information was not found in the provided documents."
    4. Keep the answer concise and factual.
    5. When using information from a source, include its source number
    like [1], [2], etc.
    6. Only cite sources that actually support the answer.

    Retrieved document context:

    {context}

    User question:

    {question}

    Return the response in this format:

    Answer:
    <answer with citations>

    Sources:
    [1] <filename> — Page <page_number>
    [2] <filename> — Page <page_number>
    """

        response = self.llm.invoke(prompt)

        return self.extract_text(
                response.content
            ).strip()

    def extract_text(self, content):
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)

                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))

            return "".join(text_parts)

        return str(content)