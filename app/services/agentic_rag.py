from typing import TypedDict

from fastapi import HTTPException
from langchain_core.documents import Document
from langchain_google_genai.chat_models import GoogleRateLimitError
from langgraph.graph import StateGraph, START, END

from app.services.langchain_rag import LangChainRAGService


class AgenticRAGState(TypedDict):
    original_question: str
    search_question: str
    route: str
    document_id: int | None
    sources: list[dict]

    documents: list[Document]
    relevant_documents: list[Document]

    context: str
    answer: str

    rewritten_question: str
    retry_count: int


class AgenticRAGService:

    def __init__(self):

        self.rag_service = LangChainRAGService()

        self.graph = self._build_graph()

    # =========================================================
    # GRAPH
    # =========================================================

    def _build_graph(self):

        graph = StateGraph(AgenticRAGState)

        graph.add_node(
            "route_question",
            self.route_question,
        )

        graph.add_node(
            "retrieve",
            self.retrieve,
        )

        graph.add_node(
            "grade_documents",
            self.grade_documents,
        )

        graph.add_node(
            "rewrite_query",
            self.rewrite_query,
        )

        graph.add_node(
            "generate",
            self.generate,
        )

        graph.add_node(
            "general_answer",
            self.general_answer,
        )

        # START → Router
        graph.add_edge(
            START,
            "route_question",
        )

        # Router decision
        graph.add_conditional_edges(
            "route_question",
            self.route_after_router,
            {
                "document_rag": "retrieve",
                "general": "general_answer",
            },
        )

        # Document RAG
        graph.add_edge(
            "retrieve",
            "grade_documents",
        )

        graph.add_conditional_edges(
            "grade_documents",
            self.route_after_grading,
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query",
            },
        )

        graph.add_edge(
            "rewrite_query",
            "retrieve",
        )

        # End
        graph.add_edge(
            "generate",
            END,
        )

        graph.add_edge(
            "general_answer",
            END,
        )

        return graph.compile()

    # =========================================================
    # SAFE LLM CALL
    # =========================================================

    def invoke_llm(self, prompt: str):

        try:

            return self.rag_service.llm.invoke(prompt)

        except GoogleRateLimitError:

            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. "
                    "Please wait for the quota reset "
                    "or use a Gemini API project with "
                    "available quota."
                ),
            )

    # =========================================================
    # ROUTER
    # =========================================================

    def route_question(
        self,
        state: AgenticRAGState,
    ):

        question = state["original_question"]

        prompt = f"""
You are a query router.

Decide whether the user's question should be answered
using the company's uploaded documents or using general
knowledge.

Choose exactly one route:

DOCUMENT
GENERAL

Use DOCUMENT when the question asks about information
that could exist inside the uploaded company documents,
such as:

- company policies
- leave policy
- working hours
- employee benefits
- company rules
- internal procedures
- information from uploaded documents

Use GENERAL when the question asks about general knowledge
that does not require the uploaded documents.

Question:
{question}

Return ONLY:

DOCUMENT

or:

GENERAL
"""

        response = self.invoke_llm(prompt)

        result = self.rag_service.extract_text(
            response.content
        ).strip().upper()

        if result == "DOCUMENT":

            route = "document"

        else:

            route = "general"

        return {
            "route": route
        }

    # =========================================================
    # ROUTER CONDITION
    # =========================================================

    def route_after_router(
        self,
        state: AgenticRAGState,
    ):

        if state["route"] == "document":

            return "document_rag"

        return "general"

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        state: AgenticRAGState,
    ):

        question = state["search_question"]

        documents = self.rag_service.retrieve(
            question=question,
            top_k=5,
            document_id=state.get("document_id"),
        )

        return {
            "documents": documents,
        }

    # =========================================================
    # GRADE ALL DOCUMENTS
    # =========================================================

    def grade_documents(
        self,
        state: AgenticRAGState,
    ):

        question = state["search_question"]

        documents = state["documents"]

        if not documents:

            return {
                "relevant_documents": []
            }

        documents_text = ""

        for index, document in enumerate(
            documents,
            start=1,
        ):

            documents_text += f"""

DOCUMENT {index}

{document.page_content}

"""

        prompt = f"""
You are a document relevance grader.

Determine which retrieved documents contain
information relevant to answering the user's question.

Question:

{question}

Retrieved documents:

{documents_text}

Rules:

1. Analyze every document.
2. Return the numbers of relevant documents.
3. A document is relevant if it contains useful
   information for answering the question.
4. Do not select unrelated documents.
5. Return ONLY the document numbers.
6. Separate numbers with commas.
7. If none are relevant, return NONE.

Examples:

1,3

or:

2,4,5

or:

NONE

Answer:
"""

        response = self.invoke_llm(prompt)

        result = self.rag_service.extract_text(
            response.content
        ).strip().upper()

        relevant_documents = []

        if result != "NONE":

            try:

                indexes = [
                    int(number.strip())
                    for number in result.split(",")
                    if number.strip().isdigit()
                ]

                for index in indexes:

                    if 1 <= index <= len(documents):

                        relevant_documents.append(
                            documents[index - 1]
                        )

            except ValueError:

                relevant_documents = []

        return {
            "relevant_documents": relevant_documents
        }

    # =========================================================
    # GRADING ROUTER
    # =========================================================

    def route_after_grading(
        self,
        state: AgenticRAGState,
    ):

        relevant_documents = state[
            "relevant_documents"
        ]

        if relevant_documents:

            return "generate"

        if state.get(
            "retry_count",
            0,
        ) >= 1:

            return "generate"

        return "rewrite_query"

    # =========================================================
    # QUERY REWRITE
    # =========================================================

    def rewrite_query(
        self,
        state: AgenticRAGState,
    ):

        question = state["original_question"]

        prompt = f"""
Rewrite the following question into a clearer
search query for retrieving information from
company documents.

Original question:

{question}

Rules:

1. Preserve the original meaning.
2. Make the query specific.
3. Do not answer the question.
4. Return ONLY the rewritten search query.

Rewritten query:
"""

        response = self.invoke_llm(prompt)

        rewritten_question = (
            self.rag_service.extract_text(
                response.content
            )
            .strip()
        )

        retry_count = (
            state.get(
                "retry_count",
                0,
            )
            + 1
        )

        return {
            "search_question": rewritten_question,
            "rewritten_question": rewritten_question,
            "retry_count": retry_count,
        }

    # =========================================================
    # GENERATE
    # =========================================================

    def generate(
        self,
        state: AgenticRAGState,
    ):

        documents = state[
            "relevant_documents"
        ]

        context = self.rag_service.build_context(
            documents
        )

        answer = self.rag_service.generate(
            question=state["original_question"],
            context=context,
        )

        return {
            "context": context,
            "answer": answer,
            "sources": self.rag_service.build_sources(documents),
        }

    # =========================================================
    # GENERAL ANSWER
    # =========================================================

    def general_answer(
        self,
        state: AgenticRAGState,
    ):

        question = state[
            "original_question"
        ]

        prompt = f"""
Answer the following general knowledge question.

Question:

{question}

Provide a concise and accurate answer.
"""

        response = self.invoke_llm(prompt)

        answer = self.rag_service.extract_text(
            response.content
        ).strip()

        return {
            "answer": answer
        }