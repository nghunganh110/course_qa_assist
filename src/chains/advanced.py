"""
Advanced RAG patterns using LangGraph.

Three architectures:
- CRAG (Corrective RAG): grade retrieved docs, rewrite query if irrelevant
- Self-RAG: LLM decides whether to retrieve, checks groundedness and usefulness
- Adaptive RAG: route query to simple/standard/complex pipeline by complexity
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, Annotated, Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------

_GRADE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no'.

Retrieved document:
{document}

User question: {question}

Relevant (yes/no):""",
    ),
])

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Rewrite the following question to improve retrieval results.
Make it more specific and focused. Output ONLY the rewritten question, nothing else.

Original question: {question}
Rewritten question:""",
    ),
])

_GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer:""",
    ),
])

_GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Is the following answer grounded in (supported by) the provided documents?
Answer 'yes' if the answer is supported by the documents, 'no' if it contains
information not found in the documents.

Documents:
{documents}

Answer: {generation}

Grounded (yes/no):""",
    ),
])

_USEFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Does the following answer address the user's question?
Answer 'yes' if it is useful, 'no' if it is off-topic or unhelpful.

Question: {question}
Answer: {generation}

Useful (yes/no):""",
    ),
])

_RETRIEVAL_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Do you need external documents to answer this question,
or can you answer from general knowledge?
Answer 'yes' if retrieval is needed, 'no' if general knowledge suffices.

Question: {question}

Need retrieval (yes/no):""",
    ),
])

_COMPLEXITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Classify the following question into one of three complexity levels:

- "simple": A direct factual question with a single, clear answer.
  Example: "What is the default model for ChatOllama?"

- "moderate": A question requiring understanding of a concept or process.
  Example: "How does MMR retrieval work in LangChain?"

- "complex": A multi-part question, comparison, or question requiring synthesis.
  Example: "Compare different text splitting strategies and when to use each."

Respond with ONLY one word: simple, moderate, or complex.

Question: {question}""",
    ),
])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _yn(text: str) -> bool:
    """Parse a yes/no response - returns True for yes."""
    return text.strip().lower().startswith("y")


def _build_context(docs: list[Document], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for i, doc in enumerate(docs, 1):
        snippet = f"[{i}] {doc.page_content[:1000]}"
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CRAG - Corrective RAG
# ---------------------------------------------------------------------------

class CRAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    grades: list[str]          # "yes" | "no" per doc
    rewrite_count: int


def build_crag(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    k: int = 5,
    relevance_threshold: float = 0.5,
    max_rewrites: int = 2,
) -> object:
    """Build a CRAG (Corrective RAG) LangGraph app.

    Flow: retrieve -> grade docs -> if enough relevant: generate
                                 -> else: rewrite query -> retrieve again

    Args:
        vectorstore: Vector store for retrieval.
        llm: Chat model for grading, rewriting, and generation.
        k: Number of documents to retrieve.
        relevance_threshold: Fraction of docs that must be relevant to proceed.
        max_rewrites: Maximum query rewrites before forcing generation.

    Returns:
        Compiled LangGraph app. Call with {"question": "..."}.
    """
    grade_chain = _GRADE_PROMPT | llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()
    generate_chain = _GENERATE_PROMPT | llm | StrOutputParser()

    def retrieve(state: CRAGState) -> CRAGState:
        docs = vectorstore.similarity_search(state["question"], k=k)
        return {**state, "documents": docs}

    def grade_documents(state: CRAGState) -> CRAGState:
        grades = []
        for doc in state["documents"]:
            raw = grade_chain.invoke({
                "document": doc.page_content[:1500],
                "question": state["question"],
            })
            grades.append("yes" if _yn(raw) else "no")
        return {**state, "grades": grades}

    def decide_after_grading(state: CRAGState) -> Literal["generate", "rewrite_query"]:
        grades = state.get("grades", [])
        if not grades:
            return "rewrite_query"
        relevant_fraction = grades.count("yes") / len(grades)
        if relevant_fraction >= relevance_threshold:
            return "generate"
        if state.get("rewrite_count", 0) >= max_rewrites:
            return "generate"  # force generation after max retries
        return "rewrite_query"

    def rewrite_query(state: CRAGState) -> CRAGState:
        new_q = rewrite_chain.invoke({"question": state["question"]}).strip()
        return {
            **state,
            "question": new_q,
            "rewrite_count": state.get("rewrite_count", 0) + 1,
        }

    def generate(state: CRAGState) -> CRAGState:
        context = _build_context(state["documents"])
        answer = generate_chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {**state, "generation": answer}

    graph = StateGraph(CRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate", generate)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Self-RAG
# ---------------------------------------------------------------------------

class SelfRAGState(TypedDict):
    question: str
    documents: list[Document]
    generation: str
    needs_retrieval: bool
    is_grounded: bool
    is_useful: bool
    retry_count: int


def build_self_rag(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    k: int = 5,
    max_retries: int = 2,
) -> object:
    """Build a Self-RAG LangGraph app.

    Flow: decide retrieval needed? -> (retrieve ->) generate
          -> check grounded? -> if not: retry with rewritten query
          -> check useful?   -> if not: retry

    Args:
        vectorstore: Vector store for retrieval.
        llm: Chat model for all self-reflection steps.
        k: Number of documents to retrieve.
        max_retries: Maximum generation retries before returning best answer.

    Returns:
        Compiled LangGraph app. Call with {"question": "..."}.
    """
    retrieval_chain = _RETRIEVAL_DECISION_PROMPT | llm | StrOutputParser()
    generate_chain = _GENERATE_PROMPT | llm | StrOutputParser()
    groundedness_chain = _GROUNDEDNESS_PROMPT | llm | StrOutputParser()
    usefulness_chain = _USEFULNESS_PROMPT | llm | StrOutputParser()
    rewrite_chain = _REWRITE_PROMPT | llm | StrOutputParser()

    def decide_retrieval(state: SelfRAGState) -> SelfRAGState:
        raw = retrieval_chain.invoke({"question": state["question"]})
        return {**state, "needs_retrieval": _yn(raw)}

    def route_retrieval(state: SelfRAGState) -> Literal["retrieve", "generate"]:
        return "retrieve" if state["needs_retrieval"] else "generate"

    def retrieve(state: SelfRAGState) -> SelfRAGState:
        docs = vectorstore.similarity_search(state["question"], k=k)
        return {**state, "documents": docs}

    def generate(state: SelfRAGState) -> SelfRAGState:
        docs = state.get("documents", [])
        context = _build_context(docs) if docs else "No external documents retrieved."
        answer = generate_chain.invoke({
            "context": context,
            "question": state["question"],
        })
        return {**state, "generation": answer}

    def check_groundedness(state: SelfRAGState) -> SelfRAGState:
        docs = state.get("documents", [])
        if not docs:
            return {**state, "is_grounded": True}  # no retrieval = no groundedness check
        doc_texts = "\n\n".join(d.page_content[:500] for d in docs[:3])
        raw = groundedness_chain.invoke({
            "documents": doc_texts,
            "generation": state["generation"],
        })
        return {**state, "is_grounded": _yn(raw)}

    def route_groundedness(state: SelfRAGState) -> Literal["check_useful", "rewrite_and_retry"]:
        if state["is_grounded"]:
            return "check_useful"
        if state.get("retry_count", 0) >= max_retries:
            return "check_useful"  # accept after max retries
        return "rewrite_and_retry"

    def check_useful(state: SelfRAGState) -> SelfRAGState:
        raw = usefulness_chain.invoke({
            "question": state["question"],
            "generation": state["generation"],
        })
        return {**state, "is_useful": _yn(raw)}

    def route_useful(state: SelfRAGState) -> Literal[END, "rewrite_and_retry"]:
        if state["is_useful"]:
            return END
        if state.get("retry_count", 0) >= max_retries:
            return END
        return "rewrite_and_retry"

    def rewrite_and_retry(state: SelfRAGState) -> SelfRAGState:
        new_q = rewrite_chain.invoke({"question": state["question"]}).strip()
        return {
            **state,
            "question": new_q,
            "retry_count": state.get("retry_count", 0) + 1,
            "documents": [],
            "generation": "",
        }

    graph = StateGraph(SelfRAGState)
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check_groundedness", check_groundedness)
    graph.add_node("check_useful", check_useful)
    graph.add_node("rewrite_and_retry", rewrite_and_retry)

    graph.set_entry_point("decide_retrieval")
    graph.add_conditional_edges("decide_retrieval", route_retrieval, {
        "retrieve": "retrieve",
        "generate": "generate",
    })
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_groundedness")
    graph.add_conditional_edges("check_groundedness", route_groundedness, {
        "check_useful": "check_useful",
        "rewrite_and_retry": "rewrite_and_retry",
    })
    graph.add_conditional_edges("check_useful", route_useful, {
        END: END,
        "rewrite_and_retry": "rewrite_and_retry",
    })
    graph.add_edge("rewrite_and_retry", "retrieve")

    return graph.compile()


# ---------------------------------------------------------------------------
# Adaptive RAG
# ---------------------------------------------------------------------------

class AdaptiveRAGState(TypedDict):
    question: str
    complexity: str            # "simple" | "moderate" | "complex"
    documents: list[Document]
    generation: str
    sub_questions: list[str]   # for complex queries
    sub_answers_text: str      # concatenated sub-question answers (complex path)


def build_adaptive_rag(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    k_simple: int = 3,
    k_moderate: int = 5,
    k_complex: int = 5,
) -> object:
    """Build an Adaptive RAG LangGraph app.

    Routes queries to different pipelines by complexity:
    - simple: fast single-shot retrieval (k=3)
    - moderate: standard retrieval (k=5)
    - complex: query decomposition + iterative retrieval + synthesis

    Args:
        vectorstore: Vector store for retrieval.
        llm: Chat model for classification, decomposition, and generation.
        k_simple: Docs to retrieve for simple queries.
        k_moderate: Docs to retrieve for moderate queries.
        k_complex: Docs per sub-question for complex queries.

    Returns:
        Compiled LangGraph app. Call with {"question": "..."}.
    """
    complexity_chain = _COMPLEXITY_PROMPT | llm | StrOutputParser()
    generate_chain = _GENERATE_PROMPT | llm | StrOutputParser()

    decompose_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """Break down the following complex question into 2-3 simpler sub-questions
that can each be answered independently. Output ONLY the sub-questions, one per line.

Question: {question}
Sub-questions:""",
        ),
    ])
    decompose_chain = decompose_prompt | llm | StrOutputParser()

    synthesize_prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            """Synthesize a comprehensive answer to the original question
using the answers to the sub-questions below.

Original question: {question}

Sub-question answers:
{sub_answers}

Comprehensive answer:""",
        ),
    ])
    synthesize_chain = synthesize_prompt | llm | StrOutputParser()

    def classify_complexity(state: AdaptiveRAGState) -> AdaptiveRAGState:
        raw = complexity_chain.invoke({"question": state["question"]}).strip().lower()
        # Normalize to valid complexity level
        if "complex" in raw:
            complexity = "complex"
        elif "moderate" in raw:
            complexity = "moderate"
        else:
            complexity = "simple"
        return {**state, "complexity": complexity}

    def route_by_complexity(
        state: AdaptiveRAGState,
    ) -> Literal["retrieve_simple", "retrieve_moderate", "decompose"]:
        c = state.get("complexity", "moderate")
        if c == "simple":
            return "retrieve_simple"
        if c == "complex":
            return "decompose"
        return "retrieve_moderate"

    def retrieve_simple(state: AdaptiveRAGState) -> AdaptiveRAGState:
        docs = vectorstore.similarity_search(state["question"], k=k_simple)
        return {**state, "documents": docs}

    def retrieve_moderate(state: AdaptiveRAGState) -> AdaptiveRAGState:
        docs = vectorstore.similarity_search(state["question"], k=k_moderate)
        return {**state, "documents": docs}

    def decompose(state: AdaptiveRAGState) -> AdaptiveRAGState:
        raw = decompose_chain.invoke({"question": state["question"]})
        sub_qs = [
            line.strip().lstrip("-•123456789. ")
            for line in raw.strip().splitlines()
            if line.strip()
        ][:3]
        return {**state, "sub_questions": sub_qs, "documents": []}

    def retrieve_and_answer_sub(state: AdaptiveRAGState) -> AdaptiveRAGState:
        """Retrieve for each sub-question and collect all docs + partial answers."""
        sub_qs = state.get("sub_questions", [])
        all_docs: list[Document] = []
        sub_answers: list[str] = []
        seen_sources: set[str] = set()

        for sub_q in sub_qs:
            docs = vectorstore.similarity_search(sub_q, k=k_complex)
            # Deduplicate by source
            for doc in docs:
                src = doc.metadata.get("source", "") + doc.page_content[:50]
                if src not in seen_sources:
                    all_docs.append(doc)
                    seen_sources.add(src)
            # Generate partial answer
            context = _build_context(docs)
            answer = generate_chain.invoke({"context": context, "question": sub_q})
            sub_answers.append(f"Q: {sub_q}\nA: {answer}")

        return {**state, "documents": all_docs, "sub_answers_text": "\n\n".join(sub_answers)}

    def generate_simple(state: AdaptiveRAGState) -> AdaptiveRAGState:
        context = _build_context(state["documents"])
        answer = generate_chain.invoke({"context": context, "question": state["question"]})
        return {**state, "generation": answer}

    def generate_moderate(state: AdaptiveRAGState) -> AdaptiveRAGState:
        context = _build_context(state["documents"])
        answer = generate_chain.invoke({"context": context, "question": state["question"]})
        return {**state, "generation": answer}

    def synthesize(state: AdaptiveRAGState) -> AdaptiveRAGState:
        sub_answers = state.get("sub_answers_text", "")
        answer = synthesize_chain.invoke({
            "question": state["question"],
            "sub_answers": sub_answers,
        })
        return {**state, "generation": answer}

    graph = StateGraph(AdaptiveRAGState)
    graph.add_node("classify_complexity", classify_complexity)
    graph.add_node("retrieve_simple", retrieve_simple)
    graph.add_node("retrieve_moderate", retrieve_moderate)
    graph.add_node("decompose", decompose)
    graph.add_node("retrieve_and_answer_sub", retrieve_and_answer_sub)
    graph.add_node("generate_simple", generate_simple)
    graph.add_node("generate_moderate", generate_moderate)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("classify_complexity")
    graph.add_conditional_edges("classify_complexity", route_by_complexity, {
        "retrieve_simple": "retrieve_simple",
        "retrieve_moderate": "retrieve_moderate",
        "decompose": "decompose",
    })
    graph.add_edge("retrieve_simple", "generate_simple")
    graph.add_edge("retrieve_moderate", "generate_moderate")
    graph.add_edge("decompose", "retrieve_and_answer_sub")
    graph.add_edge("retrieve_and_answer_sub", "synthesize")
    graph.add_edge("generate_simple", END)
    graph.add_edge("generate_moderate", END)
    graph.add_edge("synthesize", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: run a graph and time it
# ---------------------------------------------------------------------------

def run_graph(app, question: str) -> dict:
    """Run any compiled RAG graph and return the final state with elapsed_ms.

    Only the ``question`` key is passed as initial state. Counter fields like
    ``retry_count`` and ``rewrite_count`` are read with ``.get(..., 0)`` inside
    each graph node, so they default to 0 without needing to be injected here.
    Injecting extra keys not declared in a graph's TypedDict is an error in
    strict LangGraph configurations.
    """
    start = time.perf_counter()
    result = app.invoke({"question": question})
    elapsed_ms = (time.perf_counter() - start) * 1000
    result["elapsed_ms"] = elapsed_ms
    return result
