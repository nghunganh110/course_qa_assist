"""
Query translation techniques for improved RAG retrieval.

Transforms the user's question before retrieval to improve document recall.
Five techniques are implemented:

- Multi-Query: generate N alternative phrasings, merge results
- RAG Fusion: multi-query + Reciprocal Rank Fusion (RRF) merging
- HyDE: generate a hypothetical answer, embed it instead of the query
- Step-Back: abstract the query, retrieve with both original and abstract
- Decomposition: break the query into sub-questions, retrieve independently
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Prompts (domain-agnostic — work with any corpus)
# ---------------------------------------------------------------------------

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Generate {n} alternative search queries for the following question.
Each query must be a rephrasing or related angle of the original question.
Output ONLY the queries, one per line, no numbering, no explanation.

Example input: How does knowledge distillation improve model compression?
Example output:
What techniques are used to transfer knowledge from large to small models?
How do teacher-student frameworks reduce model size?
Knowledge distillation methods for efficient inference

Question: {question}
Output:""",
    ),
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Write a short technical passage (3-5 sentences) that directly answers the following question.
Write as if you are a textbook or academic paper. Be specific and technical. Do not say you don't know.

Question: {question}
Passage:""",
    ),
])

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Rewrite the following specific question as a more general, abstract question that covers the broader topic.
Output ONLY the abstract question, no explanation.

Examples:
Specific: What loss function does SRA use for span alignment?
Abstract: How do span-based knowledge distillation methods define their training objectives?

Specific: How many memory layers does MemORAI use?
Abstract: How do memory-augmented LLM architectures organize and retrieve stored information?

Specific: What is the TokenRatio metric used for?
Abstract: How are token-level preference metrics designed for language model alignment?

Specific: {question}
Abstract:""",
    ),
])

DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Break the following question into {n} simpler sub-questions that together address the original.
Output ONLY the sub-questions, one per line, no numbering, no explanation.

Example input: How does cross-tokenizer knowledge distillation handle different vocabularies?
Example output:
What is cross-tokenizer knowledge distillation?
How are token representations aligned across different tokenizers?
What are the main challenges of vocabulary mismatch in distillation?

Question: {question}
Sub-questions:""",
    ),
])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryTranslationResult:
    """Output of a query translation + retrieval run."""

    technique: str
    original_query: str
    translated_queries: list[str]
    docs: list[Document]
    elapsed_ms: float
    llm_calls: int
    params: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.docs)

    def summary(self) -> dict:
        """Return a compact dict summary."""
        return {
            "technique": self.technique,
            "original_query": self.original_query[:80],
            "translated_queries": len(self.translated_queries),
            "num_results": self.num_results,
            "llm_calls": self.llm_calls,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_lines(text: str) -> list[str]:
    """Split LLM output by newlines, strip, drop empty lines."""
    return [line.strip() for line in text.strip().splitlines() if line.strip()]


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by (source, page_content[:100]) key."""
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (
            doc.metadata.get("source", ""),
            doc.page_content[:100],
        )
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def reciprocal_rank_fusion(
    results: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion.

    Documents appearing in multiple lists get a higher fused score.
    The parameter k=60 controls rank position sensitivity.

    Args:
        results: List of ranked document lists (one per query variant).
        k: RRF smoothing parameter (default 60).

    Returns:
        Deduplicated and re-ranked list of documents.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in results:
        for rank, doc in enumerate(ranked_list):
            key = doc.metadata.get("source", "") + "|" + doc.page_content[:100]
            if key not in scores:
                scores[key] = 0.0
                doc_map[key] = doc
            scores[key] += 1.0 / (rank + k)

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# ---------------------------------------------------------------------------
# Translation functions (LLM calls only)
# ---------------------------------------------------------------------------

def translate_multi_query(
    llm: BaseChatModel,
    query: str,
    n: int = 4,
) -> list[str]:
    """Generate N alternative phrasings of the query.

    Args:
        llm: Chat model to use for generation.
        query: Original user query.
        n: Number of alternative queries to generate.

    Returns:
        List of alternative query strings (may include original).
    """
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
    text = chain.invoke({"question": query, "n": n})
    variants = _parse_lines(text)
    return variants[:n] if variants else [query]


def translate_step_back(
    llm: BaseChatModel,
    query: str,
) -> str:
    """Generate a more abstract step-back version of the query.

    Args:
        llm: Chat model to use for generation.
        query: Original user query.

    Returns:
        The step-back (abstracted) query string.
    """
    chain = STEP_BACK_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query}).strip()


def translate_decompose(
    llm: BaseChatModel,
    query: str,
    n: int = 3,
) -> list[str]:
    """Decompose a complex query into simpler sub-questions.

    Args:
        llm: Chat model to use for generation.
        query: Original user query.
        n: Number of sub-questions to generate.

    Returns:
        List of sub-question strings.
    """
    chain = DECOMPOSITION_PROMPT | llm | StrOutputParser()
    text = chain.invoke({"question": query, "n": n})
    sub_questions = _parse_lines(text)
    return sub_questions[:n] if sub_questions else [query]


def generate_hyde_doc(
    llm: BaseChatModel,
    query: str,
) -> str:
    """Generate a hypothetical document that would answer the query.

    Args:
        llm: Chat model to use for generation.
        query: Original user query.

    Returns:
        A hypothetical answer passage (to be embedded instead of the query).
    """
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": query}).strip()


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def retrieve_multi_query(
    llm: BaseChatModel,
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
    n_queries: int = 4,
) -> QueryTranslationResult:
    """Multi-Query retrieval: merge results from N query variants.

    Args:
        llm: Chat model for query generation.
        vectorstore: Vector store for similarity search.
        query: Original user query.
        k: Number of final results to return.
        n_queries: Number of alternative queries to generate.

    Returns:
        QueryTranslationResult with merged, deduplicated documents.
    """
    start = time.perf_counter()

    variants = translate_multi_query(llm, query, n=n_queries)
    llm_calls = 1

    all_docs: list[Document] = []
    for variant in variants:
        docs = vectorstore.similarity_search(variant, k=k)
        all_docs.extend(docs)

    deduped = _deduplicate(all_docs)[:k]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryTranslationResult(
        technique="multi_query",
        original_query=query,
        translated_queries=variants,
        docs=deduped,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k, "n_queries": n_queries},
    )


def retrieve_rag_fusion(
    llm: BaseChatModel,
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
    n_queries: int = 4,
    rrf_k: int = 60,
) -> QueryTranslationResult:
    """RAG Fusion: multi-query with Reciprocal Rank Fusion merging.

    Args:
        llm: Chat model for query generation.
        vectorstore: Vector store for similarity search.
        query: Original user query.
        k: Number of final results to return.
        n_queries: Number of alternative queries to generate.
        rrf_k: RRF smoothing parameter.

    Returns:
        QueryTranslationResult with RRF-ranked documents.
    """
    start = time.perf_counter()

    variants = translate_multi_query(llm, query, n=n_queries)
    llm_calls = 1

    ranked_lists: list[list[Document]] = []
    for variant in variants:
        docs = vectorstore.similarity_search(variant, k=k)
        ranked_lists.append(docs)

    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)[:k]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryTranslationResult(
        technique="rag_fusion",
        original_query=query,
        translated_queries=variants,
        docs=fused,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k, "n_queries": n_queries, "rrf_k": rrf_k},
    )


def retrieve_hyde(
    llm: BaseChatModel,
    embeddings: Embeddings,
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
) -> QueryTranslationResult:
    """HyDE retrieval: embed a hypothetical answer instead of the query.

    Args:
        llm: Chat model for hypothetical document generation.
        embeddings: Embedding model (same as the vectorstore's).
        vectorstore: Vector store for vector similarity search.
        query: Original user query.
        k: Number of results to return.

    Returns:
        QueryTranslationResult with documents retrieved via HyDE embedding.
    """
    start = time.perf_counter()

    hyde_text = generate_hyde_doc(llm, query)
    llm_calls = 1

    hyde_embedding = embeddings.embed_query(hyde_text)
    docs = vectorstore.similarity_search_by_vector(hyde_embedding, k=k)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryTranslationResult(
        technique="hyde",
        original_query=query,
        translated_queries=[hyde_text],
        docs=docs,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k},
    )


def retrieve_step_back(
    llm: BaseChatModel,
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
) -> QueryTranslationResult:
    """Step-Back retrieval: combine results from original and abstract query.

    Args:
        llm: Chat model for step-back query generation.
        vectorstore: Vector store for similarity search.
        query: Original user query.
        k: Number of final results to return.

    Returns:
        QueryTranslationResult with merged documents from both queries.
    """
    start = time.perf_counter()

    abstract_query = translate_step_back(llm, query)
    llm_calls = 1

    # Retrieve for both the original and the abstract query
    original_docs = vectorstore.similarity_search(query, k=k)
    abstract_docs = vectorstore.similarity_search(abstract_query, k=k)

    merged = _deduplicate(original_docs + abstract_docs)[:k]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryTranslationResult(
        technique="step_back",
        original_query=query,
        translated_queries=[abstract_query],
        docs=merged,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k},
    )


def retrieve_decomposition(
    llm: BaseChatModel,
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
    n_sub: int = 3,
) -> QueryTranslationResult:
    """Decomposition retrieval: split query into sub-questions, merge results.

    Args:
        llm: Chat model for query decomposition.
        vectorstore: Vector store for similarity search.
        query: Original user query.
        k: Number of final results to return.
        n_sub: Number of sub-questions to generate.

    Returns:
        QueryTranslationResult with merged documents from all sub-questions.
    """
    start = time.perf_counter()

    sub_questions = translate_decompose(llm, query, n=n_sub)
    llm_calls = 1

    all_docs: list[Document] = []
    for sub_q in sub_questions:
        docs = vectorstore.similarity_search(sub_q, k=k)
        all_docs.extend(docs)

    deduped = _deduplicate(all_docs)[:k]
    elapsed_ms = (time.perf_counter() - start) * 1000

    return QueryTranslationResult(
        technique="decomposition",
        original_query=query,
        translated_queries=sub_questions,
        docs=deduped,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k, "n_sub": n_sub},
    )
