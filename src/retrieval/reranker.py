"""
Cross-encoder and LLM-as-judge reranking for RAG pipelines.

Two reranking strategies:
- Cross-encoder: sentence-transformers CrossEncoder scores (query, doc) pairs
- LLM-as-judge: Mistral 7B rates each document's relevance on a 1-10 scale

Typical usage: retrieve 20 candidates (fast), rerank to top-k (precise).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# CrossEncoder cache - avoids reloading the model on every call
# ---------------------------------------------------------------------------

class CrossEncoderCache:
    """Process-level singleton cache for CrossEncoder models.

    Loading a CrossEncoder from disk takes ~1-2 seconds. Caching it per
    model name means the overhead is paid once per process, not per query.
    """

    _instances: dict[str, object] = {}

    @classmethod
    def get(cls, model_name: str) -> object:
        """Return a cached CrossEncoder, loading it on first access."""
        if model_name not in cls._instances:
            from sentence_transformers import CrossEncoder  # type: ignore[import]
            cls._instances[model_name] = CrossEncoder(model_name)
        return cls._instances[model_name]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT_TEMPLATE = """\
You are a relevance judge. Given a question and a document excerpt, rate how
relevant the document is for answering the question.

Respond with ONLY a single integer from 1 to 10.

1 = completely irrelevant
5 = somewhat relevant, partially answers the question
10 = perfectly relevant, directly answers the question

Question: {question}

Document: {document}

Relevance score:"""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RerankerResult:
    """Output of a reranking run."""

    technique: str             # "cross_encoder" | "llm_judge" | "no_reranking"
    query: str
    docs: list[Document]
    scores: list[float]        # relevance scores (higher = more relevant)
    candidate_count: int       # number of candidates before reranking
    elapsed_ms: float
    llm_calls: int             # 0 for cross_encoder, N for llm_judge
    params: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.docs)

    def summary(self) -> dict:
        return {
            "technique": self.technique,
            "query": self.query[:80],
            "candidate_count": self.candidate_count,
            "num_results": self.num_results,
            "top_score": round(self.scores[0], 4) if self.scores else None,
            "llm_calls": self.llm_calls,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str,
    candidates: list[Document],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k: int = 5,
) -> RerankerResult:
    """Rerank candidates using a cross-encoder model.

    The cross-encoder scores each (query, document) pair jointly, producing
    richer relevance signals than bi-encoder cosine similarity.

    Args:
        query: User query string.
        candidates: Documents to rerank (typically 15-20 from initial retrieval).
        model_name: HuggingFace cross-encoder model identifier.
        top_k: Number of documents to return after reranking.

    Returns:
        RerankerResult with top_k reranked documents and their scores.
    """
    start = time.perf_counter()
    cross_encoder = CrossEncoderCache.get(model_name)

    pairs = [(query, doc.page_content) for doc in candidates]
    raw_scores: list[float] = cross_encoder.predict(pairs).tolist()

    # Sort by score descending
    scored = sorted(zip(candidates, raw_scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:top_k]]
    top_scores = [score for _, score in scored[:top_k]]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RerankerResult(
        technique="cross_encoder",
        query=query,
        docs=top_docs,
        scores=top_scores,
        candidate_count=len(candidates),
        elapsed_ms=elapsed_ms,
        llm_calls=0,
        params={"model_name": model_name, "top_k": top_k},
    )


# ---------------------------------------------------------------------------
# LLM-as-judge reranking
# ---------------------------------------------------------------------------

def rerank_llm_judge(
    query: str,
    candidates: list[Document],
    llm: BaseChatModel,
    top_k: int = 5,
) -> RerankerResult:
    """Rerank candidates using an LLM relevance judge.

    Each document is scored independently by the LLM on a 1-10 scale.
    Slow (one LLM call per document) but handles nuanced relevance.

    Args:
        query: User query string.
        candidates: Documents to rerank.
        llm: Chat model used as relevance judge.
        top_k: Number of documents to return after reranking.

    Returns:
        RerankerResult with top_k reranked documents and their scores.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        (
            "human",
            _JUDGE_PROMPT_TEMPLATE,
        ),
    ])
    chain = prompt | llm | StrOutputParser()

    start = time.perf_counter()
    scores: list[float] = []

    for doc in candidates:
        raw = chain.invoke({
            "question": query,
            "document": doc.page_content[:1500],
        }).strip()
        # Parse the integer score; default to 1 on parse failure
        try:
            score = float("".join(c for c in raw if c.isdigit() or c == "."))
            score = max(1.0, min(10.0, score))
        except (ValueError, TypeError):
            score = 1.0
        scores.append(score)

    # Sort by score descending
    scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in scored[:top_k]]
    top_scores = [score for _, score in scored[:top_k]]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RerankerResult(
        technique="llm_judge",
        query=query,
        docs=top_docs,
        scores=top_scores,
        candidate_count=len(candidates),
        elapsed_ms=elapsed_ms,
        llm_calls=len(candidates),
        params={"top_k": top_k},
    )


# ---------------------------------------------------------------------------
# No-reranking baseline
# ---------------------------------------------------------------------------

def retrieve_no_reranking(
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
) -> RerankerResult:
    """Baseline: top-k similarity search without reranking.

    Args:
        vectorstore: ChromaDB vectorstore to search.
        query: User query string.
        k: Number of results to return.

    Returns:
        RerankerResult with technique="no_reranking".
    """
    start = time.perf_counter()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    docs = [doc for doc, _ in docs_and_scores]
    scores = [float(score) for _, score in docs_and_scores]

    return RerankerResult(
        technique="no_reranking",
        query=query,
        docs=docs,
        scores=scores,
        candidate_count=k,
        elapsed_ms=elapsed_ms,
        llm_calls=0,
        params={"k": k},
    )


# ---------------------------------------------------------------------------
# Full pipeline: retrieve candidates then rerank
# ---------------------------------------------------------------------------

def retrieve_and_rerank(
    vectorstore: VectorStore,
    query: str,
    technique: str,
    candidate_k: int = 20,
    top_k: int = 5,
    llm: BaseChatModel | None = None,
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> RerankerResult:
    """Retrieve candidates with dense search, then rerank.

    Args:
        vectorstore: ChromaDB vectorstore for candidate retrieval.
        query: User query string.
        technique: "cross_encoder" or "llm_judge".
        candidate_k: Number of candidates to retrieve before reranking.
        top_k: Number of final results to return after reranking.
        llm: Required when technique="llm_judge".
        cross_encoder_model: Model name for technique="cross_encoder".

    Returns:
        RerankerResult with top_k reranked documents.
    """
    candidates = vectorstore.similarity_search(query, k=candidate_k)

    if technique == "cross_encoder":
        return rerank_cross_encoder(
            query=query,
            candidates=candidates,
            model_name=cross_encoder_model,
            top_k=top_k,
        )
    elif technique == "llm_judge":
        if llm is None:
            raise ValueError("llm is required for llm_judge technique")
        return rerank_llm_judge(
            query=query,
            candidates=candidates,
            llm=llm,
            top_k=top_k,
        )
    else:
        raise ValueError(f"Unknown technique: {technique!r}. Use 'cross_encoder' or 'llm_judge'.")
