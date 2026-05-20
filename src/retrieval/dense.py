"""
Dense retrieval strategies using vector similarity search.

Provides similarity search and MMR (Maximum Marginal Relevance) retrieval
from a ChromaDB vector store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore


@dataclass
class RetrievalStrategyResult:
    """Output of a retrieval strategy run."""

    strategy: str
    query: str
    docs: list[Document]
    scores: list[float]
    elapsed_ms: float
    params: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.docs)

    def summary(self) -> dict:
        """Return a compact dict summary."""
        return {
            "strategy": self.strategy,
            "query": self.query[:80],
            "num_results": self.num_results,
            "top_score": round(self.scores[0], 4) if self.scores else None,
            "avg_score": (
                round(sum(self.scores) / len(self.scores), 4)
                if self.scores
                else None
            ),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


def create_dense_retriever(
    vectorstore: VectorStore,
    search_type: str = "similarity",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
) -> BaseRetriever:
    """Create a dense retriever from a vector store.

    Args:
        vectorstore: ChromaDB vector store instance.
        search_type: "similarity" or "mmr".
        k: Number of results to return.
        fetch_k: Number of candidates for MMR (ignored for similarity).
        lambda_mult: MMR diversity parameter (0=max diversity, 1=max relevance).

    Returns:
        A LangChain BaseRetriever instance.
    """
    search_kwargs: dict = {"k": k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )


def retrieve_dense(
    vectorstore: VectorStore,
    query: str,
    search_type: str = "similarity",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.7,
) -> RetrievalStrategyResult:
    """Run a dense retrieval query with timing and scores.

    Args:
        vectorstore: ChromaDB vector store instance.
        query: The search query.
        search_type: "similarity" or "mmr".
        k: Number of results to return.
        fetch_k: Number of candidates for MMR.
        lambda_mult: MMR diversity parameter.

    Returns:
        RetrievalStrategyResult with documents, scores, and timing.
    """
    start = time.perf_counter()

    if search_type == "similarity":
        results = vectorstore.similarity_search_with_score(query, k=k)
        docs = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
    elif search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        # Get scores separately (MMR does not return scores directly)
        scored = vectorstore.similarity_search_with_score(query, k=fetch_k)
        score_map = {doc.page_content[:200]: float(s) for doc, s in scored}
        scores = [score_map.get(d.page_content[:200], 0.0) for d in docs]
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    elapsed_ms = (time.perf_counter() - start) * 1000

    strategy_name = (
        "similarity" if search_type == "similarity"
        else f"mmr(lambda={lambda_mult})"
    )

    return RetrievalStrategyResult(
        strategy=strategy_name,
        query=query,
        docs=docs,
        scores=scores,
        elapsed_ms=elapsed_ms,
        params={
            "search_type": search_type,
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )
