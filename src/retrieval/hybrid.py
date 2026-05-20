"""
Hybrid retrieval combining dense and sparse strategies.

Uses EnsembleRetriever with configurable weights for
Reciprocal Rank Fusion (RRF) merging.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

from src.retrieval.dense import RetrievalStrategyResult


def create_hybrid_retriever(
    dense_retriever: BaseRetriever,
    sparse_retriever: BaseRetriever,
    weights: list[float] | None = None,
) -> EnsembleRetriever:
    """Create a hybrid retriever combining dense and sparse.

    Args:
        dense_retriever: Dense (vector similarity) retriever.
        sparse_retriever: Sparse (BM25) retriever.
        weights: Weights for [dense, sparse]. Defaults to [0.5, 0.5].

    Returns:
        An EnsembleRetriever instance.
    """
    if weights is None:
        weights = [0.5, 0.5]

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=weights,
    )


def retrieve_hybrid(
    retriever: EnsembleRetriever,
    query: str,
    weights_label: str = "0.5/0.5",
) -> RetrievalStrategyResult:
    """Run a hybrid retrieval query with timing.

    Note: EnsembleRetriever uses RRF internally and does not expose
    per-document scores, so scores are set to 0.0.

    Args:
        retriever: A pre-built EnsembleRetriever instance.
        query: The search query.
        weights_label: Human-readable label for the weight config.

    Returns:
        RetrievalStrategyResult with documents and timing.
    """
    start = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return RetrievalStrategyResult(
        strategy=f"hybrid({weights_label})",
        query=query,
        docs=docs,
        scores=[0.0] * len(docs),
        elapsed_ms=elapsed_ms,
        params={"weights": weights_label},
    )
