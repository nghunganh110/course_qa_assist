"""
Sparse retrieval using BM25 keyword matching.

Wraps langchain_community's BM25Retriever with timing and scoring.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.retrieval.dense import RetrievalStrategyResult


def create_bm25_retriever(
    chunks: list[Document],
    k: int = 5,
    preprocess_func: Callable | None = None,
) -> BM25Retriever:
    """Create a BM25 retriever from document chunks.

    Args:
        chunks: List of Document chunks to index.
        k: Number of results to return.
        preprocess_func: Optional text preprocessing function.

    Returns:
        A BM25Retriever instance.
    """
    kwargs: dict = {"k": k}
    if preprocess_func is not None:
        kwargs["preprocess_func"] = preprocess_func
    return BM25Retriever.from_documents(chunks, **kwargs)


def retrieve_bm25(
    retriever: BM25Retriever,
    query: str,
) -> RetrievalStrategyResult:
    """Run a BM25 retrieval query with timing.

    Note: BM25Retriever does not expose per-document scores via the
    LangChain interface, so scores are set to 0.0.

    Args:
        retriever: A pre-built BM25Retriever instance.
        query: The search query.

    Returns:
        RetrievalStrategyResult with documents and timing.
    """
    start = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return RetrievalStrategyResult(
        strategy="bm25",
        query=query,
        docs=docs,
        scores=[0.0] * len(docs),
        elapsed_ms=elapsed_ms,
        params={"k": retriever.k},
    )
