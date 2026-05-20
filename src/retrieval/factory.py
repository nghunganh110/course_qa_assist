"""
Retriever factory - creates retrievers from YAML configuration.

Reads the 'retrieval' section of configs/default.yaml to instantiate
the appropriate retriever strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore

from src.retrieval.dense import create_dense_retriever
from src.retrieval.hybrid import create_hybrid_retriever
from src.retrieval.sparse import create_bm25_retriever

_STRATEGY_REGISTRY: dict[str, str] = {
    "similarity": "dense",
    "dense": "dense",
    "mmr": "dense",
    "bm25": "sparse",
    "sparse": "sparse",
    "hybrid": "hybrid",
    "ensemble": "hybrid",
}


def create_retriever_from_config(
    config: dict,
    vectorstore: VectorStore,
    chunks: list[Document] | None = None,
) -> BaseRetriever:
    """Create a retriever from a configuration dictionary.

    Args:
        config: The 'retrieval' section of the pipeline config.
        vectorstore: ChromaDB vector store (for dense retrieval).
        chunks: Document chunks (required for BM25/hybrid strategies).

    Returns:
        A configured BaseRetriever instance.

    Raises:
        ValueError: If strategy is unknown or chunks missing for sparse.
    """
    strategy = config.get("strategy", "similarity")
    resolved = _STRATEGY_REGISTRY.get(strategy)
    if resolved is None:
        available = ", ".join(sorted(_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown retrieval strategy: {strategy}. Available: {available}"
        )

    final_k = config.get("final_k", 5)

    if resolved == "dense":
        dense_cfg = config.get("dense", {})
        search_type = (
            "mmr" if strategy == "mmr"
            else dense_cfg.get("search_type", "similarity")
        )
        return create_dense_retriever(
            vectorstore=vectorstore,
            search_type=search_type,
            k=final_k,
            fetch_k=dense_cfg.get("fetch_k", 20),
            lambda_mult=dense_cfg.get("lambda_mult", 0.7),
        )

    if resolved == "sparse":
        if chunks is None:
            raise ValueError("BM25/sparse strategy requires 'chunks' parameter")
        return create_bm25_retriever(chunks=chunks, k=final_k)

    # hybrid
    if chunks is None:
        raise ValueError("Hybrid strategy requires 'chunks' parameter")

    dense_cfg = config.get("dense", {})
    hybrid_cfg = config.get("hybrid", {})

    dense_ret = create_dense_retriever(
        vectorstore=vectorstore,
        search_type=dense_cfg.get("search_type", "similarity"),
        k=dense_cfg.get("k", 10),
        fetch_k=dense_cfg.get("fetch_k", 20),
        lambda_mult=dense_cfg.get("lambda_mult", 0.7),
    )
    sparse_ret = create_bm25_retriever(
        chunks=chunks,
        k=config.get("sparse", {}).get("k", 10),
    )

    return create_hybrid_retriever(
        dense_retriever=dense_ret,
        sparse_retriever=sparse_ret,
        weights=hybrid_cfg.get("weights", [0.5, 0.5]),
    )
