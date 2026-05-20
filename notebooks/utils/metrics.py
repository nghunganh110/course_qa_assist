"""
Custom metric functions for RAG exploration notebooks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""

    query: str
    docs: list[Document]
    scores: list[float]
    latency_ms: float


@dataclass
class IndexingMetrics:
    """Metrics collected during indexing."""

    num_raw_docs: int = 0
    num_chunks: int = 0
    chunk_sizes: list[int] = field(default_factory=list)
    indexing_time_s: float = 0.0

    @property
    def avg_chunk_size(self) -> float:
        return sum(self.chunk_sizes) / len(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def min_chunk_size(self) -> int:
        return min(self.chunk_sizes) if self.chunk_sizes else 0

    @property
    def max_chunk_size(self) -> int:
        return max(self.chunk_sizes) if self.chunk_sizes else 0

    def summary(self) -> dict:
        return {
            "raw_documents": self.num_raw_docs,
            "chunks": self.num_chunks,
            "avg_chunk_chars": round(self.avg_chunk_size),
            "min_chunk_chars": self.min_chunk_size,
            "max_chunk_chars": self.max_chunk_size,
            "indexing_time_s": round(self.indexing_time_s, 2),
        }


def timed_retrieval(
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
    search_type: str = "similarity",
) -> RetrievalResult:
    """Run a retrieval query and measure latency."""
    start = time.perf_counter()

    if search_type == "similarity":
        results = vectorstore.similarity_search_with_score(query, k=k)
    elif search_type == "mmr":
        # MMR doesn't return scores directly, use similarity for scoring
        results = vectorstore.similarity_search_with_score(query, k=k)
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    elapsed_ms = (time.perf_counter() - start) * 1000

    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]

    return RetrievalResult(
        query=query,
        docs=docs,
        scores=scores,
        latency_ms=elapsed_ms,
    )


def benchmark_queries(
    vectorstore: VectorStore,
    queries: list[str],
    k: int = 5,
    search_type: str = "similarity",
) -> pd.DataFrame:
    """Run multiple queries and return a summary DataFrame."""
    results = []
    for query in queries:
        r = timed_retrieval(vectorstore, query, k=k, search_type=search_type)
        results.append({
            "query": query[:80],
            "num_results": len(r.docs),
            "top_score": r.scores[0] if r.scores else None,
            "avg_score": sum(r.scores) / len(r.scores) if r.scores else None,
            "latency_ms": round(r.latency_ms, 1),
            "top_source": r.docs[0].metadata.get("source", "?") if r.docs else "N/A",
        })
    return pd.DataFrame(results)


# ── Phase 4: Information Retrieval metrics ────────────────────

import json
import math
from pathlib import Path


@dataclass
class BenchmarkQuestion:
    """A benchmark question with relevance annotations."""

    id: str
    query: str
    category: str
    relevant_sources: list[str] = field(default_factory=list)
    relevant_keywords: list[str] = field(default_factory=list)


def load_benchmark_questions(
    path: str = "./data/evaluation/benchmark_retrieval.json",
) -> list[BenchmarkQuestion]:
    """Load benchmark questions from a JSON file.

    Args:
        path: Path to the benchmark questions JSON.

    Returns:
        List of BenchmarkQuestion instances.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark questions not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return [
        BenchmarkQuestion(
            id=q["id"],
            query=q["query"],
            category=q["category"],
            relevant_sources=q.get("relevant_sources", []),
            relevant_keywords=q.get("relevant_keywords", []),
        )
        for q in raw
    ]


def is_relevant(doc: Document, question: BenchmarkQuestion) -> bool:
    """Check if a retrieved document is relevant to a benchmark question.

    Requires the source to match AND the chunk to contain at least one of the
    relevant keywords (if any are defined for the question).

    Args:
        doc: A retrieved Document.
        question: The benchmark question with relevance annotations.

    Returns:
        True if the document matches the source pattern and keyword requirements.
    """
    source = doc.metadata.get("source", "").lower()

    source_match = False
    for pattern in question.relevant_sources:
        if pattern.lower() in source:
            source_match = True
            break
            
    if not source_match:
        return False
        
    # If no keywords are defined, source match is enough
    if not question.relevant_keywords:
        return True
        
    # Check if chunk contains at least one relevant keyword
    content = doc.page_content.lower()
    for keyword in question.relevant_keywords:
        if keyword.lower() in content:
            return True
            
    return False


def precision_at_k(
    docs: list[Document],
    question: BenchmarkQuestion,
    k: int = 5,
) -> float:
    """Fraction of top-k results that are relevant.

    Args:
        docs: Retrieved documents ordered by rank.
        question: Benchmark question with relevance annotations.
        k: Number of top results to consider.

    Returns:
        Precision score between 0.0 and 1.0.
    """
    top_k = docs[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(1 for d in top_k if is_relevant(d, question))
    return relevant_count / len(top_k)


def recall_at_k(
    docs: list[Document],
    question: BenchmarkQuestion,
    k: int = 5,
    total_relevant: int | None = None,
) -> float:
    """Fraction of relevant docs found in top-k.

    Args:
        docs: Retrieved documents ordered by rank.
        question: Benchmark question with relevance annotations.
        k: Number of top results to consider.
        total_relevant: Total number of relevant docs in corpus.
            If None, uses k as estimate for chunk-level retrieval.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    top_k = docs[:k]
    if total_relevant is None:
        total_relevant = k  # Assume at least k relevant chunks exist
    if total_relevant == 0:
        return 0.0
    relevant_found = sum(1 for d in top_k if is_relevant(d, question))
    return min(relevant_found / total_relevant, 1.0)


def mean_reciprocal_rank(
    docs: list[Document],
    question: BenchmarkQuestion,
) -> float:
    """Reciprocal of the rank of the first relevant result.

    Args:
        docs: Retrieved documents ordered by rank.
        question: Benchmark question with relevance annotations.

    Returns:
        MRR score between 0.0 and 1.0.
    """
    for i, doc in enumerate(docs):
        if is_relevant(doc, question):
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    docs: list[Document],
    question: BenchmarkQuestion,
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain with binary relevance.

    Args:
        docs: Retrieved documents ordered by rank.
        question: Benchmark question with relevance annotations.
        k: Number of top results to consider.

    Returns:
        nDCG score between 0.0 and 1.0.
    """
    top_k = docs[:k]
    dcg = sum(
        (1.0 if is_relevant(doc, question) else 0.0) / math.log2(i + 2)
        for i, doc in enumerate(top_k)
    )
    # Assume ideal case has all top-k positions filled with relevant chunks
    idcg = sum(1.0 / math.log2(i + 2) for i in range(k))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def diversity_score(
    docs: list[Document],
    embedding_model,
) -> float:
    """Average pairwise cosine distance among retrieved documents.

    Args:
        docs: Retrieved documents.
        embedding_model: LangChain Embeddings instance.

    Returns:
        Average distance (0.0 = identical, higher = more diverse).
    """
    if len(docs) < 2:
        return 0.0

    import numpy as np

    texts = [d.page_content for d in docs]
    embeddings = embedding_model.embed_documents(texts)
    vecs = np.array(embeddings)

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs_normed = vecs / norms

    sim_matrix = vecs_normed @ vecs_normed.T
    n = len(docs)
    total_distance = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_distance += 1.0 - sim_matrix[i, j]
            count += 1

    return total_distance / count if count > 0 else 0.0


def compute_retrieval_metrics(
    docs: list[Document],
    question: BenchmarkQuestion,
    latency_ms: float,
    embedding_model=None,
    k: int = 5,
) -> dict:
    """Compute all retrieval metrics for a single query.

    Args:
        docs: Retrieved documents ordered by rank.
        question: Benchmark question with relevance annotations.
        latency_ms: Retrieval latency in milliseconds.
        embedding_model: Optional embeddings model for diversity score.
        k: Number of top results for precision/recall/nDCG.

    Returns:
        Dictionary with all metric values.
    """
    metrics = {
        "query_id": question.id,
        "query": question.query,
        "category": question.category,
        "num_results": len(docs),
        "precision_at_k": round(precision_at_k(docs, question, k), 4),
        "recall_at_k": round(recall_at_k(docs, question, k), 4),
        "mrr": round(mean_reciprocal_rank(docs, question), 4),
        "ndcg_at_k": round(ndcg_at_k(docs, question, k), 4),
        "latency_ms": round(latency_ms, 1),
    }
    if embedding_model is not None:
        metrics["diversity"] = round(diversity_score(docs, embedding_model), 4)
    return metrics
