"""
Query routing for multi-index RAG retrieval.

Routes queries to the most relevant sub-index before retrieval, improving
precision by searching only the relevant portion of the corpus.

Two strategies:
- Logical routing: LLM classifies the query into a category
- Semantic routing: Cosine similarity to pre-computed category centroids

Five balanced categories for the current ML systems corpus:
- foundations: early textbook chapters on ML systems basics and deployment
- training_efficiency: training, efficient AI, optimization, acceleration
- operations_robustness: MLOps, on-device learning, security, robustness
- trust_frontiers: responsible/sustainable AI, AI for good, AGI systems
- references_research: textbook back matter plus external research papers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------

VALID_CATEGORIES = (
    "foundations",
    "training_efficiency",
    "operations_robustness",
    "trust_frontiers",
    "references_research",
)

TEXTBOOK_SOURCE = "13.Machine-Learning-Systems.pdf"

# PyPDFLoader uses zero-based page indexes. These ranges split the 2,604-page
# textbook into similarly sized routed indexes while preserving broad topic order.
TEXTBOOK_PAGE_RANGES: tuple[tuple[str, int, int | None], ...] = (
    ("foundations", 0, 520),
    ("training_efficiency", 521, 1040),
    ("operations_robustness", 1041, 1560),
    ("trust_frontiers", 1561, 2080),
    ("references_research", 2081, None),
)

SECTION_TO_CATEGORY: dict[str, str] = {
    "2605.01205v1.pdf": "references_research",  # SRA: Span Representation Alignment
    "2605.01386v1.pdf": "references_research",  # MemORAI: memory for LLM agents
    "2605.03299v1.pdf": "references_research",  # LLM-XTM: cross-lingual topic modeling
    "2605.11505v2.pdf": "references_research",  # SORT: reinforcement learning tuning
    "2605.12288v2.pdf": "references_research",  # TokenRatio: token-level DPO
}

# Map benchmark question categories to routing categories (for accuracy eval)
BENCHMARK_TO_ROUTING: dict[str, str] = {
    "methodology": "foundations",
    "results": "trust_frontiers",
    "how_to": "training_efficiency",
    "factual": "foundations",
    "technical": "training_efficiency",
    "conceptual": "foundations",
    "error_related": "operations_robustness",
}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ROUTING_PROMPT = ChatPromptTemplate.from_messages([
    (
        "human",
        """Classify the following question into exactly ONE category.

Categories:
- foundations: ML systems basics, deployment paradigms, TinyML, edge/cloud/mobile ML, neural network and workflow foundations
- training_efficiency: data engineering, AI frameworks, training, efficient AI, model optimization, accelerators, benchmarking
- operations_robustness: MLOps, on-device learning, security, privacy, robust AI, production reliability
- trust_frontiers: responsible AI, sustainable AI, AI for good, AGI systems, scaling hypothesis, future directions
- references_research: appendices, references, and external research papers such as SRA, MemORAI, TokenRatio, SORT, LLM-XTM

Examples:
Question: What constraints make TinyML systems difficult to deploy?
Category: foundations

Question: How does quantization improve efficient inference?
Category: training_efficiency

Question: How should teams monitor model drift in production?
Category: operations_robustness

Question: What role does the Scaling Hypothesis play in AGI?
Category: trust_frontiers

Question: What were the main findings of the MemORAI paper?
Category: references_research

Question: {question}
Category:""",
    ),
])


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Output of a routing + retrieval run."""

    technique: str             # "logical" | "semantic" | "no_routing"
    query: str
    predicted_category: str    # one of VALID_CATEGORIES or "all"
    confidence: float          # cosine sim for semantic; 1.0 for logical; 0.0 for fallback
    docs: list[Document]
    elapsed_ms: float
    llm_calls: int             # 0 for semantic/no_routing, 1 for logical
    params: dict = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.docs)

    def summary(self) -> dict:
        return {
            "technique": self.technique,
            "query": self.query[:80],
            "predicted_category": self.predicted_category,
            "confidence": round(self.confidence, 3),
            "num_results": self.num_results,
            "llm_calls": self.llm_calls,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _deduplicate(docs: list[Document]) -> list[Document]:
    """Remove duplicate chunks by (source, page_content[:100]) key."""
    seen: set[tuple[str, str]] = set()
    unique: list[Document] = []
    for doc in docs:
        key = (doc.metadata.get("source", ""), doc.page_content[:100])
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique


def _source_name(source: str) -> str:
    """Return the final path component without importing pathlib on hot paths."""
    return source.replace("\\", "/").rsplit("/", 1)[-1]


def route_category_from_metadata(metadata: dict) -> str | None:
    """Resolve a routing category from document metadata.

    The current corpus is dominated by one long textbook. Exact filename
    routing would put almost every chunk into one collection, so the textbook is
    split into page ranges while short research PDFs use filename mapping.
    """
    source_name = _source_name(str(metadata.get("source", "")))
    if source_name == TEXTBOOK_SOURCE:
        try:
            page = int(metadata.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        for category, start, end in TEXTBOOK_PAGE_RANGES:
            if page >= start and (end is None or page <= end):
                return category
        return "references_research"
    return SECTION_TO_CATEGORY.get(source_name)


# ---------------------------------------------------------------------------
# Centroid computation
# ---------------------------------------------------------------------------

def compute_centroids(
    category_docs: dict[str, list[Document]],
    embeddings: Embeddings,
    max_chars: int = 2000,
) -> dict[str, np.ndarray]:
    """Compute centroid embeddings for each category.

    Embeds each document individually. Non-ASCII characters (scraper artifacts,
    mermaid diagram syntax) are stripped before embedding to avoid token-count
    inflation that causes context-length errors.

    Args:
        category_docs: Mapping from category name to list of documents.
        embeddings: Embedding model (must match the vectorstore's model).
        max_chars: Truncate each document to this many characters after
            stripping non-ASCII content.

    Returns:
        Mapping from category name to centroid vector (mean of all doc embeddings).
    """
    import re

    def _clean(text: str) -> str:
        # Remove non-ASCII characters (diagram syntax, replacement chars, etc.)
        return re.sub(r"[^\x00-\x7F]+", " ", text)[:max_chars]

    centroids: dict[str, np.ndarray] = {}
    for category, docs in category_docs.items():
        if not docs:
            continue
        all_vecs: list[list[float]] = []
        for doc in docs:
            text = _clean(doc.page_content)
            vec = embeddings.embed_documents([text])[0]
            all_vecs.append(vec)
        centroids[category] = np.mean(np.array(all_vecs), axis=0)
    return centroids


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_logical(
    llm: BaseChatModel,
    query: str,
) -> str:
    """Classify query using LLM.

    Args:
        llm: Chat model to use for classification.
        query: User query to classify.

    Returns:
        Category name from VALID_CATEGORIES. Falls back to "foundations" if
        the LLM output is not recognized.
    """
    chain = ROUTING_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"question": query}).strip().lower()
    normalized = raw.replace("-", "_").replace(" ", "_")
    # Accept partial matches (e.g., "api_reference." -> "api_reference")
    for cat in VALID_CATEGORIES:
        if cat in normalized:
            return cat
    return "foundations"  # safe fallback


def classify_semantic(
    query_embedding: np.ndarray | list[float],
    centroids: dict[str, np.ndarray],
) -> tuple[str, float]:
    """Classify query by cosine similarity to category centroids.

    Args:
        query_embedding: Embedded query vector.
        centroids: Pre-computed category centroid vectors.

    Returns:
        Tuple of (best_category, confidence_score).
        Confidence is the cosine similarity to the best centroid (0.0-1.0).
    """
    q = np.array(query_embedding)
    best_cat = "foundations"
    best_sim = -1.0

    for cat, centroid in centroids.items():
        sim = _cosine_similarity(q, centroid)
        if sim > best_sim:
            best_sim = sim
            best_cat = cat

    return best_cat, best_sim


# ---------------------------------------------------------------------------
# Main routing + retrieval
# ---------------------------------------------------------------------------

def route_and_retrieve(
    query: str,
    technique: str,
    collections: dict[str, VectorStore],
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
    centroids: dict[str, np.ndarray] | None = None,
    k: int = 5,
    fallback_threshold: float = 0.3,
) -> RoutingResult:
    """Route query to the right sub-index and retrieve documents.

    Args:
        query: User query string.
        technique: "logical" (LLM-based) or "semantic" (embedding-based).
        collections: Mapping from category name to vectorstore.
        llm: Required for "logical" technique.
        embeddings: Required for "semantic" technique.
        centroids: Pre-computed centroids; required for "semantic" technique.
        k: Number of results to return.
        fallback_threshold: Min cosine similarity for semantic routing.
            Below this, search all collections (fallback mode).

    Returns:
        RoutingResult with retrieved documents and routing metadata.
    """
    start = time.perf_counter()
    llm_calls = 0
    confidence = 0.0
    predicted_category = "foundations"

    if technique == "logical":
        if llm is None:
            raise ValueError("llm is required for logical routing")
        predicted_category = classify_logical(llm, query)
        llm_calls = 1
        confidence = 1.0

    elif technique == "semantic":
        if embeddings is None or centroids is None:
            raise ValueError("embeddings and centroids are required for semantic routing")
        q_emb = embeddings.embed_query(query)
        predicted_category, confidence = classify_semantic(q_emb, centroids)
        # Fallback to all collections if confidence is too low
        if confidence < fallback_threshold:
            predicted_category = "all"

    else:
        raise ValueError(f"Unknown technique: {technique!r}. Use 'logical' or 'semantic'.")

    # Retrieve documents
    if predicted_category == "all":
        # Fallback: search all collections and merge
        all_docs: list[Document] = []
        for coll in collections.values():
            all_docs.extend(coll.similarity_search(query, k=k))
        docs = _deduplicate(all_docs)[:k]
    else:
        target = collections.get(predicted_category)
        if target is None:
            # Category not in collections dict - fall back to all
            all_docs = []
            for coll in collections.values():
                all_docs.extend(coll.similarity_search(query, k=k))
            docs = _deduplicate(all_docs)[:k]
            predicted_category = "all"
        else:
            docs = target.similarity_search(query, k=k)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return RoutingResult(
        technique=technique,
        query=query,
        predicted_category=predicted_category,
        confidence=confidence,
        docs=docs,
        elapsed_ms=elapsed_ms,
        llm_calls=llm_calls,
        params={"k": k, "fallback_threshold": fallback_threshold},
    )


# ---------------------------------------------------------------------------
# No-routing baseline
# ---------------------------------------------------------------------------

def retrieve_no_routing(
    vectorstore: VectorStore,
    query: str,
    k: int = 5,
) -> RoutingResult:
    """Baseline similarity search without routing (searches full index).

    Args:
        vectorstore: Full corpus vectorstore.
        query: User query string.
        k: Number of results to return.

    Returns:
        RoutingResult with technique="no_routing".
    """
    start = time.perf_counter()
    docs = vectorstore.similarity_search(query, k=k)
    elapsed_ms = (time.perf_counter() - start) * 1000

    return RoutingResult(
        technique="no_routing",
        query=query,
        predicted_category="all",
        confidence=0.0,
        docs=docs,
        elapsed_ms=elapsed_ms,
        llm_calls=0,
        params={"k": k},
    )
