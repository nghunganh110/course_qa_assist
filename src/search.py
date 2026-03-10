"""Hybrid cosine-similarity search over indexed document chunks."""

from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .vectorizer import DocumentVectorizer


def search(
    query: str,
    vectorizer: DocumentVectorizer,
    top_k: int = 3,
    alpha: float = 0.5,
) -> List[Tuple[str, float]]:
    """Return the *top_k* most relevant chunks for *query*.

    Retrieval uses a hybrid score that blends TF-IDF keyword overlap with
    dense sentence-embedding similarity:

    .. code-block:: text

        score = alpha * tfidf_cosine + (1 - alpha) * embedding_cosine

    Args:
        query: The user's question or keyword string.
        vectorizer: A :class:`~src.vectorizer.DocumentVectorizer` that has
            already been fitted on the document chunks.
        top_k: Number of top results to return.
        alpha: Weighting factor for TF-IDF similarity (range 0–1).
            ``alpha=1`` is pure TF-IDF; ``alpha=0`` is pure embedding search.

    Returns:
        List of ``(chunk_text, score)`` tuples sorted by descending score.

    Raises:
        RuntimeError: If *vectorizer* has not been fitted.
    """
    if vectorizer.chunks is None:
        raise RuntimeError("The vectorizer has not been fitted yet.")

    tfidf_query, embedding_query = vectorizer.transform_query(query)

    tfidf_scores: np.ndarray = cosine_similarity(
        tfidf_query, vectorizer.tfidf_matrix
    ).flatten()
    embedding_scores: np.ndarray = cosine_similarity(
        embedding_query, vectorizer.embeddings
    ).flatten()

    combined_scores = alpha * tfidf_scores + (1 - alpha) * embedding_scores

    top_k = min(top_k, len(vectorizer.chunks))
    top_indices = np.argsort(combined_scores)[::-1][:top_k]

    return [(vectorizer.chunks[i], float(combined_scores[i])) for i in top_indices]
