"""TF-IDF vectorization and sentence-embedding model wrapper."""

from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


class DocumentVectorizer:
    """Combines TF-IDF and sentence embeddings to represent document chunks.

    TF-IDF captures exact keyword matches; sentence embeddings capture
    semantic (contextual) meaning.  Both representations are used together
    by :class:`src.search.HybridSearcher` to rank results.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embedder: Optional[object] = None,
    ) -> None:
        """Initialize the vectorizer.

        Args:
            embedding_model: Name of the sentence-transformers model to load.
                Defaults to ``"all-MiniLM-L6-v2"`` which is fast and accurate.
                Ignored when *embedder* is provided.
            embedder: Pre-instantiated embedding object exposing an
                ``encode(sentences, **kwargs) -> np.ndarray`` method.
                Pass a custom or mock embedder here to avoid downloading a
                model at construction time (useful for testing).
        """
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._embedding_model_name = embedding_model
        self._embedder: Optional[object] = embedder  # may be None (lazy-loaded)

        # Populated after :meth:`fit`
        self.chunks: Optional[List[str]] = None
        self.tfidf_matrix: Optional[spmatrix] = None
        self.embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy model loading — avoids importing torch at import time.
    # ------------------------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, chunks: List[str]) -> None:
        """Index a list of text chunks.

        Computes both TF-IDF vectors and sentence embeddings for every chunk
        and stores them for later similarity search.

        Args:
            chunks: Non-empty list of text strings to index.
        """
        if not chunks:
            raise ValueError("chunks must be a non-empty list.")
        self.chunks = list(chunks)
        self.tfidf_matrix = self.tfidf.fit_transform(self.chunks)
        self.embeddings = self.embedder.encode(
            self.chunks, show_progress_bar=False, convert_to_numpy=True
        )

    def transform_query(self, query: str) -> Tuple[spmatrix, np.ndarray]:
        """Vectorize a user query.

        Args:
            query: The question or keyword string from the user.

        Returns:
            Tuple of ``(tfidf_vector, embedding_vector)``.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before transform_query().")
        tfidf_vec = self.tfidf.transform([query])
        embedding_vec = self.embedder.encode([query], convert_to_numpy=True)
        return tfidf_vec, embedding_vec
