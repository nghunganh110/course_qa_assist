"""Unit tests for the PDF chunking, vectorization, and search pipeline."""

import numpy as np
import pytest

from src.pdf_processor import chunk_text
from src.vectorizer import DocumentVectorizer
from src.search import search


# ---------------------------------------------------------------------------
# Mock embedder – returns deterministic unit vectors without downloading a model
# ---------------------------------------------------------------------------

class _MockEmbedder:
    """Simple deterministic embedder for offline testing.

    Each sentence is represented as a unit vector whose direction is derived
    from the hash of its text, giving unique but reproducible embeddings.
    """

    DIM = 32  # embedding dimensionality

    def encode(self, sentences, **kwargs) -> np.ndarray:
        vecs = []
        for s in sentences:
            rng = np.random.default_rng(abs(hash(s)) % (2**31))
            v = rng.random(self.DIM).astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-8))
        return np.array(vecs, dtype=np.float32)


def _make_vectorizer() -> DocumentVectorizer:
    return DocumentVectorizer(embedder=_MockEmbedder())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing handles text data.",
    "Python is a popular programming language for data science.",
    "Linear algebra is fundamental to machine learning algorithms.",
]


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_basic_split(self):
        text = " ".join(f"word{i}" for i in range(500))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.split()) <= 100

    def test_overlap_between_consecutive_chunks(self):
        text = " ".join(f"word{i}" for i in range(300))
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        assert len(chunks) >= 2
        words0 = set(chunks[0].split())
        words1 = set(chunks[1].split())
        # The last `overlap` words of chunk 0 should appear at the start of chunk 1
        assert len(words0 & words1) > 0

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []

    def test_short_text_returns_single_chunk(self):
        text = "hello world"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert chunks == ["hello world"]

    def test_chunk_size_exactly_one_word(self):
        text = "a b c d e"
        chunks = chunk_text(text, chunk_size=1, overlap=0)
        assert chunks == ["a", "b", "c", "d", "e"]

    def test_whitespace_normalization(self):
        text = "  hello   world  "
        chunks = chunk_text(text)
        assert chunks[0] == "hello world"


# ---------------------------------------------------------------------------
# DocumentVectorizer
# ---------------------------------------------------------------------------

class TestDocumentVectorizer:
    def test_fit_stores_chunks_and_matrices(self):
        vec = _make_vectorizer()
        vec.fit(SAMPLE_CHUNKS)
        assert vec.chunks == SAMPLE_CHUNKS
        assert vec.tfidf_matrix is not None
        assert vec.tfidf_matrix.shape[0] == len(SAMPLE_CHUNKS)
        assert vec.embeddings is not None
        assert vec.embeddings.shape[0] == len(SAMPLE_CHUNKS)

    def test_fit_empty_raises(self):
        vec = _make_vectorizer()
        with pytest.raises(ValueError):
            vec.fit([])

    def test_transform_query_returns_correct_shapes(self):
        vec = _make_vectorizer()
        vec.fit(SAMPLE_CHUNKS)
        tfidf_q, emb_q = vec.transform_query("machine learning")
        assert tfidf_q.shape[0] == 1
        assert emb_q.shape[0] == 1

    def test_transform_query_before_fit_raises(self):
        vec = _make_vectorizer()
        with pytest.raises(RuntimeError):
            vec.transform_query("hello")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

class TestSearch:
    @pytest.fixture(scope="class")
    def fitted_vectorizer(self):
        vec = _make_vectorizer()
        vec.fit(SAMPLE_CHUNKS)
        return vec

    def test_returns_top_k_results(self, fitted_vectorizer):
        results = search("machine learning", fitted_vectorizer, top_k=2)
        assert len(results) == 2

    def test_results_sorted_by_score(self, fitted_vectorizer):
        results = search("machine learning", fitted_vectorizer, top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_chunk_ranked_first_tfidf_only(self, fitted_vectorizer):
        # With alpha=1 (pure TF-IDF) the ML-related chunk must surface to top-3
        results = search("machine learning algorithms", fitted_vectorizer, top_k=3, alpha=1.0)
        top_chunks = " ".join(chunk.lower() for chunk, _ in results)
        assert "machine learning" in top_chunks

    def test_scores_between_zero_and_one(self, fitted_vectorizer):
        results = search("neural networks", fitted_vectorizer, top_k=3)
        for _, score in results:
            assert 0.0 <= score <= 1.0

    def test_top_k_capped_at_chunk_count(self, fitted_vectorizer):
        results = search("query", fitted_vectorizer, top_k=100)
        assert len(results) == len(SAMPLE_CHUNKS)

    def test_unfitted_vectorizer_raises(self):
        vec = _make_vectorizer()
        with pytest.raises(RuntimeError):
            search("query", vec)

    def test_alpha_pure_tfidf(self, fitted_vectorizer):
        results = search("deep learning", fitted_vectorizer, top_k=1, alpha=1.0)
        assert len(results) == 1

    def test_alpha_pure_embedding(self, fitted_vectorizer):
        results = search("deep learning", fitted_vectorizer, top_k=1, alpha=0.0)
        assert len(results) == 1
