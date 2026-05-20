"""
Embedding model factory for the RAG pipeline.

Provides a unified interface to create and benchmark embedding models
from different providers (Ollama, sentence-transformers).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


@dataclass
class EmbeddingModelInfo:
    """Metadata about an embedding model."""

    name: str
    provider: str
    model_id: str
    dimensions: int
    max_tokens: int = 8192
    max_chars: int = 24000
    description: str = ""


@dataclass
class EmbeddingBenchmark:
    """Benchmark results for an embedding model."""

    model_name: str
    embed_corpus_time_s: float = 0.0
    avg_query_latency_ms: float = 0.0
    num_documents: int = 0
    num_queries: int = 0
    query_latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "embed_corpus_time_s": round(self.embed_corpus_time_s, 2),
            "avg_query_latency_ms": round(self.avg_query_latency_ms, 1),
            "num_documents": self.num_documents,
            "num_queries": self.num_queries,
        }


def load_model_registry(
    config_path: str = "configs/models.yaml",
) -> dict[str, EmbeddingModelInfo]:
    """Load embedding model definitions from the model registry."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        registry = yaml.safe_load(f)

    models = {}
    for key, cfg in registry.get("embedding_models", {}).items():
        models[key] = EmbeddingModelInfo(
            name=key,
            provider=cfg["provider"],
            model_id=cfg["model"],
            dimensions=cfg["dimensions"],
            max_tokens=cfg.get("max_tokens", 8192),
            max_chars=cfg.get("max_chars", 24000),
            description=cfg.get("description", ""),
        )
    return models


def create_embedding_model(
    provider: str,
    model_id: str,
    **kwargs,
) -> Embeddings:
    """Create a LangChain embedding model instance.

    Args:
        provider: "ollama" or "sentence-transformers".
        model_id: Model identifier (e.g. "nomic-embed-text", "all-MiniLM-L6-v2").
        **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
        A LangChain Embeddings instance.
    """
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        class RobustOllamaEmbeddings(OllamaEmbeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                if not self._client:
                    raise ValueError("Ollama client is not initialized.")
                
                batch_size = 50
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    try:
                        res = self._client.embed(
                            self.model, batch, options=self._default_params, keep_alive=self.keep_alive, truncate=True
                        )["embeddings"]
                        all_embeddings.extend(res)
                    except Exception:
                        for text in batch:
                            try:
                                res = self._client.embed(
                                    self.model, [text], options=self._default_params, keep_alive=self.keep_alive, truncate=True
                                )["embeddings"]
                                all_embeddings.extend(res)
                            except Exception:
                                res = self._client.embed(
                                    self.model, [text[:400]], options=self._default_params, keep_alive=self.keep_alive, truncate=True
                                )["embeddings"]
                                all_embeddings.extend(res)
                return all_embeddings

            def embed_query(self, text: str) -> list[float]:
                return self.embed_documents([text])[0]

            async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                if not self._async_client:
                    raise ValueError("Ollama client is not initialized.")
                batch_size = 50
                all_embeddings = []
                for i in range(0, len(texts), batch_size):
                    batch = texts[i : i + batch_size]
                    try:
                        res = (await self._async_client.embed(
                            self.model, batch, options=self._default_params, keep_alive=self.keep_alive, truncate=True
                        ))["embeddings"]
                        all_embeddings.extend(res)
                    except Exception:
                        for text in batch:
                            try:
                                res = (await self._async_client.embed(
                                    self.model, [text], options=self._default_params, keep_alive=self.keep_alive, truncate=True
                                ))["embeddings"]
                                all_embeddings.extend(res)
                            except Exception:
                                res = (await self._async_client.embed(
                                    self.model, [text[:400]], options=self._default_params, keep_alive=self.keep_alive, truncate=True
                                ))["embeddings"]
                                all_embeddings.extend(res)
                return all_embeddings

            async def aembed_query(self, text: str) -> list[float]:
                return (await self.aembed_documents([text]))[0]

        return RobustOllamaEmbeddings(model=model_id, **kwargs)

    if provider == "sentence-transformers":
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model_kwargs = kwargs.pop("model_kwargs", {"device": "cpu"})
        encode_kwargs = kwargs.pop("encode_kwargs", {"normalize_embeddings": True})
        return HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )

    raise ValueError(f"Unknown embedding provider: {provider}")


def create_from_registry(
    model_key: str,
    config_path: str = "configs/models.yaml",
    **kwargs,
) -> tuple[Embeddings, EmbeddingModelInfo]:
    """Create an embedding model from the model registry by key.

    Returns:
        Tuple of (embedding model instance, model info).
    """
    registry = load_model_registry(config_path)
    if model_key not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(
            f"Model '{model_key}' not in registry. Available: {available}"
        )
    info = registry[model_key]
    model = create_embedding_model(info.provider, info.model_id, **kwargs)
    return model, info


def filter_by_max_chars(
    texts: list[str],
    max_chars: int,
) -> tuple[list[str], int]:
    """Filter texts that exceed a character limit.

    Returns:
        Tuple of (filtered texts, number of dropped texts).
    """
    filtered = [t for t in texts if len(t) <= max_chars]
    return filtered, len(texts) - len(filtered)


def filter_chunks_by_max_chars(
    chunks: list[Document],
    max_chars: int,
) -> tuple[list[Document], int]:
    """Filter Document chunks that exceed a character limit.

    Returns:
        Tuple of (filtered chunks, number of dropped chunks).
    """
    filtered = [c for c in chunks if len(c.page_content) <= max_chars]
    return filtered, len(chunks) - len(filtered)


def get_strictest_max_chars(
    model_infos: list[EmbeddingModelInfo],
) -> int:
    """Return the smallest max_chars across a list of models."""
    return min(info.max_chars for info in model_infos)


def benchmark_embedding(
    model: Embeddings,
    texts: list[str],
    queries: list[str],
    model_name: str = "unknown",
    batch_size: int = 100,
) -> EmbeddingBenchmark:
    """Benchmark an embedding model on corpus embedding and query latency.

    If a batch fails (e.g. context length exceeded), falls back to
    embedding texts one by one and skips those that still fail.

    Args:
        model: LangChain Embeddings instance.
        texts: Corpus texts to embed.
        queries: Query strings to measure latency.
        model_name: Label for the results.
        batch_size: Number of texts per embedding batch.

    Returns:
        EmbeddingBenchmark with timing results.
    """
    skipped = 0

    # Corpus embedding time
    start = time.perf_counter()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            model.embed_documents(batch)
        except Exception:
            # Fallback: embed one by one to isolate failing texts
            for t in batch:
                try:
                    model.embed_documents([t])
                except Exception:
                    skipped += 1
    corpus_time = time.perf_counter() - start

    if skipped:
        print(f"  [{model_name}] Skipped {skipped}/{len(texts)} texts (context length exceeded)")

    embedded_count = len(texts) - skipped

    # Query latencies
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        model.embed_query(q)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    return EmbeddingBenchmark(
        model_name=model_name,
        embed_corpus_time_s=corpus_time,
        avg_query_latency_ms=avg_latency,
        num_documents=embedded_count,
        num_queries=len(queries),
        query_latencies_ms=latencies,
    )
