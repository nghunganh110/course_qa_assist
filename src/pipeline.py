"""Unified RAG pipeline assembled from config-driven components.

Combines the best components identified across Phases 2-9 into a single,
configurable pipeline: hybrid retrieval + cross-encoder reranking + generation.

Usage:
    from src.pipeline import RAGPipeline

    pipeline = RAGPipeline(config, vectorstore, chunks, embeddings, llm)
    result = pipeline.invoke("What is RAG?")
    print(result.answer)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Output of a single pipeline invocation."""

    question: str
    answer: str
    contexts: list[str]
    source_docs: list[Document]
    elapsed_ms: float
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0

    @property
    def num_contexts(self) -> int:
        return len(self.contexts)

    def summary(self) -> dict:
        return {
            "question": self.question[:80],
            "answer_length": len(self.answer),
            "num_contexts": self.num_contexts,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
        }


# ---------------------------------------------------------------------------
# Reranked retriever wrapper
# ---------------------------------------------------------------------------

class RerankedRetriever(BaseRetriever):
    """Wraps a base retriever with cross-encoder reranking.

    Retrieves candidate_k documents from the base retriever, then reranks
    them using a cross-encoder model and returns the top_k.
    """

    base_retriever: object
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        from src.retrieval.reranker import rerank_cross_encoder

        candidates = self.base_retriever.invoke(query)
        result = rerank_cross_encoder(
            query=query,
            candidates=candidates,
            model_name=self.reranker_model,
            top_k=self.top_k,
        )
        return result.docs


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_TEMPLATE = """You are a helpful teaching assistant. Answer the student's question
using ONLY the provided lecture notes and course materials.
If the materials don't contain enough information, say so clearly.
Be accurate, concise, and explain concepts in a way that helps the student understand.

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {d.page_content[:800]}" for i, d in enumerate(docs)
    )


class RAGPipeline:
    """Configurable RAG pipeline assembled from components.

    Args:
        config: Pipeline configuration dict (from configs/default.yaml).
        vectorstore: Initialized ChromaDB vector store.
        chunks: Document chunks (needed for BM25 in hybrid retrieval).
        embeddings: LangChain embeddings model.
        llm: LangChain chat model.
    """

    def __init__(
        self,
        config: dict,
        vectorstore: VectorStore,
        chunks: list[Document],
        embeddings: Embeddings,
        llm: BaseChatModel,
    ) -> None:
        self.config = config
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.embeddings = embeddings
        self.llm = llm

        # Build retriever from config
        from src.retrieval.factory import create_retriever_from_config

        retrieval_cfg = config.get("retrieval", {})
        base_retriever = create_retriever_from_config(
            config=retrieval_cfg,
            vectorstore=vectorstore,
            chunks=chunks,
        )

        # Max documents to use (EnsembleRetriever may return more than this)
        self._final_k = retrieval_cfg.get("final_k", 5)

        # Optionally wrap with reranking
        reranking_cfg = config.get("reranking", {})
        if reranking_cfg.get("enabled", False):
            self._retriever = RerankedRetriever(
                base_retriever=base_retriever,
                reranker_model=reranking_cfg.get(
                    "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                ),
                top_k=reranking_cfg.get("top_k", 5),
            )
        else:
            self._retriever = base_retriever

        # Load prompt from config, falling back to the default teaching assistant prompt
        prompt_template = config.get("generation", {}).get(
            "prompt_template", _DEFAULT_PROMPT_TEMPLATE
        )
        rag_prompt = ChatPromptTemplate.from_messages([
            ("human", prompt_template),
        ])

        # Generation-only chain (no retriever - we retrieve separately in invoke())
        self._gen_chain = rag_prompt | llm | StrOutputParser()

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        vectorstore: VectorStore,
        chunks: list[Document],
        embeddings: Embeddings,
        llm: BaseChatModel,
    ) -> RAGPipeline:
        """Build pipeline from a YAML config file.

        Args:
            config_path: Path to the YAML configuration file.
            vectorstore: Initialized ChromaDB vector store.
            chunks: Document chunks (needed for BM25 in hybrid retrieval).
            embeddings: LangChain embeddings model.
            llm: LangChain chat model.

        Returns:
            Configured RAGPipeline instance.
        """
        from src.config import load_config

        config = load_config(config_path)
        return cls(config, vectorstore, chunks, embeddings, llm)

    @property
    def retriever(self) -> BaseRetriever:
        """Expose the retriever for external use (e.g. RAGAS evaluation)."""
        return self._retriever

    def invoke(self, question: str) -> PipelineResult:
        """Run the full RAG pipeline on a question.

        Args:
            question: User question string.

        Returns:
            PipelineResult with answer, contexts, and timing.
        """
        start = time.perf_counter()

        # Retrieve (cap at final_k - EnsembleRetriever may return more)
        t_ret = time.perf_counter()
        docs = self._retriever.invoke(question)[:self._final_k]
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        contexts = [d.page_content for d in docs]

        # Generate (using already-retrieved docs, no second retrieval)
        t_gen = time.perf_counter()
        answer = self._gen_chain.invoke({
            "context": _format_docs(docs),
            "question": question,
        })
        generation_ms = (time.perf_counter() - t_gen) * 1000

        if not isinstance(answer, str):
            answer = str(answer)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PipelineResult(
            question=question,
            answer=answer,
            contexts=contexts,
            source_docs=docs,
            elapsed_ms=elapsed_ms,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    async def astream(self, question: str) -> AsyncGenerator[dict, None]:
        """Stream the RAG pipeline: retrieve, then stream generation tokens.

        Yields dicts with keys:
            type="retrieval" -> docs, retrieval_ms
            type="token"     -> token (str chunk)
            type="done"      -> answer, elapsed_ms, source_docs, contexts
        """
        start = time.perf_counter()

        # Retrieval (synchronous, run in thread to avoid blocking event loop)
        # Cap at final_k - EnsembleRetriever may return more
        t_ret = time.perf_counter()
        all_docs = await asyncio.to_thread(self._retriever.invoke, question)
        docs = all_docs[:self._final_k]
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        yield {
            "type": "retrieval",
            "docs": docs,
            "retrieval_ms": retrieval_ms,
        }

        # Stream generation tokens
        contexts = [d.page_content for d in docs]
        full_answer: list[str] = []
        async for chunk in self._gen_chain.astream({
            "context": _format_docs(docs),
            "question": question,
        }):
            token = chunk if isinstance(chunk, str) else str(chunk)
            full_answer.append(token)
            yield {"type": "token", "token": token}

        elapsed_ms = (time.perf_counter() - start) * 1000
        yield {
            "type": "done",
            "answer": "".join(full_answer),
            "elapsed_ms": elapsed_ms,
            "source_docs": docs,
            "contexts": contexts,
        }

    def component_summary(self) -> dict:
        """Return a summary of the pipeline components for display."""
        retrieval_cfg = self.config.get("retrieval", {})
        reranking_cfg = self.config.get("reranking", {})
        return {
            "llm": self.config.get("llm", {}).get("model", "unknown"),
            "embeddings": self.config.get("embeddings", {}).get("model", "unknown"),
            "retrieval_strategy": retrieval_cfg.get("strategy", "unknown"),
            "retrieval_k": retrieval_cfg.get("final_k", 5),
            "reranking_enabled": reranking_cfg.get("enabled", False),
            "reranking_model": reranking_cfg.get("model", "none"),
            "reranking_top_k": reranking_cfg.get("top_k", 5),
        }
