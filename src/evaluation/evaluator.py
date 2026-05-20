"""
RAGAS evaluation pipeline for RAG configurations.

Provides a reusable runner that:
- Collects (question, answer, contexts, ground_truth) tuples from any RAG chain
- Wraps local Mistral + nomic-embed-text for RAGAS metrics
- Returns structured results compatible with the notebook comparison tables

Usage:
    from src.evaluation.evaluator import RagasEvaluator, EvalSample

    evaluator = RagasEvaluator(llm=..., embeddings=...)
    samples = evaluator.collect_samples(rag_chain, retriever, questions_with_gt)
    scores = evaluator.evaluate(samples)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalSample:
    """One row in the RAGAS evaluation dataset."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    category: str = ""
    question_id: str = ""
    latency_ms: float = 0.0


@dataclass
class EvalResult:
    """Output of a full RAGAS evaluation run."""

    config_name: str
    samples: list[EvalSample]
    scores: dict[str, float]          # metric_name -> mean score
    per_sample_scores: list[dict]     # one dict per sample
    elapsed_ms: float
    params: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "config": self.config_name,
            "n_samples": len(self.samples),
            "faithfulness": round(self.scores.get("faithfulness", 0), 4),
            "answer_relevancy": round(self.scores.get("answer_relevancy", 0), 4),
            "context_precision": round(self.scores.get("context_precision", 0), 4),
            "context_recall": round(self.scores.get("context_recall", 0), 4),
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class RagasEvaluator:
    """Wraps RAGAS evaluation with local LLM and embeddings.

    Args:
        llm: LangChain chat model (Mistral 7B via Ollama).
        embeddings: LangChain embeddings model (nomic-embed-text).
    """

    def __init__(self, llm: BaseChatModel, embeddings: Embeddings) -> None:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        self._ragas_llm = LangchainLLMWrapper(llm)
        self._ragas_emb = LangchainEmbeddingsWrapper(embeddings)

    def _get_metrics(self):
        """Return configured RAGAS metric instances (RAGAS >= 0.4 instance-based API)."""
        from ragas.metrics import (  # type: ignore[import]
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )

        return [
            Faithfulness(llm=self._ragas_llm),
            AnswerRelevancy(llm=self._ragas_llm, embeddings=self._ragas_emb),
            ContextPrecision(llm=self._ragas_llm),
            ContextRecall(llm=self._ragas_llm),
        ]

    def collect_samples(
        self,
        rag_chain,
        retriever: BaseRetriever,
        questions: list[dict],
    ) -> list[EvalSample]:
        """Run the RAG chain and retriever on each question.

        Args:
            rag_chain: Any callable/chain that accepts a question string and returns
                a string answer.
            retriever: LangChain retriever used to fetch context documents.
            questions: List of dicts with keys: question, ground_truth, category, id.

        Returns:
            List of EvalSample ready for evaluate().
        """
        samples = []
        for q in questions:
            question = q["question"]
            start = time.perf_counter()

            docs = retriever.invoke(question)
            contexts = [d.page_content for d in docs]
            answer = rag_chain.invoke(question)
            if not isinstance(answer, str):
                answer = str(answer)

            latency_ms = (time.perf_counter() - start) * 1000

            samples.append(EvalSample(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=q.get("ground_truth", ""),
                category=q.get("category", ""),
                question_id=q.get("id", ""),
                latency_ms=latency_ms,
            ))

        return samples

    def evaluate(
        self,
        samples: list[EvalSample],
        config_name: str = "pipeline",
        params: dict | None = None,
    ) -> EvalResult:
        """Run RAGAS metrics on collected samples.

        Args:
            samples: Output of collect_samples().
            config_name: Human-readable name for this pipeline config.
            params: Optional metadata (chunking, retrieval strategy, etc.).

        Returns:
            EvalResult with mean scores and per-sample breakdown.
        """
        from datasets import Dataset

        start = time.perf_counter()

        records = [
            {
                "question": s.question,
                "answer": s.answer,
                "contexts": s.contexts,
                "ground_truth": s.ground_truth,
            }
            for s in samples
        ]
        dataset = Dataset.from_list(records)

        metrics = self._get_metrics()

        from ragas import evaluate  # type: ignore[import]
        # Metrics carry their own llm/embeddings (RAGAS >= 0.4 instance API).
        # raise_exceptions=False lets RAGAS return NaN for individual failures
        # instead of crashing the whole evaluation run.
        result_df = evaluate(
            dataset=dataset,
            metrics=metrics,
            raise_exceptions=False,
        ).to_pandas()

        elapsed_ms = (time.perf_counter() - start) * 1000

        metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        mean_scores = {
            col: float(result_df[col].mean())
            for col in metric_cols
            if col in result_df.columns
        }
        per_sample = result_df.to_dict(orient="records")

        return EvalResult(
            config_name=config_name,
            samples=samples,
            scores=mean_scores,
            per_sample_scores=per_sample,
            elapsed_ms=elapsed_ms,
            params=params or {},
        )
