"""
Chunking strategies for RAG pipeline.

Provides 5 configurable chunking approaches, from naive to semantic,
all returning lists of LangChain Documents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings


# ── Data structures ────────────────────────────────────────────


@dataclass
class ChunkingResult:
    """Output of a chunking strategy run."""

    strategy: str
    chunks: list[Document]
    elapsed_s: float
    params: dict = field(default_factory=dict)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def sizes(self) -> list[int]:
        return [len(c.page_content) for c in self.chunks]


# ── Strategy functions ─────────────────────────────────────────


def chunk_fixed(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n",
) -> ChunkingResult:
    """Fixed-size character splitting.

    Uses RecursiveCharacterTextSplitter with a single preferred separator so
    that chunk_size is always respected (CharacterTextSplitter does not
    guarantee the limit when no separator is found within the window).
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=[separator, ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    start = time.perf_counter()
    chunks = splitter.split_documents(docs)
    elapsed = time.perf_counter() - start
    return ChunkingResult(
        strategy="fixed",
        chunks=chunks,
        elapsed_s=elapsed,
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "separator": repr(separator)},
    )


def chunk_recursive(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> ChunkingResult:
    """Recursive character splitting with a hierarchy of separators."""
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
    )
    start = time.perf_counter()
    chunks = splitter.split_documents(docs)
    elapsed = time.perf_counter() - start
    return ChunkingResult(
        strategy="recursive",
        chunks=chunks,
        elapsed_s=elapsed,
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
    )


def chunk_token(
    docs: list[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 50,
    encoding_name: str = "cl100k_base",
) -> ChunkingResult:
    """Token-based splitting using tiktoken."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    start = time.perf_counter()
    chunks = splitter.split_documents(docs)
    elapsed = time.perf_counter() - start
    return ChunkingResult(
        strategy="token",
        chunks=chunks,
        elapsed_s=elapsed,
        params={
            "chunk_size_tokens": chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
            "encoding": encoding_name,
        },
    )


def chunk_markdown(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ChunkingResult:
    """Two-stage markdown-aware splitting.

    Stage 1: split on markdown headers to get logical sections.
    Stage 2: apply recursive splitting on sections that exceed chunk_size.
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    start = time.perf_counter()
    all_chunks: list[Document] = []
    for doc in docs:
        # Stage 1: markdown header split
        md_chunks = md_splitter.split_text(doc.page_content)
        for md_chunk in md_chunks:
            # Merge original metadata with header metadata
            merged_meta = {**doc.metadata, **md_chunk.metadata}
            enriched = Document(page_content=md_chunk.page_content, metadata=merged_meta)

            if len(enriched.page_content) > chunk_size:
                # Stage 2: recursive split for oversized sections
                sub_chunks = secondary_splitter.split_documents([enriched])
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(enriched)

    elapsed = time.perf_counter() - start
    return ChunkingResult(
        strategy="markdown",
        chunks=all_chunks,
        elapsed_s=elapsed,
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
    )


def chunk_semantic(
    docs: list[Document],
    embeddings: Embeddings,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float = 75,
    max_doc_chars: int = 20000,
) -> ChunkingResult:
    """Semantic splitting based on embedding similarity between sentences.

    Documents exceeding max_doc_chars are skipped (too long for the embedding
    model's context window). Documents that fail for other reasons are also
    skipped with a warning printed to stdout.
    """
    from langchain_experimental.text_splitter import SemanticChunker

    splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount,
    )
    start = time.perf_counter()

    all_chunks: list[Document] = []
    skipped = 0
    for doc in docs:
        if len(doc.page_content) > max_doc_chars:
            skipped += 1
            continue
        try:
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
        except Exception:
            skipped += 1

    elapsed = time.perf_counter() - start
    if skipped:
        print(f"  [semantic] Skipped {skipped}/{len(docs)} docs (too long or embedding error)")
    return ChunkingResult(
        strategy="semantic",
        chunks=all_chunks,
        elapsed_s=elapsed,
        params={
            "breakpoint_type": breakpoint_threshold_type,
            "breakpoint_amount": breakpoint_threshold_amount,
            "max_doc_chars": max_doc_chars,
            "skipped_docs": skipped,
        },
    )


def chunk_course_pdf(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> ChunkingResult:
    """PDF-aware splitting generalized for university course materials.
    
    Uses a two-stage approach:
    1. Splits on common course headers (e.g., 'Week 1', 'Grading Policy')
    2. Enforces strict chunk_size using standard character splitting.
    """
    # Stage 1: Split on headers
    header_separator = r"\n(?:\d+[\.\)]\s*)?[A-Z][a-zA-Z0-9\s:_-]{2,60}(?=\n)"
    stage1_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000000,  # Arbitrarily large so it only splits on headers
        chunk_overlap=0,
        separators=[header_separator],
        is_separator_regex=True,
        length_function=len,
    )
    
    # Stage 2: Standard character splitting to strictly enforce chunk_size
    stage2_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False,
        length_function=len,
    )

    start = time.perf_counter()
    # Apply two-stage splitting
    stage1_chunks = stage1_splitter.split_documents(docs)
    chunks = stage2_splitter.split_documents(stage1_chunks)
    
    elapsed = time.perf_counter() - start
    return ChunkingResult(
        strategy="course_pdf",
        chunks=chunks,
        elapsed_s=elapsed,
        params={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "two_stage": True},
    )

