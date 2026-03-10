"""PDF text extraction and document chunking."""

import re
from typing import List

import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Extracted text as a single string.
    """
    text_parts: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Input text to split.
        chunk_size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunks.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


def process_pdf(pdf_path: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """Extract text from a PDF and return overlapping text chunks.

    Args:
        pdf_path: Path to the PDF file.
        chunk_size: Number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunks ready for indexing.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No extractable text found in the PDF.")
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
