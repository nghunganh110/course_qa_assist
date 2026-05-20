"""Document cleaning for lecture notes (PDF, PPTX, DOCX extracts).

Strips common noise from extracted text: page numbers, slide markers,
excessive whitespace, and other artifacts from document conversion.
"""

import re
from langchain_core.documents import Document


def strip_headers_footers(text: str) -> str:
    """Remove common PDF headers/footers (page numbers, dates, etc.)."""
    # Standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # "Page X of Y" patterns
    text = re.sub(r"(?i)page\s+\d+\s*(of\s+\d+)?", "", text)
    return text


def strip_slide_markers(text: str) -> str:
    """Remove slide number markers from PowerPoint extracts."""
    text = re.sub(r"(?i)slide\s+\d+", "", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse excessive blank lines and trailing spaces."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    return text.strip()


def clean_document(doc: Document) -> Document:
    """Apply all cleaning steps to a single document.

    Returns a new Document with cleaned page_content.
    Metadata is preserved and a 'cleaned' flag is added.
    """
    text = doc.page_content
    text = strip_headers_footers(text)
    text = strip_slide_markers(text)
    text = normalize_whitespace(text)
    metadata = {**doc.metadata, "cleaned": True}
    return Document(page_content=text, metadata=metadata)


def clean_corpus(
    docs: list[Document],
    min_content_length: int = 50,
) -> tuple[list[Document], dict]:
    """Clean documents and filter out near-empty results.

    Same interface as the original — pipeline.py and app.py
    don't need to change.

    Args:
        docs: Raw documents to clean.
        min_content_length: Minimum character count after cleaning.
            Documents shorter than this are discarded.

    Returns:
        Tuple of (cleaned documents, stats dict).
    """
    cleaned = []
    dropped = 0

    for doc in docs:
        clean_doc = clean_document(doc)
        if len(clean_doc.page_content) >= min_content_length:
            cleaned.append(clean_doc)
        else:
            dropped += 1

    stats = {
        "original_count": len(docs),
        "cleaned_count": len(cleaned),
        "dropped_count": dropped,
        "original_total_chars": sum(len(d.page_content) for d in docs),
        "cleaned_total_chars": sum(len(d.page_content) for d in cleaned),
    }
    return cleaned, stats
