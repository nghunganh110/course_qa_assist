# Course Notes QA Assistant

An interactive chat application that lets you upload PDF course notes and retrieve context-specific answers by finding the most relevant passages.

## Features

- **PDF Upload** — Extract and index text from any PDF course notes.
- **Hybrid Search** — Combines TF-IDF keyword matching and sentence-embedding cosine similarity for accurate retrieval.
- **Document Chunking** — Splits documents into overlapping word-based chunks to preserve context.
- **Demo UI** — Clean Gradio interface with a chat window for question-and-answer interaction.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  PDF Upload                                          │
│     │                                                │
│     ▼                                                │
│  pdf_processor.py  ── extract text ── chunk text     │
│     │                                                │
│     ▼                                                │
│  vectorizer.py  ── TF-IDF matrix + sentence embeds   │
│     │                                                │
│     ▼                                                │
│  search.py  ── cosine similarity ── ranked passages  │
│     │                                                │
│     ▼                                                │
│  app.py  ── Gradio UI                                │
└──────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the demo
python app.py
```

Open the URL printed in the terminal (default: http://127.0.0.1:7860).

## Running Tests

```bash
pip install pytest
pytest tests/
```

## Project Structure

```
course_qa_assist/
├── app.py                  # Gradio demo UI
├── requirements.txt        # Python dependencies
├── src/
│   ├── pdf_processor.py    # PDF extraction and chunking
│   ├── vectorizer.py       # TF-IDF + sentence embeddings
│   └── search.py           # Hybrid cosine similarity search
└── tests/
    └── test_pipeline.py    # Unit tests
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 200 | Words per chunk |
| `overlap` | 50 | Overlapping words between chunks |
| `top_k` | 3 | Number of passages to return |
| `alpha` | 0.5 | TF-IDF weight (1-alpha = embedding weight) |
| `embedding_model` | `all-MiniLM-L6-v2` | Sentence-transformer model |