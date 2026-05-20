# Course QA Assistant

![Python](https://img.shields.io/badge/Python-3.13-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![LangGraph](https://img.shields.io/badge/LangGraph-advanced_RAG-green)
![Ollama](https://img.shields.io/badge/Ollama-local-purple)
![Chainlit](https://img.shields.io/badge/Chainlit-chat_UI-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-blueviolet)

A local Retrieval-Augmented Generation system for asking questions over course lecture materials. The project explores indexing, chunking, embeddings, retrieval, query translation, routing, reranking, advanced RAG patterns, and RAGAS evaluation through benchmark notebooks, then wraps the selected pipeline in a Chainlit chat interface.

Corpus: lecture PDFs in `data/lectures`. Models: `mistral:7b` and `mxbai-embed-large` through Ollama. Vector store: local ChromaDB. No external LLM API is required for the default pipeline.

---

## Motivation

Most RAG demos stop at one pipeline: load documents, chunk, embed, retrieve, and generate. This repo treats RAG as an engineering decision process. Each notebook isolates one component, compares alternatives on the same corpus, saves metrics to `results/`, and feeds the final application design.

| Phase | What was explored | Components tested | Current finding |
| --- | --- | --- | --- |
| **Indexing** | Baseline corpus processing | PDF loading, cleaning, Chroma indexing | 2,703 pages -> 2,666 cleaned docs -> 9,137 chunks |
| **Chunking** | Splitting strategy | Fixed, recursive, token, markdown, semantic sample | Recursive 1000/200 is the practical default |
| **Embeddings** | Embedding model choice | MiniLM, nomic, BGE small, mxbai large | `mxbai_large` selected for final pipeline |
| **Retrieval** | Source ranking | Similarity, MMR, BM25, hybrid, multi-query | MMR wins retrieval-only Precision@5, but not final RAGAS |
| **Query Translation** | LLM query transforms | HyDE, RAG Fusion, step-back, multi-query, decomposition | HyDE/RAG Fusion improve retrieval but add latency |
| **Routing** | Query routing | Logical and semantic routing | Not enabled in final app |
| **Reranking** | Candidate reranking | Cross-encoder, LLM-as-judge | Cross-encoder is strong retrieval-only, optional in UI |
| **Advanced RAG** | Control-flow patterns | CRAG, Self-RAG, Adaptive RAG | Extra latency dominates on this setup |
| **Evaluation** | End-to-end answer quality | Naive, hybrid+reranked, HyDE | Naive dense retrieval wins Phase 9 RAGAS |
| **Full Pipeline** | Assembly | Phase 10 selected components | Dense similarity top-5 is final default |

The final default is intentionally simple: recursive chunks, `mxbai-embed-large`, dense similarity top-5, and `mistral:7b`. It won the saved Phase 9 RAGAS run with the best average score and latency.

---

## Features

- **10 benchmark notebooks** covering the RAG pipeline from ingestion to end-to-end evaluation
- **Chainlit chat app** with streaming answers, source panels, timing, and conversation history
- **Local-first inference** through Ollama for both generation and embeddings
- **Document uploads from the UI** with automatic loading, cleaning, chunking, and indexing
- **Multiple answer modes**: recommended dense retrieval, MMR, hybrid + cross-encoder, direct LLM
- **Conversation memory** using local LLM follow-up reformulation
- **Comparison mode** for RAG vs direct LLM answers
- **Config-driven pipeline** through `configs/default.yaml` and `configs/models.yaml`
- **JSON chat persistence** through `src/data_layer.py`
- **RAGAS evaluation** for faithfulness, answer relevancy, context precision, and context recall

---

## Architecture Overview

```text
Lecture files
data/lectures/
      |
      v
------------------------------------------------------
                  Ingestion Pipeline

  loaders.py -> cleaners.py -> chunkers.py
  PDF/DOCX/PPTX/TXT/MD
  clean_corpus(min_content_length=50)
  recursive chunks: 1000 chars / 200 overlap
------------------------------------------------------
                     |
                     v
          ----------------------------
          ChromaDB vector store
          mxbai-embed-large
          1024 dimensions
          ----------------------------
                      |
------------------------------------------------------
                  Retrieval Layer

  Default: dense similarity top-5
  Optional: MMR, hybrid BM25+dense, cross-encoder
------------------------------------------------------
                      |
                      v
          ----------------------------
          mistral:7b via Ollama
          temperature=0.0
          grounded answer prompt
          ----------------------------
                      |
                      v
          ----------------------------
          Chainlit interface
          streaming, sources,
          settings, uploads
          ----------------------------
```

---

## Tech Stack

| Layer | Technology | Notes |
| --- | --- | --- |
| LLM | `mistral:7b` via Ollama | Local generation, temperature 0.0 |
| Embeddings | `mxbai-embed-large` via Ollama | 1024 dimensions |
| Vector store | ChromaDB | Persistent local vector database |
| Sparse retrieval | BM25 | Used by optional hybrid mode |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Optional recall-focused UI mode |
| Framework | LangChain | Retrieval, prompts, chains, document abstractions |
| Advanced RAG | LangGraph | CRAG, Self-RAG, Adaptive RAG experiments |
| Evaluation | RAGAS | End-to-end answer metrics |
| UI | Chainlit | Streaming chat app and settings panel |
| Config | YAML | Pipeline config and model registry |

---

## Project Structure

```text
course_qa_assist/
|-- configs/
|   |-- default.yaml                  # Pipeline configuration
|   `-- models.yaml                   # Embedding and reranker registry
|
|-- notebooks/
|   |-- 01_indexing_basics.ipynb
|   |-- 02_chunking_strategies.ipynb
|   |-- 03_embeddings_comparison.ipynb
|   |-- 04_retrieval_methods.ipynb
|   |-- 05_query_translation.ipynb
|   |-- 06_routing.ipynb
|   |-- 07_reranking.ipynb
|   |-- 08_advanced_rag.ipynb
|   |-- 09_evaluation_ragas.ipynb
|   |-- 10_full_pipeline.ipynb
|   `-- utils/
|
|-- src/
|   |-- app.py                        # Phase 11 Chainlit interface
|   |-- pipeline.py                   # Unified sync/async RAG pipeline
|   |-- data_layer.py                 # JSON conversation persistence
|   |-- config.py                     # YAML config loader
|   |-- ingestion/
|   |   |-- loaders.py                # PDF/DOCX/PPTX/TXT/MD loaders
|   |   |-- cleaners.py               # Text cleanup
|   |   `-- chunkers.py               # Fixed, recursive, token, markdown, semantic
|   |-- embeddings/
|   |   `-- models.py                 # Embedding factory and registry loading
|   |-- retrieval/
|   |   |-- dense.py                  # Similarity and MMR
|   |   |-- sparse.py                 # BM25
|   |   |-- hybrid.py                 # Ensemble retrieval
|   |   |-- factory.py                # Config-driven retriever factory
|   |   `-- reranker.py               # Cross-encoder and LLM reranking
|   |-- chains/
|   |   |-- query_translation.py
|   |   |-- routing.py
|   |   `-- advanced.py
|   `-- evaluation/
|       `-- evaluator.py              # RAGAS helper classes
|
|-- data/
|   |-- lectures/                     # Local lecture files, ignored except .gitkeep
|   `-- evaluation/
|       |-- benchmark_retrieval.json  # Retrieval benchmark questions
|       `-- ground_truth.json         # End-to-end evaluation ground truth
|
|-- public/
|   `-- style.css                     # Chainlit UI tweaks
|-- chainlit.md                       # Chainlit welcome screen
|-- .chainlit/config.toml             # Chainlit UI configuration
|-- results/                          # Regenerable experiment outputs, ignored
`-- vectorstore/                      # ChromaDB persistence, ignored
```

---

## Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | 3.13 recommended | Current notebooks were run with Python 3.13 |
| Ollama | latest | Required for `mistral:7b` and `mxbai-embed-large` |
| GPU | optional | Speeds up local generation and reranking |

Pull the local models:

```bash
ollama pull mistral:7b
ollama pull mxbai-embed-large
```

---

## Installation

This repository currently does not track a locked dependency file. A practical setup is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install \
  chainlit \
  chromadb \
  langchain \
  langchain-chroma \
  langchain-community \
  langchain-core \
  langchain-ollama \
  langchain-text-splitters \
  langgraph \
  ragas \
  datasets \
  pandas \
  numpy \
  matplotlib \
  seaborn \
  pypdf \
  docx2txt \
  unstructured \
  python-pptx \
  rank-bm25 \
  sentence-transformers \
  tiktoken \
  pyyaml
```

If you already use the local conda environment from development, run commands with:

```bash
/home/hunganh/miniconda3/envs/test_env/bin/python
```

---

## Usage

### 1. Add lecture documents

Put course files in `data/lectures/`. Supported extensions:

- `.pdf`
- `.docx`
- `.pptx`
- `.txt`
- `.md`

The directory contents are ignored by git, except `data/lectures/.gitkeep`.

### 2. Launch the Chainlit app

```bash
DEBUG=false chainlit run src/app.py --port 8000 --headless
```

Open:

```text
http://localhost:8000
```

The app loads the lecture corpus, reuses the Chroma collection when it matches the current chunk count, and rebuilds it when stale.

### 3. Upload documents from the UI

The Chainlit app supports spontaneous file upload. Uploaded files are copied into `data/lectures/`, loaded, cleaned, chunked with the same recursive splitter, added to ChromaDB, and included in the active session pipeline.

### 4. Run notebooks

```bash
jupyter notebook notebooks/
```

The notebooks are numbered by phase and can be run independently, but later phases expect earlier result files in `results/`.

---

## Chainlit Modes

| Mode | Retrieval | Reranking | Use case |
| --- | --- | --- | --- |
| **Recommended (dense similarity)** | Dense similarity top-k | None | Phase 10 default and Phase 9 RAGAS winner |
| **MMR retrieval** | Max Marginal Relevance | None | More diverse context, retrieval-only strong performer |
| **Hybrid + cross-encoder** | BM25 + dense | Cross-encoder | Recall-focused mode; slower |
| **Direct LLM** | None | None | Compare against no retrieval |

Chat features:

| Feature | Description |
| --- | --- |
| Streaming | Token-by-token answer display |
| Sources | Retrieved chunks shown as side panels |
| Timing | Retrieval, generation, and total latency footer |
| Memory | Follow-up question reformulation with recent chat history |
| Comparison | RAG answer and direct LLM answer for the same question |
| Persistence | JSON-based local chat history |

---

## Configuration

Main pipeline settings live in `configs/default.yaml`.

| Setting | Current value | Description |
| --- | --- | --- |
| `llm.model` | `mistral:7b` | Ollama chat model |
| `llm.temperature` | `0.0` | Deterministic generation |
| `llm.num_ctx` | `4096` | Context window setting |
| `embeddings.model` | `mxbai-embed-large` | Ollama embedding model |
| `embeddings.registry_key` | `mxbai_large` | Model registry key |
| `chunking.strategy` | `recursive` | Default chunker |
| `chunking.chunk_size` | `1000` | Character chunk size |
| `chunking.chunk_overlap` | `200` | Character overlap |
| `retrieval.final_k` | `5` | Number of chunks used for generation |
| `query_translation.enabled` | `false` | Disabled in final app |
| `routing.enabled` | `false` | Disabled in final app |

Note: `configs/default.yaml` still contains a hybrid/reranking baseline section. `src/app.py` overrides the app default to the Phase 10 winner: dense similarity with reranking disabled.

---

## Benchmark Results

### Corpus summary

| Stage | Count |
| --- | ---: |
| Raw document pages | 2,703 |
| Cleaned documents | 2,666 |
| Recursive chunks | 9,137 |
| Source files in current run | 6 PDFs |

### Phase 9 RAGAS evaluation

The saved end-to-end evaluation uses 12 ground-truth questions across four categories.

| Config | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Score | Avg Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **naive** | **0.9181** | **0.9402** | 1.0000 | 0.7679 | **0.9065** | **2384.6 ms** |
| hybrid_reranked | 0.8542 | 0.8375 | 1.0000 | **0.8139** | 0.8764 | 3235.0 ms |
| hyde | 0.8491 | 0.8633 | 1.0000 | 0.7253 | 0.8594 | 7857.3 ms |

### Retrieval-only highlights

| Strategy | Precision@5 | MRR | Latency |
| --- | ---: | ---: | ---: |
| **MMR lambda=0.5** | **0.88** | **0.90** | 764.9 ms |
| MMR lambda=0.3 | 0.84 | 0.90 | 1114.4 ms |
| Multi-query | 0.80 | 0.80 | 4654.8 ms |
| Similarity | 0.76 | 0.8667 | 716.8 ms |
| BM25 | 0.60 | 0.8667 | **34.8 ms** |

### Reranking highlights

| Technique | Precision@5 | MRR | Latency |
| --- | ---: | ---: | ---: |
| **Cross-encoder** | **0.96** | **1.00** | 40.7 ms |
| LLM-as-judge | 0.84 | 0.84 | 9480.8 ms |
| No reranking | 0.76 | 0.8667 | **26.7 ms** |

### Advanced RAG highlights

| Pattern | Avg Precision | Avg MRR | Avg Latency |
| --- | ---: | ---: | ---: |
| Baseline similarity | 0.50 | 0.666 | **32 ms** |
| CRAG | 0.50 | 0.666 | 5385 ms |
| Self-RAG | 0.40 | 0.500 | 9810 ms |
| Adaptive RAG | 0.50 | 0.666 | 4504 ms |

---

## Key Lessons

1. Dense similarity is the best current default when judged end-to-end by RAGAS.
2. Retrieval-only improvements do not automatically improve generated answer quality.
3. Cross-encoder reranking is useful as an optional recall-focused mode, but not the default on the saved RAGAS run.
4. HyDE and RAG Fusion can improve retrieval metrics, but local LLM latency is high.
5. Advanced control-flow patterns add significant latency with `mistral:7b`.
6. Keeping the pipeline simple makes the Chainlit app faster, easier to debug, and easier to extend with uploaded documents.

---

## Data and Git Hygiene

The following are intentionally ignored:

- `data/lectures/*` except `.gitkeep`
- `data/raw/`
- `data/chat_history/`
- `vectorstore/`
- `results/`
- `.env`
- Chainlit runtime files under `.files/`

This keeps secrets, uploaded documents, generated vectors, chat history, and experiment outputs out of git.
