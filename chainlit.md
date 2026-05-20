# Course QA Assistant

Ask questions about the lecture PDFs in `data/lectures` and get grounded answers with source chunks.

## Pipeline

The default mode matches the Phase 10 winner:

- `mxbai-embed-large` embeddings
- Recursive chunks with 1000 characters and 200 overlap
- Dense similarity retrieval, top 5
- Local `mistral:7b` generation
- Source chunks shown beside each answer

## Settings

Use the settings panel to switch modes:

- **Recommended (dense similarity)**: best Phase 9 RAGAS average score and latency
- **MMR retrieval**: more diverse context from the retrieval-only benchmark
- **Hybrid + cross-encoder**: recall-focused mode with reranking
- **Direct LLM**: no retrieval, useful as a baseline

You can also change the number of retrieved chunks, hide sources, enable follow-up question memory, or compare RAG against direct LLM output.

## Example Questions

- What is the role of an ML System in the AI lifecycle?
- What are the primary characteristics and constraints of TinyML systems?
- How do large-scale ML systems manage coordination strategies for real-time processing?
- What are common failure modes in ML production systems?

Everything runs locally through Ollama and ChromaDB.
