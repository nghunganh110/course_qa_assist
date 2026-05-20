"""Chainlit interface for the Course QA RAG pipeline.

Launch with:
    chainlit run src/app.py --port 8000

This is Phase 11: the interactive wrapper around the Phase 10 pipeline.
The default mode intentionally matches the Phase 10 end-to-end winner:
Mistral 7B + mxbai-embed-large + recursive chunks + dense similarity top-k.
"""

from __future__ import annotations

import logging
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from chainlit.user import User
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from src.config import load_config
from src.data_layer import JsonDataLayer
from src.embeddings.models import create_from_registry
from src.ingestion.chunkers import chunk_recursive
from src.ingestion.cleaners import clean_corpus
from src.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    load_documents_from_directory,
    load_single_file,
)
from src.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths and Phase 10 constants
# ---------------------------------------------------------------------------

CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
MODELS_YAML = PROJECT_ROOT / "configs" / "models.yaml"
PERSIST_DIR = PROJECT_ROOT / "vectorstore" / "chroma_db"
LECTURES_DIR = PROJECT_ROOT / "data" / "lectures"
CHAT_HISTORY_DIR = PROJECT_ROOT / "data" / "chat_history"

COLLECTION_NAME = "full_pipeline_course_mxbai_naive"
EMBED_MODEL_KEY = "mxbai_large"
LLM_MODEL = "mistral:7b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_K = 5


# ---------------------------------------------------------------------------
# Startup initialization
# ---------------------------------------------------------------------------

print("[init] Loading Phase 10 configuration...")
CONFIG = load_config(str(CONFIG_PATH))

print("[init] Loading embeddings...")
EMBEDDINGS, EMB_INFO = create_from_registry(
    EMBED_MODEL_KEY,
    config_path=str(MODELS_YAML),
)

print("[init] Loading LLM...")
LLM = ChatOllama(model=LLM_MODEL, temperature=0.0)

print("[init] Loading lecture corpus...")
LECTURES_DIR.mkdir(parents=True, exist_ok=True)
RAW_DOCS = load_documents_from_directory(str(LECTURES_DIR))
CLEANED_DOCS, CLEAN_STATS = clean_corpus(RAW_DOCS, min_content_length=50)
CHUNKS = chunk_recursive(
    CLEANED_DOCS,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
).chunks

print(
    "[init] Corpus ready: "
    f"{len(RAW_DOCS)} pages -> {len(CLEANED_DOCS)} cleaned docs -> "
    f"{len(CHUNKS)} chunks"
)


def _open_or_build_vectorstore() -> Chroma:
    """Open the Phase 10 Chroma collection, rebuilding if it is missing/stale."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=EMBEDDINGS,
        persist_directory=str(PERSIST_DIR),
    )

    try:
        existing_count = vectorstore._collection.count()
    except Exception:
        existing_count = 0

    if existing_count == len(CHUNKS):
        print(
            f"[init] Reusing collection '{COLLECTION_NAME}' "
            f"({existing_count} docs)"
        )
        return vectorstore

    if existing_count:
        print(
            f"[init] Collection '{COLLECTION_NAME}' has {existing_count} docs; "
            f"rebuilding for {len(CHUNKS)} chunks."
        )
        try:
            vectorstore.delete_collection()
        except Exception:
            logger.exception("Could not delete stale Chroma collection")

    print(f"[init] Building collection '{COLLECTION_NAME}'...")
    start = time.perf_counter()
    vectorstore = Chroma.from_documents(
        documents=CHUNKS,
        embedding=EMBEDDINGS,
        collection_name=COLLECTION_NAME,
        persist_directory=str(PERSIST_DIR),
    )
    elapsed_s = time.perf_counter() - start
    print(f"[init] Collection built in {elapsed_s:.1f}s")
    return vectorstore


VECTORSTORE = _open_or_build_vectorstore()
print("[init] Ready.")


# ---------------------------------------------------------------------------
# Chainlit persistence and auth
# ---------------------------------------------------------------------------

@cl.data_layer
def init_data_layer():
    return JsonDataLayer(CHAT_HISTORY_DIR)


@cl.header_auth_callback
async def header_auth(headers: dict) -> Optional[User]:
    """Transparent local single-user auth."""
    return User(identifier="local", metadata={"role": "user"})


# ---------------------------------------------------------------------------
# RAG mode presets
# ---------------------------------------------------------------------------

DIRECT_LLM_MODE = "Direct LLM"

MODE_CONFIGS = {
    "Recommended (dense similarity)": {
        "description": "Phase 10 winner: best RAGAS score and fastest RAGAS latency.",
        "retrieval": {
            "strategy": "similarity",
            "dense": {
                "search_type": "similarity",
                "k": DEFAULT_K,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
            "final_k": DEFAULT_K,
        },
        "reranking": {"enabled": False},
    },
    "MMR retrieval": {
        "description": "Retrieval-only winner from Phase 4; useful for more diverse context.",
        "retrieval": {
            "strategy": "mmr",
            "dense": {
                "search_type": "mmr",
                "k": DEFAULT_K,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            },
            "final_k": DEFAULT_K,
        },
        "reranking": {"enabled": False},
    },
    "Hybrid + cross-encoder": {
        "description": "Recall-focused mode; slower and lower Phase 9 RAGAS average.",
        "retrieval": {
            "strategy": "hybrid",
            "dense": {
                "search_type": "similarity",
                "k": 10,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
            "sparse": {"k": 10},
            "hybrid": {"weights": [0.5, 0.5]},
            "final_k": DEFAULT_K,
        },
        "reranking": {
            "enabled": True,
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": DEFAULT_K,
        },
    },
}

MODE_NAMES = list(MODE_CONFIGS) + [DIRECT_LLM_MODE]
DEFAULT_MODE = "Recommended (dense similarity)"

DEFAULT_SETTINGS = {
    "rag_mode": DEFAULT_MODE,
    "num_results": DEFAULT_K,
    "show_sources": True,
    "conversation_memory": True,
    "comparison_mode": False,
}


def _build_config(mode: str, num_results: int) -> dict:
    """Return a config dict for the selected UI mode."""
    cfg = deepcopy(CONFIG)
    preset = MODE_CONFIGS.get(mode, MODE_CONFIGS[DEFAULT_MODE])

    cfg["llm"]["model"] = LLM_MODEL
    cfg["llm"]["temperature"] = 0.0
    cfg["embeddings"]["registry_key"] = EMBED_MODEL_KEY
    cfg["embeddings"]["model"] = EMB_INFO.model_id
    cfg["chunking"]["strategy"] = "recursive"
    cfg["chunking"]["chunk_size"] = CHUNK_SIZE
    cfg["chunking"]["chunk_overlap"] = CHUNK_OVERLAP
    cfg["query_translation"]["enabled"] = False
    cfg["routing"]["enabled"] = False

    cfg["retrieval"] = deepcopy(preset["retrieval"])
    cfg["retrieval"]["final_k"] = int(num_results)

    if cfg["retrieval"]["strategy"] in {"similarity", "mmr"}:
        cfg["retrieval"].setdefault("dense", {})
        cfg["retrieval"]["dense"]["k"] = int(num_results)

    cfg["reranking"] = deepcopy(preset["reranking"])
    if cfg["reranking"].get("enabled"):
        cfg["reranking"]["top_k"] = int(num_results)

    return cfg


def _build_pipeline(mode: str, num_results: int) -> RAGPipeline:
    """Build a session-scoped RAGPipeline."""
    return RAGPipeline(
        config=_build_config(mode, num_results),
        vectorstore=VECTORSTORE,
        chunks=CHUNKS,
        embeddings=EMBEDDINGS,
        llm=LLM,
    )


def _settings_widgets(settings: dict) -> list:
    mode = settings.get("rag_mode", DEFAULT_MODE)
    return [
        Select(
            id="rag_mode",
            label="Answer mode",
            values=MODE_NAMES,
            initial_index=MODE_NAMES.index(mode)
            if mode in MODE_NAMES else MODE_NAMES.index(DEFAULT_MODE),
        ),
        Slider(
            id="num_results",
            label="Retrieved chunks",
            initial=int(settings.get("num_results", DEFAULT_K)),
            min=1,
            max=10,
            step=1,
        ),
        Switch(
            id="show_sources",
            label="Show sources",
            initial=bool(settings.get("show_sources", True)),
        ),
        Switch(
            id="conversation_memory",
            label="Conversation memory",
            initial=bool(settings.get("conversation_memory", True)),
        ),
        Switch(
            id="comparison_mode",
            label="Compare with direct LLM",
            initial=bool(settings.get("comparison_mode", False)),
        ),
    ]


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

MAX_HISTORY_TURNS = 3


def _format_history(history: list[dict]) -> str:
    lines = []
    for msg in history[-MAX_HISTORY_TURNS * 2:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"].replace("\n", " ")[:400]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def _reformulate(question: str, history: list[dict]) -> str:
    """Turn follow-up questions into standalone questions when possible."""
    if len(history) < 2:
        return question

    prompt = (
        "Rewrite the follow-up question as a standalone question using the "
        "conversation history. Return only the rewritten question.\n\n"
        f"Conversation:\n{_format_history(history)}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    try:
        response = await LLM.ainvoke([HumanMessage(content=prompt)])
        standalone = response.content.strip()
        if 5 < len(standalone) < 500:
            return standalone
    except Exception:
        logger.exception("Question reformulation failed")

    return question


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _source_title(source: str) -> str:
    return Path(source).name if source else "unknown"


def _page_label(doc) -> str:
    page = doc.metadata.get("page")
    if page is None:
        return "page unknown"
    try:
        return f"page {int(page) + 1}"
    except (TypeError, ValueError):
        return f"page {page}"


def _format_retrieval_step(docs, retrieval_ms: float) -> str:
    lines = [f"Retrieved {len(docs)} chunks in {retrieval_ms:.0f} ms.\n"]
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = _source_title(source)
        preview = doc.page_content[:260].replace("\n", " ")
        if len(doc.page_content) > 260:
            preview += "..."
        lines.append(f"{i}. {title} ({_page_label(doc)})")
        lines.append(f"   {preview}\n")
    return "\n".join(lines)


def _source_elements(docs) -> tuple[str, list[cl.Text]]:
    elements: list[cl.Text] = []
    lines = [
        "\n\n---\n",
        f"**Sources ({len(docs)} retrieved chunks)**\n",
    ]

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        title = _source_title(source)
        page = _page_label(doc)
        char_count = len(doc.page_content)
        element_name = f"Source {i} - {title}"

        lines.append(f"{i}. {element_name} - {page}, {char_count} chars")

        panel = "\n".join([
            f"# {title}",
            "",
            f"**File:** {source}",
            f"**Location:** {page}",
            f"**Chunk length:** {char_count} characters",
            "",
            "---",
            "",
            doc.page_content,
        ])
        elements.append(cl.Text(name=element_name, content=panel, display="side"))

    return "\n".join(lines), elements


async def _stream_direct_answer(question: str, title: str = "Direct LLM") -> None:
    msg = cl.Message(content="")
    await msg.stream_token(f"**{title}:**\n\n")

    start = time.perf_counter()
    async for chunk in LLM.astream([HumanMessage(content=question)]):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        await msg.stream_token(token)
    elapsed_ms = (time.perf_counter() - start) * 1000

    await msg.stream_token(f"\n\n---\n_Generation: {elapsed_ms:,.0f} ms_")
    await msg.send()


# ---------------------------------------------------------------------------
# Chainlit lifecycle
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start():
    pipeline = _build_pipeline(DEFAULT_MODE, DEFAULT_K)
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("history", [])
    cl.user_session.set("settings", dict(DEFAULT_SETTINGS))

    await cl.ChatSettings(_settings_widgets(DEFAULT_SETTINGS)).send()

    summary = pipeline.component_summary()
    await cl.Message(
        content=(
            "**Course QA Assistant ready**\n\n"
            f"- Mode: {DEFAULT_MODE}\n"
            f"- Corpus: {len(RAW_DOCS)} pages, {len(CLEANED_DOCS)} cleaned docs, "
            f"{len(CHUNKS)} chunks\n"
            f"- Embeddings: {EMB_INFO.model_id} ({EMB_INFO.dimensions} dims)\n"
            f"- Retrieval: {summary['retrieval_strategy']} top-{summary['retrieval_k']}\n"
            f"- Reranking: {'enabled' if summary['reranking_enabled'] else 'disabled'}\n"
            f"- LLM: {summary['llm']}"
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    metadata = thread.get("metadata", {})
    settings = metadata.get("chat_settings", dict(DEFAULT_SETTINGS))
    cl.user_session.set("settings", settings)

    mode = settings.get("rag_mode", DEFAULT_MODE)
    num_results = int(settings.get("num_results", DEFAULT_K))
    pipeline_mode = DEFAULT_MODE if mode == DIRECT_LLM_MODE else mode
    cl.user_session.set("pipeline", _build_pipeline(pipeline_mode, num_results))

    history = []
    for step in thread.get("steps", []):
        step_type = step.get("type")
        content = step.get("output", "")
        if step_type == "user_message":
            history.append({"role": "user", "content": content})
        elif step_type == "assistant_message":
            history.append({"role": "assistant", "content": content})
    cl.user_session.set("history", history)

    await cl.ChatSettings(_settings_widgets(settings)).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    old_settings = cl.user_session.get("settings", {})
    cl.user_session.set("settings", settings)

    mode = settings.get("rag_mode", DEFAULT_MODE)
    num_results = int(settings.get("num_results", DEFAULT_K))

    mode_changed = mode != old_settings.get("rag_mode")
    k_changed = num_results != int(old_settings.get("num_results", DEFAULT_K))
    comparison_changed = (
        settings.get("comparison_mode") != old_settings.get("comparison_mode")
    )

    if mode != DIRECT_LLM_MODE and (mode_changed or k_changed):
        pipeline = _build_pipeline(mode, num_results)
        cl.user_session.set("pipeline", pipeline)
        summary = pipeline.component_summary()
        description = MODE_CONFIGS.get(mode, {}).get("description", "")
        await cl.Message(
            content=(
                f"Settings updated: **{mode}**\n\n"
                f"- {description}\n"
                f"- Retrieval: {summary['retrieval_strategy']} top-{num_results}\n"
                f"- Reranking: {'enabled' if summary['reranking_enabled'] else 'disabled'}"
            )
        ).send()
    elif mode == DIRECT_LLM_MODE and mode_changed:
        await cl.Message(
            content="Settings updated: **Direct LLM**. Retrieval is disabled."
        ).send()

    if comparison_changed:
        state = "enabled" if settings.get("comparison_mode") else "disabled"
        await cl.Message(content=f"Comparison mode {state}.").send()


async def _handle_file_upload(message: cl.Message) -> bool:
    """Index uploaded files into the active collection."""
    if not message.elements:
        return False

    files = [element for element in message.elements if isinstance(element, cl.File)]
    if not files:
        return False

    global CHUNKS
    indexed_chunks = 0

    for file in files:
        ext = Path(file.name).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            await cl.Message(
                content=(
                    f"Unsupported file type: **{file.name}**. "
                    f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
            ).send()
            continue

        destination = LECTURES_DIR / file.name
        shutil.copy(file.path, destination)

        new_docs = load_single_file(str(destination))
        cleaned_docs, _ = clean_corpus(new_docs, min_content_length=50)
        new_chunks = chunk_recursive(
            cleaned_docs,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        ).chunks

        if not new_chunks:
            await cl.Message(content=f"No indexable text found in **{file.name}**.").send()
            continue

        VECTORSTORE.add_documents(new_chunks)
        CHUNKS = [*CHUNKS, *new_chunks]
        indexed_chunks += len(new_chunks)

        await cl.Message(
            content=(
                f"Indexed **{file.name}**: {len(new_docs)} pages, "
                f"{len(new_chunks)} chunks."
            )
        ).send()

    if indexed_chunks:
        settings = cl.user_session.get("settings", DEFAULT_SETTINGS)
        mode = settings.get("rag_mode", DEFAULT_MODE)
        if mode != DIRECT_LLM_MODE:
            cl.user_session.set(
                "pipeline",
                _build_pipeline(mode, int(settings.get("num_results", DEFAULT_K))),
            )

    return indexed_chunks > 0


@cl.on_message
async def on_message(message: cl.Message):
    if await _handle_file_upload(message) and not message.content.strip():
        return

    settings = cl.user_session.get("settings", DEFAULT_SETTINGS)
    history: list[dict] = cl.user_session.get("history", [])
    mode = settings.get("rag_mode", DEFAULT_MODE)
    comparison = bool(settings.get("comparison_mode", False))
    question = message.content.strip()

    if not question:
        return

    if settings.get("conversation_memory", True):
        question = await _reformulate(question, history)

    if mode == DIRECT_LLM_MODE and not comparison:
        await _stream_direct_answer(question, title="Direct LLM")
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": "Direct LLM answer streamed."})
        cl.user_session.set("history", history)
        return

    pipeline: RAGPipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        pipeline = _build_pipeline(DEFAULT_MODE, DEFAULT_K)
        cl.user_session.set("pipeline", pipeline)

    msg = cl.Message(content="")
    if comparison:
        await msg.stream_token("**RAG answer:**\n\n")

    source_docs = []
    retrieval_ms = 0.0
    total_ms = 0.0

    async for event in pipeline.astream(question):
        if event["type"] == "retrieval":
            source_docs = event["docs"]
            retrieval_ms = event["retrieval_ms"]
            async with cl.Step(
                name="Retrieval",
                type="retrieval",
                default_open=False,
            ) as step:
                step.output = _format_retrieval_step(source_docs, retrieval_ms)

        elif event["type"] == "token":
            await msg.stream_token(event["token"])

        elif event["type"] == "done":
            total_ms = event["elapsed_ms"]

    generation_ms = max(total_ms - retrieval_ms, 0.0)

    if settings.get("show_sources", True) and source_docs:
        source_text, elements = _source_elements(source_docs)
        await msg.stream_token(source_text)
        msg.elements = elements

    await msg.stream_token(
        f"\n\n_Retrieval: {retrieval_ms:,.0f} ms "
        f"| Generation: {generation_ms:,.0f} ms "
        f"| Total: {total_ms:,.0f} ms_"
    )
    await msg.send()

    history.append({"role": "user", "content": message.content})
    history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("history", history)

    if comparison and mode != DIRECT_LLM_MODE:
        await _stream_direct_answer(question, title="Direct LLM")
