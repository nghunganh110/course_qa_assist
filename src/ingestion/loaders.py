"""Document loaders for the Course QA pipeline.

Loads lecture notes from PDF, DOCX, PPTX, and TXT files.
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Map file extensions to LangChain loader classes
_LOADER_MAP = {
    ".pdf": "langchain_community.document_loaders.PyPDFLoader",
    ".docx": "langchain_community.document_loaders.Docx2txtLoader",
    ".pptx": "langchain_community.document_loaders.UnstructuredPowerPointLoader",
    ".txt": "langchain_community.document_loaders.TextLoader",
    ".md": "langchain_community.document_loaders.TextLoader",
}

SUPPORTED_EXTENSIONS = set(_LOADER_MAP.keys())


def _get_loader(file_path: Path):
    """Dynamically import and instantiate the right loader for a file."""
    ext = file_path.suffix.lower()
    loader_path = _LOADER_MAP.get(ext)
    if not loader_path:
        raise ValueError(f"Unsupported file type: {ext}")

    module_path, class_name = loader_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    loader_class = getattr(module, class_name)
    return loader_class(str(file_path))


def load_documents_from_directory(
    directory: str = "./data/lectures",
    course_name: str = "default",
) -> list[Document]:
    """Load all supported documents from a directory.

    Each document gets metadata: source (filename), file_type, course.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Lecture directory not found: {directory}")

    documents = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(dir_path.glob(f"**/*{ext}")):
            try:
                loader = _get_loader(file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata.update({
                        "source": str(file_path.relative_to(dir_path)),
                        "file_type": ext,
                        "course": course_name,
                    })
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} pages from {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to load {file_path.name}: {e}")

    print(f"Loaded {len(documents)} document pages from {directory}")
    return documents


def load_single_file(file_path: str, course_name: str = "default") -> list[Document]:
    """Load a single file (used for runtime uploads)."""
    path = Path(file_path)
    loader = _get_loader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata.update({
            "source": path.name,
            "file_type": path.suffix.lower(),
            "course": course_name,
        })
    return docs
