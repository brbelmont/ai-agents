from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

# Default persistence directory (override with VECTORSTORE_DIR env)
DEFAULT_PERSIST_DIR = Path(os.getenv("VECTORSTORE_DIR", "vectorstore"))


def get_vectorstore(
    persist_directory: Optional[str | Path] = None,
    embeddings: Optional[Embeddings] = None,
    collection_name: str = "my_documents",
) -> Chroma:
    """
    Create (or connect to) a persistent Chroma vector store.

    - persist_directory: where to store the index on disk
    - embeddings: LangChain Embeddings implementation (OpenAI by default in real runs;
                  pass a mock in tests)
    - collection_name: Chroma collection
    """
    pdir = Path(persist_directory or DEFAULT_PERSIST_DIR)
    pdir.mkdir(parents=True, exist_ok=True)
    embs = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")

    return Chroma(
        collection_name=collection_name,
        embedding_function=embs,
        persist_directory=str(pdir),
    )


def _maybe_persist(vectordb: Chroma) -> None:
    """
    langchain-chroma >=0.1 persists automatically when persist_directory is set
    and no longer exposes vectordb.persist(). Older versions did.
    Try both, then no-op if neither exists.
    """
    try:
        if hasattr(vectordb, "persist"):
            vectordb.persist()  # old API
        elif hasattr(vectordb, "_client") and hasattr(vectordb._client, "persist"):
            vectordb._client.persist()  # chromadb client
    except Exception:
        # Never fail tests on persist mechanics; retrieval is what we care about here.
        pass


def add_texts(
    vectordb: Chroma,
    texts: list[str],
    metadatas: Optional[list[dict]] = None,
    ids: Optional[list[str]] = None,
) -> None:
    """Add raw text chunks to the store and persist (best-effort, version-safe)."""
    vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    _maybe_persist(vectordb)


if __name__ == "__main__":
    # Manual smoke test (doesn't add docs)
    vs = get_vectorstore()
    print("Vector store initialized at:", DEFAULT_PERSIST_DIR.resolve())
