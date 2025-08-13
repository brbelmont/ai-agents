from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .vectorstore import add_texts  # from Step 1


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking knobs. Tweak per corpus and retrieval quality."""

    chunk_size: int = 800
    chunk_overlap: int = 160


def make_splitter(cfg: ChunkingConfig | None = None) -> RecursiveCharacterTextSplitter:
    cfg = cfg or ChunkingConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],  # prefer semantic boundaries first
    )


def chunk_texts(texts: Sequence[str], cfg: ChunkingConfig | None = None) -> List[str]:
    """Split input docs into retrieval-friendly chunks"""
    splitter = make_splitter(cfg)
    out: List[str] = []
    for t in texts:
        if not t:
            continue
        out.extend(splitter.split_text(t))
    return out


def ingest_texts(
    vectordb: Chroma,
    texts: Sequence[str],
    cfg: ChunkingConfig | None = None,
    base_metadata: Optional[dict] = None,
    id_prefix: str = "doc",
) -> Tuple[int, int]:
    """
    Chunk and add texts to the store.

    Returns:
        (num_source_docs, num_chunks_added)
    """

    chunks = chunk_texts(texts, cfg)
    metadatas = []
    for i in range(len(chunks)):
        md = dict(base_metadata) if base_metadata else {}
        if base_metadata and "filename" in base_metadata:
            md["source"] = f"{base_metadata['filename']}:{i}"
        else:
            md["source"] = f"{id_prefix}:{i}"
        metadatas.append(md)
    ids = [f"{id_prefix}-{i}" for i in range(len(chunks))]

    add_texts(vectordb, chunks, metadatas=metadatas, ids=ids)
    return (len(texts), len(chunks))
