from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

import hashlib
import math

from agents.vectorstore import get_vectorstore
from agents.ingest import ingest_texts, chunk_texts, ChunkingConfig


class MockEmbeddings(Embeddings):
    """Deterministic 2D embeddings based on length and keyword presence; avoids network calls."""

    def _vec(self, text: str) -> List[float]:
        # Stable, deterministic hash-based embedding over character bigrams
        s = "".join(ch for ch in text.lower() if ch.isalpha())
        buckets = 32
        v = [0.0] * buckets
        for i in range(len(s) - 1):
            bg = s[i : i + 2]
            h = int.from_bytes(hashlib.md5(bg.encode("utf-8")).digest()[:4], "little")
            v[h % buckets] += 1.0
        # L2 normalize to make cosine/dot product focus on overlap, not length
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def test_chunking_shapes():
    texts = ["Para1.\n\nPara2 is longer. " * 20]
    chunks = chunk_texts(texts, ChunkingConfig(chunk_size=200, chunk_overlap=40))
    assert all(80 <= len(c) <= 260 for c in chunks)  # rough bounds
    assert any("\n\n" in c or "\n" in c for c in chunks)  # prefers boundaries


def test_ingest_and_retrieve(tmp_path: Path):
    vs = get_vectorstore(
        persist_directory=tmp_path / "vs",
        embeddings=MockEmbeddings(),
        collection_name="ingest_test",
    )

    corpus = [
        "Python is a high-level, general-purpose programming language. "
        "It emphasizes code readability with the use of significant indentation.",
        "LangChain provides a standard interface for chains, enables composition, and "
        "integrates with numerous tools for building LLM-powered applications.",
        "Chroma is a local vector database well-suited for development workflows.",
    ]

    num_docs, num_chunks = ingest_texts(
        vs, corpus, ChunkingConfig(chunk_size=120, chunk_overlap=30)
    )
    assert num_docs == len(corpus)
    assert num_chunks >= len(corpus)  # chunking shouldn't reduce count

    hits = vs.similarity_search("What is LangChain used for?", k=2)
    assert len(hits) >= 1
    assert any("LangChain" in d.page_content for d in hits)
