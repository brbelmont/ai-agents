from pathlib import Path
from typing import List
from langchain_core.embeddings import Embeddings

from agents.vectorstore import get_vectorstore, add_texts


class MockEmbeddings(Embeddings):
    """
    Deterministic, offline embedding. Dimension=1; value based on length.
    """

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t) + 1)] for t in texts]  # avoid zero-vector

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text) + 1)]


def test_vectorstore_init_and_retrieval(tmp_path: Path):
    vs = get_vectorstore(
        persist_directory=tmp_path / "vs",
        embeddings=MockEmbeddings(),
        collection_name="test_docs",
    )

    texts = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet",
        "pythin is great for building AI agents",
    ]
    add_texts(vs, texts)

    docs = vs.similarity_search("fox", k=2)
    assert len(docs) == 2
    assert all(hasattr(d, "page_content") for d in docs)
