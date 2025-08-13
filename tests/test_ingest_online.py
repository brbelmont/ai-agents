import os
from dotenv import load_dotenv

load_dotenv()
import pytest
from pathlib import Path

from agents.vectorstore import get_vectorstore
from agents.ingest import ingest_texts, ChunkingConfig

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ONLINE") != "1", reason="set RUN_ONLINE=1 to run this test"
)


def test_ingest_with_openai(tmp_path: Path):
    vs = get_vectorstore(
        persist_directory=tmp_path / "vs"
    )  # uses real OpenAIEmbeddings
    corpus = ["OpenAI embeddings map text to vector space for semantic retrieval."]
    _, chunks = ingest_texts(
        vs, corpus, ChunkingConfig(chunk_size=200, chunk_overlap=20)
    )
    assert chunks >= 1
    hits = vs.similarity_search("semantic search with embeddings", k=1)
    assert len(hits) == 1
