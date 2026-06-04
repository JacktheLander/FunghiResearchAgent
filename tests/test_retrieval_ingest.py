from __future__ import annotations

from pathlib import Path

from fungi_rag.config import Settings
from fungi_rag.embeddings import HashingEmbeddingBackend
from fungi_rag.ingest import DocumentIngestor
from fungi_rag.retrieval import ChunkRepository, HybridRetriever


def settings_for(tmp_path: Path) -> Settings:
    settings = Settings(
        embedding_backend="hashing",
        chroma_dir=tmp_path / "chroma",
        upload_dir=tmp_path / "uploads",
        source_raw_dir=tmp_path / "sources",
        source_state_path=tmp_path / "sources.jsonl",
        index_dir=tmp_path / "index",
        output_dir=tmp_path / "outputs",
        chunk_size=300,
        chunk_overlap=40,
    )
    settings.ensure_directories()
    return settings


def test_ingest_chunks_and_hybrid_retrieval(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    source = tmp_path / "data" / "fungi.txt"
    source.parent.mkdir()
    source.write_text(
        """
        # Fungal morphology

        Hyphae form threadlike filaments. A mycelium expands through a substrate and
        helps fungi absorb nutrients by increasing surface area.

        # Decomposition

        Saprotrophic fungi decompose lignin and cellulose, returning nutrients to soil.
        """,
        encoding="utf-8",
    )
    settings = settings_for(tmp_path)
    chunks = DocumentIngestor(settings).ingest_path(source)
    assert chunks

    stored = ChunkRepository(settings.index_dir).load_chunks()
    assert stored[0].local_path
    assert not Path(stored[0].local_path).is_absolute()
    retriever = HybridRetriever(
        stored,
        embeddings=HashingEmbeddingBackend(),
        settings=settings,
        prefer_chroma=False,
    )
    packet, trace = retriever.retrieve("How do hyphae help nutrient absorption?", top_k=2)
    assert packet.items
    assert "hyphae" in packet.items[0].snippet.lower()
    assert trace.vector_candidates or trace.keyword_candidates


def test_retriever_normalizes_funghi_query(tmp_path: Path) -> None:
    source = tmp_path / "ecology.txt"
    source.write_text("Fungi are important decomposers in forest nutrient cycles.", encoding="utf-8")
    settings = settings_for(tmp_path)
    DocumentIngestor(settings).ingest_path(source)
    chunks = ChunkRepository(settings.index_dir).load_chunks()
    retriever = HybridRetriever(
        chunks,
        embeddings=HashingEmbeddingBackend(),
        settings=settings,
        prefer_chroma=False,
    )
    packet, _trace = retriever.retrieve("funghi decomposers", top_k=1)
    assert packet.normalized_query == "fungi decomposers"
    assert packet.items
