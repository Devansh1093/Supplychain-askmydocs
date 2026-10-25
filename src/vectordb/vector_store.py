from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings.embedder import create_embeddings


def open_store(directory: Path, collection: str, model_name: str) -> Chroma:
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection,
        persist_directory=str(directory),
        embedding_function=create_embeddings(model_name),
    )
