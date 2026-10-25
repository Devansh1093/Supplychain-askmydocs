from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document


def dense_retrieve(store: Chroma, query: str, k: int) -> list[tuple[Document, float]]:
    return [(doc, float(score)) for doc, score in store.similarity_search_with_score(query, k=k)]
