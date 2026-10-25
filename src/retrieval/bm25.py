from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


class BM25Retriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.corpus = [tokenize(doc.page_content) for doc in documents]
        self.index = BM25Okapi(self.corpus) if self.corpus else None

    def retrieve(self, query: str, k: int) -> list[tuple[Document, float]]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:k]
        return [(self.documents[index], float(score)) for index, score in ranked if score > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls([Document(page_content=row["page_content"], metadata=row["metadata"]) for row in payload])
