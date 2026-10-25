from __future__ import annotations

from collections import defaultdict

from langchain_core.documents import Document


def reciprocal_rank_fusion(
    dense: list[tuple[Document, float]],
    keyword: list[tuple[Document, float]],
    rrf_k: int,
    limit: int,
) -> list[tuple[Document, float]]:
    fused: dict[str, tuple[Document, float]] = {}
    for results in (dense, keyword):
        for rank, (doc, _) in enumerate(results, start=1):
            key = str(doc.metadata.get("chunk_id", ""))
            current = fused.get(key, (doc, 0.0))
            fused[key] = (doc, current[1] + 1.0 / (rrf_k + rank))
    return sorted(fused.values(), key=lambda pair: pair[1], reverse=True)[:limit]
