from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from src.config import resolve_path
from src.ingestion.loader import read_source_manifest
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.retriever import dense_retrieve
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.hashing import file_sha256
from src.utils.hashing import stable_hash
from src.vectordb.vector_store import open_store

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, config: dict[str, Any], verify_freshness: bool = True) -> None:
        self.config = config
        manifest_path = resolve_path(config, "index_manifest")
        if not manifest_path.exists():
            raise RuntimeError("No index manifest found. Run `python main.py ingest` first.")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._check_freshness(verify_freshness)
        self.store = open_store(
            resolve_path(config, "chroma_dir"), self.manifest["active_collection"], config["models"]["embedding"]
        )
        bm25_path = resolve_path(config, "bm25_index")
        if not bm25_path.exists():
            raise RuntimeError("BM25 index is missing. Run `python main.py ingest` first.")
        self.bm25 = BM25Retriever.load(bm25_path)
        self.reranker = CrossEncoderReranker(config["models"]["reranker"], int(config["retrieval"]["reranker_batch_size"]))

    def _check_freshness(self, enabled: bool) -> None:
        if not enabled:
            return
        data_dir = resolve_path(self.config, "data_dir")
        sources = read_source_manifest(resolve_path(self.config, "source_manifest"))
        listed_files = {str(source["file"]) for source in sources}
        actual_files = {path.name for path in data_dir.glob("*.pdf")}
        if listed_files != actual_files:
            raise RuntimeError("Source inventory changed; update data/sources.yaml and run `python main.py ingest`.")
        expected_settings = stable_hash({
            "parser_version": self.config["ingestion"]["parser_version"],
            "chunk_config": self.config["chunking"],
            "embed_model": self.config["models"]["embedding"],
        })
        if expected_settings != self.manifest.get("settings_hash"):
            raise RuntimeError("Parser, chunking, or embedding settings changed; run `python main.py ingest`.")
        if set(self.manifest.get("documents", {})) != {str(source["doc_id"]) for source in sources}:
            raise RuntimeError("Index manifest does not match data/sources.yaml; run `python main.py ingest`.")
        for source in sources:
            file_path = data_dir / source["file"]
            indexed = self.manifest.get("documents", {}).get(source["doc_id"])
            if not file_path.is_file() or not indexed:
                raise RuntimeError(f"Source/index mismatch for {source['file']}; run `python main.py ingest`.")
            if file_sha256(file_path) != indexed.get("sha256"):
                raise RuntimeError(f"{source['file']} changed since ingestion; run `python main.py ingest`.")

    def candidates(self, query: str, mode: str) -> tuple[list[Document], list[tuple[Document, float]], list[tuple[Document, float]]]:
        settings = self.config["retrieval"]
        fetch = max(int(settings["rerank_in"]), int(settings["final_k"]))
        dense = dense_retrieve(self.store, query, max(int(settings["dense_k"]), fetch))
        if mode == "dense":
            ranked = dense[:fetch]
        elif mode in {"hybrid", "hybrid_rerank"}:
            keyword = self.bm25.retrieve(query, max(int(settings["bm25_k"]), fetch))
            ranked = reciprocal_rank_fusion(dense, keyword, int(settings["rrf_k"]), fetch)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        candidate_docs = [doc for doc, _ in ranked[:fetch]]
        scored = self.reranker.score(query, candidate_docs)
        output = scored if mode == "hybrid_rerank" else ranked
        if mode == "hybrid_rerank":
            output = self._collapse_confirmed_duplicates(output)[: int(settings["final_k"])]
        else:
            output = output[: int(settings["final_k"])]
        return candidate_docs, output, scored

    def _collapse_confirmed_duplicates(self, ranked: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        labels_path = resolve_path(self.config, "overlap_labels")
        if not labels_path.exists():
            return ranked
        label_data = json.loads(labels_path.read_text(encoding="utf-8"))
        labels = label_data.get("labels", {})
        kept: list[tuple[Document, float]] = []
        for doc, score in ranked:
            duplicate_of: tuple[Document, float] | None = None
            for earlier_doc, earlier_score in kept:
                first, second = sorted((str(doc.metadata["chunk_id"]), str(earlier_doc.metadata["chunk_id"])))
                if labels.get(f"{first}|{second}") == "duplicate":
                    duplicate_of = (earlier_doc, earlier_score)
                    break
            if duplicate_of:
                winner = duplicate_of[0]
                aliases = list(winner.metadata.get("duplicate_files", []))
                aliases.extend([str(doc.metadata["file"]), *doc.metadata.get("duplicate_files", [])])
                winner.metadata["duplicate_files"] = list(dict.fromkeys(aliases))
                continue
            kept.append((doc, score))
        return kept

    def all_chunks(self) -> list[Document]:
        payload = self.store._collection.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(payload.get("documents", []), payload.get("metadatas", []), strict=True)
        ]

    @staticmethod
    def gate_score(output: list[tuple[Document, float]], reranked: list[tuple[Document, float]]) -> float | None:
        scores = {str(doc.metadata.get("chunk_id")): score for doc, score in reranked}
        selected_scores = [scores[str(doc.metadata.get("chunk_id"))] for doc, _ in output if str(doc.metadata.get("chunk_id")) in scores]
        return max(selected_scores) if selected_scores else None

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        result = self.store._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result.get("ids"):
            return None
        return {"chunk_id": result["ids"][0], "text": result["documents"][0], "metadata": result["metadatas"][0]}
