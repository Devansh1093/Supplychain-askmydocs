from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb

from src.chunking.chunker import TokenChunker
from src.config import resolve_path
from src.embeddings.embedder import create_embeddings
from src.ingestion.loader import load_pdf, read_source_manifest
from src.utils.hashing import file_sha256, stable_hash

logger = logging.getLogger(__name__)


def _read_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return fallback
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def ingest_all(config: dict[str, Any], accept_report: bool = False) -> dict[str, Any]:
    root = Path(config["_root"])
    data_dir = resolve_path(config, "data_dir")
    source_path = resolve_path(config, "source_manifest")
    index_manifest_path = resolve_path(config, "index_manifest")
    sources = read_source_manifest(source_path)
    if not sources:
        raise RuntimeError(f"No source PDFs configured in {source_path}")
    listed_files = {str(row["file"]) for row in sources}
    actual_pdfs = {path.name for path in data_dir.glob("*.pdf")}
    unlisted = sorted(actual_pdfs - listed_files)
    missing = sorted(listed_files - actual_pdfs)
    if unlisted or missing:
        raise RuntimeError(f"Source manifest mismatch; unlisted PDFs={unlisted}, missing PDFs={missing}")

    previous = _read_json(index_manifest_path, {})
    chunk_config = config["chunking"]
    ingest_config = config["ingestion"]
    fingerprint_data = {
        "parser_version": ingest_config["parser_version"],
        "chunk_config": chunk_config,
        "embed_model": config["models"]["embedding"],
    }
    global_fingerprint = stable_hash(fingerprint_data)
    full_rebuild = not previous or previous.get("settings_hash") != global_fingerprint

    report: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(), "documents": []}
    prepared: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str]] = {}
    old_docs = previous.get("documents", {})
    for source_index, source in enumerate(sources, start=1):
        path = data_dir / source["file"]
        print(f"[ingest] Extracting {source_index}/{len(sources)}: {path.name}", flush=True)
        if not path.is_file():
            raise RuntimeError(f"Source PDF not found: {path}")
        current_hash = file_sha256(path)
        declared_hash = source.get("sha256")
        if declared_hash and str(declared_hash).lower() != current_hash:
            logger.warning("Source manifest hash mismatch for %s; current file will be re-ingested", path.name)
        pages, stats = load_pdf(path, source, config)
        report["documents"].append(stats)
        old = old_docs.get(source["doc_id"], {})
        changed = full_rebuild or old.get("sha256") != current_hash
        if changed:
            prepared[source["doc_id"]] = (source, pages, current_hash)
            print(f"[ingest] {path.name}: extracted {len(pages)} pages; scheduled for indexing", flush=True)
        else:
            print(f"[ingest] {path.name}: unchanged; keeping indexed chunks", flush=True)

    report_path = resolve_path(config, "extraction_report")
    _write_json(report_path, report)
    flagged = [doc for doc in report["documents"] if doc["flagged"]]
    if flagged and not accept_report:
        names = ", ".join(doc["file"] for doc in flagged)
        raise RuntimeError(
            f"Extraction quality checks flagged {names}. Review {report_path} and pass --accept-report "
            "only after reviewing the sample pages. No index changes were made."
        )

    collection_name = previous.get("active_collection", config["paths"]["chroma_collection"])
    if full_rebuild:
        collection_name = f"{config['paths']['chroma_collection']}_{global_fingerprint[:10]}"
    chroma_dir = resolve_path(config, "chroma_dir")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    if full_rebuild:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    collection = client.get_or_create_collection(collection_name)
    embeddings = create_embeddings(config["models"]["embedding"])
    chunker = TokenChunker(
        config["models"]["embedding"], int(chunk_config["target_tokens"]), int(chunk_config["overlap_tokens"])
    )
    output_docs: dict[str, dict[str, Any]] = {} if full_rebuild else dict(old_docs)
    source_ids = {str(row["doc_id"]) for row in sources}
    for removed_id in set(output_docs) - source_ids:
        collection.delete(where={"doc_id": removed_id})
        output_docs.pop(removed_id, None)

    for doc_id, (source, pages, current_hash) in prepared.items():
        collection.delete(where={"doc_id": doc_id})
        chunks = chunker.split_pages(pages, source)
        print(f"[ingest] {source['file']}: embedding {len(chunks)} chunks", flush=True)
        if chunks:
            vectors = embeddings.embed_documents([chunk.page_content for chunk in chunks])
            collection.add(
                ids=[chunk.metadata["chunk_id"] for chunk in chunks],
                embeddings=vectors,
                documents=[chunk.page_content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
            )
        print(f"[ingest] {source['file']}: indexed {len(chunks)} chunks", flush=True)
        output_docs[doc_id] = {
            "file": source["file"], "sha256": current_hash,
            "parser_version": ingest_config["parser_version"],
            "chunk_config_hash": stable_hash(chunk_config),
            "embed_model": config["models"]["embedding"], "n_chunks": len(chunks),
        }

    stored = collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document
    from src.retrieval.bm25 import BM25Retriever

    all_docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(stored.get("documents", []), stored.get("metadatas", []), strict=True)
    ]
    bm25 = BM25Retriever(all_docs)
    bm25.save(resolve_path(config, "bm25_index"))
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "active_collection": collection_name,
        "settings_hash": global_fingerprint,
        "parser_version": ingest_config["parser_version"],
        "chunk_config_hash": stable_hash(chunk_config),
        "embed_model": config["models"]["embedding"],
        "index_content_hash": stable_hash({doc_id: entry["sha256"] for doc_id, entry in output_docs.items()}),
        "documents": output_docs,
    }
    _write_json(index_manifest_path, manifest)
    logger.info("Ingest complete: %s chunks across %s documents", len(all_docs), len(output_docs))
    return {"documents": len(output_docs), "chunks": len(all_docs), "collection": collection_name, "report": str(report_path)}
