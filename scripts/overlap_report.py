from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.vectordb.vector_store import open_store


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(text: str, width: int) -> set[str]:
    tokens = _tokens(text)
    return {" ".join(tokens[index:index + width]) for index in range(max(0, len(tokens) - width + 1))}


def _signature(shingles: set[str], permutations: int) -> tuple[int, ...]:
    if not shingles:
        return tuple([2**64 - 1] * permutations)
    return tuple(
        min(int.from_bytes(hashlib.blake2b(f"{seed}:{item}".encode(), digest_size=8).digest(), "big") for item in shingles)
        for seed in range(permutations)
    )


def _pair_id(left: str, right: str) -> str:
    first, second = sorted((left, right))
    return f"{first}|{second}"


def build_report() -> dict[str, Any]:
    config = load_config()
    manifest_path = resolve_path(config, "index_manifest")
    if not manifest_path.exists():
        raise RuntimeError("No ingested index found. Run `python main.py ingest` first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store = open_store(resolve_path(config, "chroma_dir"), manifest["active_collection"], config["models"]["embedding"])
    collection = store._collection
    payload = collection.get(include=["documents", "metadatas", "embeddings"])
    texts = payload.get("documents", [])
    metas = payload.get("metadatas", [])
    ids = payload.get("ids", [])
    embeddings = np.asarray(payload.get("embeddings", []), dtype=np.float32)
    rows = [{"chunk_id": ids[i], "text": texts[i], **metas[i]} for i in range(len(ids))]
    threshold = float(config["thresholds"]["overlap_sim"])
    overlap_config = config["overlap"]
    preview_chars = int(overlap_config["preview_chars"])
    semantic_best: dict[str, dict[str, Any]] = {}
    ids_by_doc: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        ids_by_doc[str(row["doc_id"])].append(index)
    count = len(rows)
    candidate_count = min(int(overlap_config["semantic_candidates"]), max(1, count - 1))
    for doc_id, indices in ids_by_doc.items():
        if len(ids_by_doc) < 2 or not indices:
            continue
        nearest = collection.query(
            query_embeddings=[embeddings[index].tolist() for index in indices],
            n_results=candidate_count,
            where={"doc_id": {"$ne": doc_id}},
            include=["distances", "metadatas", "documents"],
        )
        for local_index, query_index in enumerate(indices):
            found_ids = nearest.get("ids", [[]])[local_index]
            if not found_ids:
                continue
            distance = float(nearest["distances"][local_index][0])
            neighbor_meta = nearest["metadatas"][local_index][0]
            neighbor_text = nearest["documents"][local_index][0]
            cosine = 1.0 - distance / 2.0
            if cosine >= threshold:
                left, right = rows[query_index], {"chunk_id": found_ids[0], "text": neighbor_text, **neighbor_meta}
                pair_id = _pair_id(left["chunk_id"], right["chunk_id"])
                semantic_best[pair_id] = {
                    "pair_id": pair_id, "kind": "semantic", "similarity": cosine,
                    "left": {"chunk_id": left["chunk_id"], "file": left["file"], "text_preview": left["text"][:preview_chars]},
                    "right": {"chunk_id": right["chunk_id"], "file": right["file"], "text_preview": right["text"][:preview_chars]},
                }

    width = int(overlap_config["shingle_words"])
    permutations = int(overlap_config["minhash_permutations"])
    bands = int(overlap_config["minhash_bands"])
    if permutations % bands:
        raise ValueError("overlap.minhash_permutations must be divisible by overlap.minhash_bands")
    rows_per_band = permutations // bands
    signatures: dict[int, tuple[int, ...]] = {}
    shingle_sets: dict[int, set[str]] = {}
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        shingles = _shingles(row["text"], width)
        if not shingles:
            continue
        signature = _signature(shingles, permutations)
        signatures[index], shingle_sets[index] = signature, shingles
        for band in range(bands):
            start = band * rows_per_band
            buckets[(band, signature[start:start + rows_per_band])].append(index)
    candidate_pairs: set[tuple[int, int]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for offset, left_index in enumerate(bucket):
            for right_index in bucket[offset + 1:]:
                if rows[left_index]["doc_id"] != rows[right_index]["doc_id"]:
                    candidate_pairs.add(tuple(sorted((left_index, right_index))))
    lexical_pairs = []
    for left_index, right_index in candidate_pairs:
        left_set, right_set = shingle_sets[left_index], shingle_sets[right_index]
        similarity = len(left_set & right_set) / len(left_set | right_set)
        if similarity >= threshold:
            left, right = rows[left_index], rows[right_index]
            pair_id = _pair_id(left["chunk_id"], right["chunk_id"])
            lexical_pairs.append({
                "pair_id": pair_id, "kind": "near_verbatim", "similarity": similarity,
                "left": {"chunk_id": left["chunk_id"], "file": left["file"], "text_preview": left["text"][:preview_chars]},
                "right": {"chunk_id": right["chunk_id"], "file": right["file"], "text_preview": right["text"][:preview_chars]},
            })

    topics = {}
    for topic, terms in overlap_config["topic_terms"].items():
        topics[topic] = {
            doc_id: sum(1 for row in rows if row["doc_id"] == doc_id and any(term.lower() in row["text"].lower() for term in terms))
            for doc_id in ids_by_doc
        }
    label_path = resolve_path(config, "overlap_labels")
    labels = json.loads(label_path.read_text(encoding="utf-8")) if label_path.exists() else {"labels": {}}
    pairs = list(semantic_best.values()) + lexical_pairs
    by_id = {row["pair_id"]: row for row in pairs}
    for pair_id, label in labels.get("labels", {}).items():
        if pair_id in by_id:
            by_id[pair_id]["human_label"] = label
    pairs = sorted(by_id.values(), key=lambda row: row["similarity"], reverse=True)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(), "index_settings_hash": manifest["settings_hash"],
        "threshold": threshold, "chunks": len(rows), "topic_counts": topics,
        "pairs": pairs[: int(overlap_config["top_pairs"])],
        "label_file": str(label_path.relative_to(ROOT)),
        "label_schema": {"labels": {"<pair_id>": "duplicate | paraphrase | conflict"}},
    }


def main() -> int:
    report = build_report()
    report_dir = ROOT / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "overlap_report.json"
    md_path = report_dir / "overlap_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# PDF overlap review", "", f"Chunks inspected: {report['chunks']}", f"Similarity threshold: {report['threshold']}", "", "Manually set labels in `overlap_labels.json` using the pair IDs below.", ""]
    for pair in report["pairs"]:
        lines.extend([f"## {pair['pair_id']}", f"- Similarity: {pair['similarity']:.3f} ({pair['kind']})", f"- Current label: {pair.get('human_label', 'unreviewed')}", f"- A — `{pair['left']['file']}` / `{pair['left']['chunk_id']}`: {pair['left']['text_preview']}", f"- B — `{pair['right']['file']}` / `{pair['right']['chunk_id']}`: {pair['right']['text_preview']}", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
