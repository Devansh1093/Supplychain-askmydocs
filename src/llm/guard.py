from __future__ import annotations

from typing import Any

from langchain_core.documents import Document


REFUSAL = "I can't find support for that in the provided documents."
PARTIAL_NOTICE = "I couldn't find support for the remaining requested details."


def guard_response(payload: dict[str, Any], retrieved: list[Document]) -> dict[str, Any]:
    by_id = {str(doc.metadata.get("chunk_id")): doc for doc in retrieved}
    safe_claims: list[dict[str, Any]] = []
    citations: list[str] = []
    for claim in payload.get("claims", []):
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
            continue
        ids = claim.get("chunk_ids")
        if not isinstance(ids, list) or not ids:
            continue
        valid_ids = [str(chunk_id) for chunk_id in ids if str(chunk_id) in by_id]
        if len(valid_ids) != len(ids):
            continue
        text = claim["text"].strip()
        if not text:
            continue
        files = list(dict.fromkeys(
            name
            for chunk_id in valid_ids
            for name in [str(by_id[chunk_id].metadata["file"]), *by_id[chunk_id].metadata.get("duplicate_files", [])]
        ))
        safe_claims.append({"text": text, "chunk_ids": valid_ids, "files": files})
        citations.extend(files)
    if not safe_claims:
        return {"answer": REFUSAL, "citations": [], "refused": True, "claims": []}
    lines = [f"{claim['text']} {' '.join(f'[{name}]' for name in claim['files'])}" for claim in safe_claims]
    if payload.get("unsupported"):
        lines.append(PARTIAL_NOTICE)
    unique_files = list(dict.fromkeys(citations))
    return {"answer": "\n\n".join(lines), "citations": [{"file": file} for file in unique_files], "refused": False, "claims": safe_claims}
