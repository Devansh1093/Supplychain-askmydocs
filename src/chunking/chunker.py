from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from transformers import AutoTokenizer


class TokenChunker:
    def __init__(self, model_name: str, target_tokens: int, overlap_tokens: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if target_tokens <= overlap_tokens:
            raise ValueError("chunking.target_tokens must be larger than chunking.overlap_tokens")
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens

    def split_pages(self, pages: list[dict[str, Any]], source: dict[str, Any]) -> list[Document]:
        all_ids: list[int] = []
        page_ranges: list[tuple[int, int, int, str | None]] = []
        page_separator = self.tokenizer.encode("\n\n", add_special_tokens=False)
        for page in pages:
            text = page["text"]
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            if all_ids:
                all_ids.extend(page_separator)
            page_ranges.append((len(all_ids), len(all_ids) + len(token_ids), int(page["pdf_page"]), str(page["printed_page"]) if page["printed_page"] else None))
            all_ids.extend(token_ids)
        if not all_ids:
            return []
        documents: list[Document] = []
        start = 0
        part = 0
        step = self.target_tokens - self.overlap_tokens
        while start < len(all_ids):
            end = min(start + self.target_tokens, len(all_ids))
            page_start = next((page_number for left, right, page_number, _ in page_ranges if right > start), 1)
            page_end = next((page_number for left, right, page_number, _ in page_ranges if right >= end), len(pages))
            first_printed = pages[page_start - 1]["printed_page"] if page_start <= len(pages) else None
            last_printed = pages[page_end - 1]["printed_page"] if page_end <= len(pages) else None
            text = self.tokenizer.decode(all_ids[start:end], skip_special_tokens=True).strip()
            if text:
                metadata = {
                    "doc_id": source["doc_id"], "file": source["file"], "title": source.get("title", source["file"]),
                    "pdf_page": page_start,
                    "pdf_page_start": page_start, "pdf_page_end": page_end,
                    "printed_page": first_printed if first_printed == last_printed and first_printed else "",
                    "printed_page_start": first_printed or "", "printed_page_end": last_printed or "",
                    "chunk_id": f"{source['doc_id']}:{part}",
                }
                documents.append(Document(page_content=text, metadata=metadata))
                part += 1
            if end == len(all_ids):
                break
            start += step
        return documents
