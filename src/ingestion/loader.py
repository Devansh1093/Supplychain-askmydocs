from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import yaml

from src.utils.hashing import file_sha256


PRINTED_PAGE_PATTERNS = (
    re.compile(r"^\s*(?:page\s+)?(\d{1,4})\s*$", re.IGNORECASE),
    re.compile(r"^\s*[-–—]?\s*(\d{1,4})\s*[-–—]?\s*$"),
)


def read_source_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("data/sources.yaml must contain a 'sources' list")
    return sources


def _printed_page(text: str, scan_lines: int) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[-scan_lines:]:
        for pattern in PRINTED_PAGE_PATTERNS:
            match = pattern.match(line)
            if match:
                return match.group(1)
    return None


def _junk_ratio(text: str) -> float:
    if not text:
        return 0.0
    junk = 0
    for char in text:
        category = unicodedata.category(char)
        if char == "\ufffd" or category in {"Co", "Cc", "Cs"} and char not in "\n\r\t":
            junk += 1
    return junk / len(text)


def _header_footer_candidates(page_texts: list[str], config: dict[str, Any]) -> set[str]:
    if len(page_texts) < int(config["repeat_header_min_pages"]):
        return set()
    counts: Counter[str] = Counter()
    for text in page_texts:
        lines = [re.sub(r"\s+", " ", line).strip().lower() for line in text.splitlines() if line.strip()]
        scan = int(config["repeat_header_scan_lines"])
        for line in lines[:scan] + lines[-scan:]:
            if len(line) >= int(config["repeat_header_min_chars"]):
                counts[line] += 1
    required = max(int(config["repeat_header_min_pages"]), int(len(page_texts) * float(config["repeat_header_fraction"]) + 0.999))
    return {line for line, count in counts.items() if count >= required}


def _strip_repeated_lines(text: str, repeated: set[str]) -> tuple[str, int]:
    kept: list[str] = []
    stripped = 0
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip().lower()
        if normalized and normalized in repeated:
            stripped += 1
        else:
            kept.append(line)
    return "\n".join(kept).strip(), stripped


def load_pdf(path: Path, source: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with fitz.open(path) as pdf:
        page_texts = [page.get_text("text", sort=True) for page in pdf]
    repeated = _header_footer_candidates(page_texts, config["ingestion"])
    pages: list[dict[str, Any]] = []
    stripped_lines = 0
    empty_pages: list[int] = []
    word_counts: list[int] = []
    total_chars = 0
    junk_chars = 0
    page_quality: list[tuple[int, float]] = []
    for index, raw_text in enumerate(page_texts, start=1):
        text, removed = _strip_repeated_lines(raw_text, repeated)
        stripped_lines += removed
        words = len(text.split())
        word_counts.append(words)
        if words < int(config["chunking"]["min_page_words"]):
            empty_pages.append(index)
        total_chars += len(text)
        page_junk_ratio = _junk_ratio(text)
        junk_chars += sum(1 for char in text if char == "\ufffd" or unicodedata.category(char) in {"Co", "Cc", "Cs"} and char not in "\n\r\t")
        page_quality.append((index, page_junk_ratio))
        pages.append({"pdf_page": index, "printed_page": _printed_page(raw_text, int(config["ingestion"]["printed_page_scan_lines"])), "text": text})

    page_count = len(page_texts)
    empty_fraction = len(empty_pages) / page_count if page_count else 1.0
    junk_ratio = junk_chars / total_chars if total_chars else 0.0
    stats = {
        "doc_id": source["doc_id"], "file": path.name, "sha256": file_sha256(path),
        "pages": page_count, "empty_pages": empty_pages,
        "empty_page_count": len(empty_pages),
        "empty_page_fraction": empty_fraction,
        "avg_words_per_page": (sum(word_counts) / page_count) if page_count else 0,
        "junk_glyph_ratio": junk_ratio,
        "header_footer_lines_stripped": stripped_lines,
        "flagged": empty_fraction > float(config["ingestion"]["max_empty_page_fraction"])
        or junk_ratio > float(config["ingestion"]["max_junk_glyph_ratio"]),
        "flag_reasons": [],
    }
    if empty_fraction > float(config["ingestion"]["max_empty_page_fraction"]):
        stats["flag_reasons"].append("empty page fraction exceeds configured limit")
    if junk_ratio > float(config["ingestion"]["max_junk_glyph_ratio"]):
        stats["flag_reasons"].append("junk glyph ratio exceeds configured limit")
    if stats["flagged"]:
        sample_count = int(config["ingestion"]["report_sample_pages"])
        sample_pages = empty_pages[:sample_count] if empty_pages else [page for page, _ in sorted(page_quality, key=lambda item: item[1], reverse=True)[:sample_count]]
        samples = []
        for page_number in sample_pages:
            text = pages[page_number - 1]["text"]
            samples.append({"pdf_page": page_number, "text_preview": text[: int(config["ingestion"]["sample_preview_chars"]) ]})
        stats["review_samples"] = samples
    return pages, stats
