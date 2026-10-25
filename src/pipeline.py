from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import require_calibrated_threshold
from src.llm.guard import REFUSAL, guard_response
from src.llm.llm_client import OllamaClient
from src.prompts.prompt_templates import build_messages
from src.retrieval.service import RetrievalService
from src.utils.logging import configure_logging

logger = logging.getLogger(__name__)


class SupplyDocsPipeline:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        log_path = Path(config["_root"]) / config["paths"]["log_file"]
        configure_logging(log_path, config["logging"]["level"])
        self.thresholds = {
            mode: require_calibrated_threshold(config, mode)
            for mode in ("dense", "hybrid", "hybrid_rerank")
        }
        self.retrieval = RetrievalService(config)
        self.llm = OllamaClient(config)

    def ask(self, query: str, mode: str = "hybrid_rerank") -> dict[str, Any]:
        response, _, _ = self.ask_with_retrieval(query, mode)
        return response

    def ask_with_retrieval(self, query: str, mode: str = "hybrid_rerank") -> tuple[dict[str, Any], list[Any], list[Any]]:
        if mode not in self.thresholds:
            raise ValueError(f"Unknown retrieval mode: {mode}")
        threshold = self.thresholds[mode]
        if not query.strip():
            raise ValueError("query must not be empty")
        candidates, output, scored = self.retrieval.candidates(query, mode)
        log_evidence = [
            {"chunk_id": doc.metadata.get("chunk_id"), "file": doc.metadata.get("file"), "reranker_score": score}
            for doc, score in scored
        ]
        gate_score = self.retrieval.gate_score(output, scored)
        if gate_score is None or gate_score < threshold:
            logger.info("query=%r refused_by_gate retrieved=%s", query, log_evidence)
            return {"answer": REFUSAL, "citations": [], "refused": True}, output, scored
        docs = [doc for doc, _ in output]
        logger.info("query=%r mode=%s retrieved=%s", query, mode, log_evidence)
        if not docs:
            return {"answer": REFUSAL, "citations": [], "refused": True}, output, scored
        payload = self.llm.generate_json(build_messages(query, docs))
        guarded = guard_response(payload, docs)
        logger.info("query=%r refused=%s", query, guarded["refused"])
        return {key: guarded[key] for key in ("answer", "citations", "refused")}, output, scored
