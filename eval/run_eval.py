from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path, require_calibrated_threshold, retrieval_settings_hash
from src.embeddings.embedder import create_embeddings
from src.pipeline import SupplyDocsPipeline
from src.retrieval.service import RetrievalService
from src.utils.hashing import file_sha256, stable_hash

logger = logging.getLogger(__name__)


def load_cases(path: Path, split: str) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Evaluation file not found: {path}. Curate the golden set before evaluation.")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        required = {"id", "question", "reference_answer", "reference_contexts", "answerable", "type", "source_docs", "reference_sources", "split", "notes"}
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
        if not isinstance(row["reference_contexts"], list) or not all(isinstance(item, str) for item in row["reference_contexts"]):
            raise ValueError(f"{path}:{line_number} reference_contexts must be a list of text strings")
        if row["answerable"] and not row["reference_contexts"]:
            raise ValueError(f"{path}:{line_number} answerable questions need supporting reference_contexts")
        if row["answerable"] and not row["reference_sources"]:
            raise ValueError(f"{path}:{line_number} answerable questions need reference_sources file/page anchors for retrieval metrics")
        if not row["answerable"] and (row["reference_contexts"] or row["reference_sources"]):
            raise ValueError(f"{path}:{line_number} unanswerable questions must have empty reference contexts and source anchors")
        if row["split"] == split:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No '{split}' examples found in {path}")
    return rows


def _page_matches(chunk: dict[str, Any], reference: dict[str, Any]) -> bool:
    if chunk.get("file") != reference.get("file"):
        return False
    try:
        page = int(reference["pdf_page"])
        return int(chunk.get("pdf_page_start", chunk.get("pdf_page", -1))) <= page <= int(chunk.get("pdf_page_end", chunk.get("pdf_page", -1)))
    except (TypeError, ValueError, KeyError):
        return False


def _text_matches(chunk: dict[str, Any], references: list[str], threshold: float) -> bool:
    text_tokens = set(" ".join(chunk.get("text", "").lower().split()).split())
    for reference in references:
        normalized = " ".join(reference.lower().split())
        candidate = " ".join(chunk.get("text", "").lower().split())
        if normalized and normalized in candidate:
            return True
        tokens = set(normalized.split())
        if tokens and len(tokens & text_tokens) / len(tokens) >= threshold:
            return True
    return False


def _is_relevant(doc: Any, row: dict[str, Any], config: dict[str, Any]) -> bool:
    metadata = doc.metadata
    anchors = row.get("reference_sources", [])
    if anchors:
        return any(_page_matches(metadata, anchor) for anchor in anchors)
    references = row.get("reference_contexts", [])
    return _text_matches({"text": doc.page_content}, references, float(config["eval"]["reference_match_threshold"]))


def _retrieval_scores(retrieved: list[Any], row: dict[str, Any], config: dict[str, Any]) -> dict[str, float | None]:
    if not row["answerable"]:
        return {"hit_at_k": None, "mrr": None}
    relevant = [_is_relevant(doc, row, config) for doc in retrieved]
    k = int(config["retrieval"]["eval_k"])
    hit = float(any(relevant[:k]))
    reciprocal_rank = next((1.0 / index for index, found in enumerate(relevant, start=1) if found), 0.0)
    return {"hit_at_k": hit, "mrr": reciprocal_rank}


def calibrate(config: dict[str, Any]) -> dict[str, Any]:
    dataset_path = resolve_path(config, "testing_jsonl")
    dev = load_cases(dataset_path, "dev")
    answerable_count = sum(bool(row["answerable"]) for row in dev)
    unanswerable_count = len(dev) - answerable_count
    minimum = int(config["eval"]["minimum_dev_examples"])
    per_class = int(config["eval"]["minimum_dev_per_class"])
    if len(dev) < minimum or answerable_count < per_class or unanswerable_count < per_class:
        raise RuntimeError(
            f"Threshold calibration needs at least {minimum} dev examples and {per_class} from each class; "
            f"found {len(dev)} total, {answerable_count} answerable, {unanswerable_count} unanswerable."
        )
    retrieval = RetrievalService(config)
    mode_calibrations: dict[str, dict[str, Any]] = {}
    mode_thresholds: dict[str, float] = {}
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        scores: list[tuple[float, bool]] = []
        for row in dev:
            _, output, reranked = retrieval.candidates(row["question"], mode)
            gate_score = retrieval.gate_score(output, reranked)
            scores.append((gate_score if gate_score is not None else -math.inf, bool(row["answerable"])))
        finite_scores = [score for score, _ in scores if math.isfinite(score)]
        if not finite_scores:
            raise RuntimeError(f"Could not produce reranker scores for dev questions in mode '{mode}'; ingest documents first.")
        epsilon = float(config["eval"]["threshold_epsilon"])
        candidates = sorted({min(finite_scores) - epsilon, max(finite_scores) + epsilon, *finite_scores})
        best: tuple[float, float] | None = None
        target_metric = str(config["eval"]["refusal_target_metric"])
        if target_metric != "balanced_accuracy":
            raise ValueError(f"Unsupported eval.refusal_target_metric: {target_metric}")
        for threshold in candidates:
            true_positive = sum(actual and score >= threshold for score, actual in scores)
            false_negative = sum(actual and score < threshold for score, actual in scores)
            true_negative = sum((not actual) and score < threshold for score, actual in scores)
            false_positive = sum((not actual) and score >= threshold for score, actual in scores)
            sensitivity = true_positive / max(1, true_positive + false_negative)
            specificity = true_negative / max(1, true_negative + false_positive)
            balanced_accuracy = (sensitivity + specificity) / 2
            candidate = (balanced_accuracy, threshold)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        mode_thresholds[mode] = best[1]
        mode_calibrations[mode] = {
            "mode": mode, "date": datetime.now(timezone.utc).date().isoformat(),
            "dev_set_hash": stable_hash(dev), "dev_set_sha256": file_sha256(dataset_path),
            "reranker_model": config["models"]["reranker"],
            "index_settings_hash": retrieval.manifest["settings_hash"],
            "index_content_hash": retrieval.manifest["index_content_hash"],
            "retrieval_settings_hash": retrieval_settings_hash(config),
            "threshold": best[1], "metric": target_metric, "balanced_accuracy": best[0],
            "sample_count": len(dev), "answerable_count": answerable_count, "unanswerable_count": unanswerable_count,
            "score_type": "raw_cross_encoder_logit",
        }
    calibration = mode_calibrations["hybrid_rerank"]
    config_path = ROOT / "config.yaml"
    disk_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    disk_config["thresholds"]["refuse_threshold"] = mode_thresholds["hybrid_rerank"]
    disk_config["thresholds"]["calibration"] = calibration
    disk_config["thresholds"]["mode_thresholds"] = mode_thresholds
    disk_config["thresholds"]["mode_calibrations"] = mode_calibrations
    config_path.write_text(yaml.safe_dump(disk_config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps(calibration, indent=2))
    return calibration


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def _package_versions() -> dict[str, str]:
    packages = ["langchain", "langchain-core", "langchain-community", "langchain-chroma", "langchain-ollama", "chromadb", "sentence-transformers", "transformers", "rank-bm25", "PyMuPDF", "ragas", "datasets", "PyYAML", "fastapi"]
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _ragas_scores(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    answerable = [row for row in rows if row["answerable"]]
    if not answerable:
        return []
    from datasets import Dataset
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache
    from langchain_ollama import ChatOllama
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    cache_path = resolve_path(config, "cache_dir") / "judge.sqlite"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=str(cache_path)))
    llm_config = config["llm"]
    if llm_config.get("provider") != "ollama":
        raise ValueError("llm.provider must be 'ollama' for Ragas evaluation")
    judge = ChatOllama(
        model=config["eval"]["judge_model"],
        base_url=llm_config["base_url"],
        temperature=0,
    )
    dataset = Dataset.from_list([
        {"question": row["question"], "answer": row["answer"], "contexts": row["contexts"], "ground_truth": row["reference_answer"]}
        for row in answerable
    ])
    result = evaluate(
        dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge, embeddings=create_embeddings(config["models"]["embedding"]),
        run_config=RunConfig(
            max_workers=int(config["eval"]["max_workers"]),
            max_retries=int(config["eval"]["max_retries"]),
            max_wait=int(config["eval"]["max_wait_s"]),
            timeout=int(config["eval"]["timeout_s"]),
            seed=int(config["eval"]["random_seed"]),
        ),
    )
    frame = result.to_pandas()
    values = []
    for _, series in frame.iterrows():
        values.append({key: (None if not math.isfinite(float(value)) else float(value)) for key, value in series.items() if key in {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}})
    return values


def run_evaluation(config: dict[str, Any], mode: str) -> Path:
    threshold_used = require_calibrated_threshold(config, mode)
    dataset_path = resolve_path(config, "testing_jsonl")
    cases = load_cases(dataset_path, "test")
    pipeline = SupplyDocsPipeline(config)
    result_rows = []
    for case in cases:
        response, output, scored = pipeline.ask_with_retrieval(case["question"], mode)
        retrieved_docs = [doc for doc, _ in output]
        gate_score = pipeline.retrieval.gate_score(output, scored)
        retrieval_metrics = _retrieval_scores(retrieved_docs, case, config)
        result_rows.append({
            "id": case["id"], "question": case["question"], "answerable": case["answerable"],
            "answer": response["answer"], "refused": response["refused"],
            "reference_answer": case["reference_answer"], "reference_contexts": case["reference_contexts"],
            "reranker_gate_score": gate_score,
            "contexts": [doc.page_content for doc in retrieved_docs],
            "retrieved": [{"chunk_id": doc.metadata["chunk_id"], "file": doc.metadata["file"], "pdf_page": doc.metadata["pdf_page"], "score": score}
                          for doc, score in output],
            **retrieval_metrics,
        })
    ragas = _ragas_scores(result_rows, config)
    cursor = 0
    for row in result_rows:
        if row["answerable"] and cursor < len(ragas):
            row["ragas"] = ragas[cursor]
            cursor += 1
        else:
            row["ragas"] = None
    outcomes = [row for row in result_rows if row["answerable"] is not None]
    correct_refusal = sum(bool(row["refused"]) == (not bool(row["answerable"])) for row in outcomes)
    summary = {
        "mode": mode, "count": len(result_rows),
        "refusal_accuracy": correct_refusal / max(1, len(outcomes)),
        "hit_at_k": _mean([row["hit_at_k"] for row in result_rows]),
        "mrr": _mean([row["mrr"] for row in result_rows]),
        "ragas": {metric: _mean([row.get("ragas", {}).get(metric) for row in result_rows if row.get("ragas")]) for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")},
    }
    config_for_hash = {key: value for key, value in config.items() if not key.startswith("_")}
    run_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    sha = _git_sha()
    output_path = ROOT / "eval" / "results" / f"{run_date}_{sha}_{mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "git_sha": sha,
        "python_version": platform.python_version(),
        "config_hash": stable_hash(config_for_hash), "dataset_sha256": file_sha256(dataset_path),
        "calibration": (config["thresholds"]["calibration"] if mode == "hybrid_rerank" else config["thresholds"]["mode_calibrations"][mode]), "model_versions": _package_versions(),
        "judge_model": config["eval"]["judge_model"], "refuse_threshold": threshold_used,
        "score_type": "raw_cross_encoder_logit", "summary": summary, "questions": result_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_summary_table(mode, summary, output_path)
    print(f"Saved {output_path}")
    return output_path


def _mean(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def _write_summary_table(mode: str, summary: dict[str, Any], source: Path) -> None:
    table_path = ROOT / "eval" / "results" / "summary.md"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    prior: dict[str, dict[str, Any]] = {}
    if table_path.exists():
        lines = table_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            if line.startswith("|") and "---" not in line and "Mode" not in line:
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if len(cells) >= 10:
                    prior[cells[0]] = {"count": cells[1], "refusal_accuracy": cells[2], "hit": cells[3], "mrr": cells[4], "faithfulness": cells[5], "answer_relevancy": cells[6], "context_precision": cells[7], "context_recall": cells[8], "path": cells[9]}
    prior[mode] = {
        "count": str(summary["count"]), "refusal_accuracy": _fmt(summary["refusal_accuracy"]),
        "hit": _fmt(summary["hit_at_k"]), "mrr": _fmt(summary["mrr"]),
        "faithfulness": _fmt(summary["ragas"]["faithfulness"]),
        "answer_relevancy": _fmt(summary["ragas"]["answer_relevancy"]),
        "context_precision": _fmt(summary["ragas"]["context_precision"]),
        "context_recall": _fmt(summary["ragas"]["context_recall"]),
        "path": source.name,
    }
    lines = ["# Evaluation results", "", "Results are produced against the held-out `test` split. Metrics are per run; compare rows only when dataset, config, embedding/reranker models, and judge model match.", "", "| Mode | Questions | Refusal accuracy | Hit@k | MRR | Faithfulness | Answer relevancy | Context precision | Context recall | Results |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, row in sorted(prior.items()):
        lines.append(f"| {name} | {row['count']} | {row['refusal_accuracy']} | {row['hit']} | {row['mrr']} | {row['faithfulness']} | {row['answer_relevancy']} | {row['context_precision']} | {row['context_recall']} | `{row['path']}` |")
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    digits = int(load_config()["eval"]["score_rounding_digits"])
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dense", "hybrid", "hybrid_rerank"])
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.calibrate:
        calibrate(config)
    elif args.mode:
        run_evaluation(config, args.mode)
    else:
        parser.error("choose --calibrate or --mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
