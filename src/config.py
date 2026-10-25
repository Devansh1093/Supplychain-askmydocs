from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.utils.hashing import stable_hash


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid configuration file: {config_path}")
    config["_root"] = str(ROOT)
    return config


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config["paths"][key])
    return path if path.is_absolute() else Path(config["_root"]) / path


def retrieval_settings_hash(config: dict[str, Any]) -> str:
    labels_path = resolve_path(config, "overlap_labels")
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {"labels": {}}
    return stable_hash({"retrieval": config["retrieval"], "overlap_labels": labels})


def require_calibrated_threshold(config: dict[str, Any], mode: str = "hybrid_rerank") -> float:
    thresholds = config.get("thresholds", {})
    if mode == "hybrid_rerank":
        threshold = thresholds.get("refuse_threshold")
        metadata = thresholds.get("calibration")
    else:
        threshold = thresholds.get("mode_thresholds", {}).get(mode)
        metadata = thresholds.get("mode_calibrations", {}).get(mode)
    if threshold is None or not isinstance(metadata, dict):
        raise RuntimeError(
            "refuse_threshold is uncalibrated. Curate data/testing.jsonl and run "
            "`python eval/run_eval.py --calibrate` using its dev split first."
        )
    try:
        numeric_threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("thresholds.refuse_threshold must be numeric") from exc
    try:
        recorded_threshold = float(metadata.get("threshold", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Calibration record does not contain a numeric threshold; recalibrate it.") from exc
    if recorded_threshold != numeric_threshold:
        raise RuntimeError("Configured refusal threshold differs from its calibration record; recalibrate it.")
    if metadata.get("reranker_model") != config["models"]["reranker"]:
        raise RuntimeError("Reranker model changed since calibration; recalibrate the refusal threshold.")
    if metadata.get("mode") != mode:
        raise RuntimeError(f"The refusal threshold is not calibrated for retrieval mode '{mode}'.")
    dataset_path = resolve_path(config, "testing_jsonl")
    if not dataset_path.exists():
        raise RuntimeError("Evaluation dataset is missing; refusal threshold calibration cannot be verified.")
    dev_rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("split") == "dev":
                dev_rows.append(row)
    if stable_hash(dev_rows) != metadata.get("dev_set_hash"):
        raise RuntimeError("Dev evaluation set changed since threshold calibration; recalibrate it.")
    manifest_path = resolve_path(config, "index_manifest")
    if not manifest_path.exists():
        raise RuntimeError("Index manifest is missing; ingest source documents before asking or evaluating.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("settings_hash") != metadata.get("index_settings_hash"):
        raise RuntimeError("Embedding, parser, or chunk settings changed since calibration; recalibrate the threshold.")
    if manifest.get("index_content_hash") != metadata.get("index_content_hash"):
        raise RuntimeError("Indexed source documents changed since calibration; recalibrate the threshold.")
    if retrieval_settings_hash(config) != metadata.get("retrieval_settings_hash"):
        raise RuntimeError("Retrieval settings or duplicate labels changed since calibration; recalibrate the threshold.")
    return numeric_threshold
