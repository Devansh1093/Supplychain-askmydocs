# SupplyDocsQA

Ask questions over the supply-chain PDFs in `data/`. Answers cite source filenames and refuse when evidence is missing. This repository uses local Ollama generation and evaluation, local dense retrieval, BM25/RRF hybrid retrieval, reranking, a FastAPI API, and the evaluation/calibration workflow. The React UI is deferred until Phase 3 evaluation is completed, as required by `AGENTS.md`.

## Architecture

| Area | Implementation |
| --- | --- |
| Runtime | Python 3.11, LangChain |
| Source documents | PDF files in `data/`, described by `data/sources.yaml` |
| Parsing | PyMuPDF, page-preserving metadata |
| Chunking | Token-counted windows with overlap |
| Embeddings | Local `BAAI/bge-small-en-v1.5` |
| Vector store | Persistent local ChromaDB |
| Keyword retrieval | rank_bm25 |
| Fusion and reranking | Reciprocal Rank Fusion and local cross-encoder |
| Generation | Local Ollama via `langchain-ollama` |
| API | FastAPI |
| Evaluation | Ragas plus retrieval metrics and per-question JSON output |

## Setup

1. Install Python 3.11 and create an environment: `py -3.11 -m venv .venv`.
2. Activate it and install packages: `pip install -r requirements.txt`.
3. Install Ollama, start it, and pull the configured model: `ollama pull qwen2.5:7b`.
4. Review `config.yaml` and `data/sources.yaml`.
5. Ingest the PDFs: `python main.py ingest`. After a successful ingest or source change, run `python scripts/overlap_report.py` and review its flagged pairs.
6. Curate `data/testing.jsonl` using the schema in `AGENTS.md`; evaluation intentionally refuses to run against an empty or uncalibrated dataset.
7. Calibrate refusal thresholds for all retrieval modes on `dev`: `python eval/run_eval.py --calibrate`.
8. Start the API: `uvicorn src.api.routes:app --reload`.

The embedding and reranker models download on first use. Keep source PDFs under version control or provide them alongside this repository before ingesting. Generated Chroma data, logs, caches, reports, and manifests are ignored by Git.

## Commands

```bash
python main.py ingest
python main.py ask "What documents say about inventory control?"
python eval/run_eval.py --mode dense
python eval/run_eval.py --mode hybrid
python eval/run_eval.py --mode hybrid_rerank
uvicorn src.api.routes:app --reload
```

## Evaluation data

Each line in `data/testing.jsonl` is one JSON object; the JSON Schema is in `eval/testing.schema.json`. `reference_contexts` holds the actual supporting text and `reference_sources` separately anchors it to file/PDF page for retrieval metrics. See `AGENTS.md` for split rules and metric definitions. The uploaded PDFs do not include a verified golden Q&A set; author and review the ~50 questions from extracted source text before treating results as a baseline.
