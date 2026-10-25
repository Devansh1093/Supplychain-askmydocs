# AGENTS.md — SupplyDocsQA

## Goal
"Ask my docs" RAG system over supply chain documents (PDF + Markdown).
Answers cite the source file and decline when retrieved chunks don't support a response.


## Tech stack
- Python 3.11, plain LangChain 
- Vector store: ChromaDB (local, persistent)
- Embeddings: `BAAI/bge-small-en-v1.5` (local)
- Keyword search: `rank_bm25`, fused with dense results via RRF
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local)
- LLM: OpenAI API via `langchain-openai`
- Eval: Ragas
- API: FastAPI
- Frontend: React + Vite

## Project structure
```
SupplyDocsQA/
├── README.md
├── requirements.txt
├── .env                    # OPENAI_API_KEY (never commit)
├── .gitignore
├── config.yaml             # chunk size, overlap, top_k, thresholds, model names
├── main.py                 # entry point: CLI / starts app
├── data/
│   ├── raw/                # source PDFs + markdown
│   └── testing.jsonl        # eval Q&A (incl. unanswerable)
├── src/
│   ├── ingestion/          # loader.py        load PDFs/markdown with source and page metadata
│   ├── chunking/           # chunker.py       500-800 tokens, 100 overlap
│   ├── embeddings/         # embedder.py      text -> vectors
│   ├── vectordb/           # vector_store.py  Chroma operations
│   ├── retrieval/          # retriever.py (dense), bm25.py, hybrid.py (RRF), reranker.py
│   ├── prompts/            # prompt_templates.py
│   ├── llm/                # llm_client.py (OpenAI), guard.py (citation check / refusal)
│   ├── api/                # routes.py        FastAPI endpoints
│   ├── utils/              # helpers.py
│   └── pipeline.py         # ask(query) -> {answer, citations, refused}
├── eval/                   # run_eval.py (Ragas) + results/
├── frontend/               # React app (Phase 4)
```

## Build phases (finish each before the next)
1. **Fundamentals:** ingest, chunk, embed into Chroma, dense top-K retrieval, answer with citations.
2. **Quality + safety:** BM25 + dense with RRF, cross-encoder rerank (fetch ~20, keep top 5), refuse when chunks don't support an answer.
3. **Evaluation:** ~50 golden Q&A (~10 unanswerable), Ragas faithfulness / answer relevancy / context precision / context recall. Report dense vs hybrid vs hybrid+rerank.
4. **Frontend:** React + Vite chat UI with citation chips, source panel, refusal state, and retrieval-mode toggle. Streamlit is not part of the project.

## API (FastAPI)
- `POST /ask` `{query, mode}` -> `{answer, citations: [{file}], refused}`. User-facing citations contain only the source filename, with no chapter, page numbers, or chunk IDs. Internal retrieved-chunk metadata is separate and may include `{chunk_id, file, pdf_page, printed_page, text, score}`; do not expose it as the citation object.
- `POST /ask/stream` (SSE), `GET /chunk/{chunk_id}`, `GET /health`
- Enable CORS for `http://localhost:5173`.

## Evaluation
- Store one JSON object per line in `data/testing.jsonl` with fields: `id`, `question`, `reference_answer`, `reference_contexts` (list of supporting text strings), `answerable` (boolean), `type` (`factual`, `formula`, `multi_doc`, `conflict`, or `unanswerable`), `source_docs`, `split` (`dev` or `test`), and `notes`. For deterministic retrieval hit@k/MRR labels, also provide `reference_sources` as a list of `{file, pdf_page}` anchors; page anchors are evaluation metadata only and never user-facing citations.
- Start with about 50 questions. Assign roughly 30% to `dev` for tuning thresholds and prompts, and 70% to `test` for final reporting only. Never tune against `test`.
- Make roughly 20–30% of questions unanswerable. Include at least 3 formula questions and 3 conflict questions. `reference_contexts` may be empty for unanswerable questions.
- Use the same set for retrieval metrics (hit@k and MRR against the source chunks containing `reference_contexts`) and Ragas metrics (faithfulness, answer relevancy, context precision, and context recall). Keep `pdf_page` and `printed_page` as internal chunk metadata; do not put page pointers in `reference_contexts`.
- For reproducibility, use temperature 0 and pin model names, library versions, and the config hash in each results file. Use one fixed judge model; changing it starts a new baseline.
- Cache generation and judge LLM calls on disk by a key including model, parameters, prompt, and input content. Keep Ragas concurrency low, retry with exponential backoff and jitter, and resume interrupted runs from cache.
- Save per-question scores, not only averages, to `eval/results/<date>_<git_sha>.json`.

## Ingestion & index maintenance
- Track each ingested document in `data/manifest.json` with `doc_id`, `sha256`, `parser_version`,
  `chunk_config_hash`, `embed_model`, and `n_chunks`.
- When a source file is added or its hash changes, remove that document's old chunks from Chroma
  and BM25, then re-ingest that document. Rebuild the BM25 index from stored chunks on every
  ingest; this is expected to be inexpensive.
- When the embedding model, chunk settings, or parser version changes, build a fresh vector
  collection and swap to it only after the full rebuild succeeds. Do not mix vectors produced
  with different settings in one active collection.
- Write `data/reports/extraction_report.json` after ingestion. For each document, record page
  count, empty-page count (fewer than 30 words), average words per page, junk-glyph ratio, and
  count of stripped header/footer lines.
- Flag a document if more than 15% of its pages are empty or its junk-glyph ratio indicates
  corrupted extraction. Review five representative flagged pages alongside extracted text before
  indexing; ingestion should fail on a flagged document unless explicitly run with
  `--accept-report`.
- Keep parser version and all ingestion/chunking settings in the manifest so stale indexes can be
  detected and rebuilt deterministically.

## Rules
1. User-facing citations use `[file]` only. Do not extract or store chapter metadata. Do not show page numbers in citations. Store `pdf_page` and `printed_page` in chunk metadata for debugging, extraction review, and evaluation `reference_contexts`.
2. Prompt: answer only from the provided context; otherwise return the refusal message: "I can't find support for that in the provided documents."
3. Retrieval gate: if the top reranker score is below the calibrated threshold for the selected retrieval mode in `config.yaml`, refuse without calling the LLM. Calibrate separate thresholds for dense, hybrid, and hybrid+rerank on the dev split; do not hard-code them. Store calibration details (date, dev-set version/hash, reranker model, mode, index hash, and chosen threshold) with evaluation results. App startup and evaluation must fail with a clear error if any required threshold is missing, null, or lacks calibration metadata; never use an implicit default.
4. Generation returns structured JSON: `{claims: [{text, chunk_ids}], unsupported: [...]}`. Each claim must reference one or more retrieved chunk IDs. `unsupported` contains brief descriptions of requested information the retrieved context could not support; it is for the guard only and must never be presented as factual answer content.
5. In `guard.py`, discard claims that reference chunk IDs outside the retrieved set or have no supporting retrieved context. If no supported claims remain, return the fixed refusal message. Do not add uncited claims. For partial support, use `unsupported` only to write the required one-line notice that some requested information was not found; do not reveal unsupported claims or guesses.
6. For partial support, answer the supported part and add one sentence explaining what information was not found. Do not guess to fill gaps.
7. When sources conflict on a formula, number, or definition, present each supported version with its citations and state that the sources differ. Never merge or average conflicting values.
8. Use the fixed refusal message: "I can't find support for that in the provided documents." Log refusals and the retrieved chunk IDs/scores so the searched evidence can be reviewed; do not vary the refusal wording with query-specific details.
9. All tunables live in `config.yaml`. No magic numbers in code.
10. Secrets only in `.env`. Never commit keys.
11. Cache OpenAI responses and Ragas judge calls so repeated evaluations are reproducible and do not repeat API calls.
12. Each retrieval stage (dense, bm25, hybrid, rerank) must be callable and testable on its own.
13. Log retrieved chunk ids and scores per query to `logs/app.log`.
14. Keep modules small and typed. Don't add dependencies outside the stack without asking.
15. Don't build text-to-SQL, the React frontend, or AWS deployment before Phase 3 is done.

## Commands
```bash
pip install -r requirements.txt
python main.py ingest               # load, chunk, embed, build indexes
python main.py ask "your question"
uvicorn src.api.routes:app --reload
python eval/run_eval.py --mode dense|hybrid|hybrid_rerank
pytest -q
cd frontend && npm install && npm run dev
```

## Done when
- [ ] Phase 1: cited answers working end-to-end
- [ ] Phase 2: hybrid + rerank + refusal working
- [ ] Phase 3: Ragas results table committed to `eval/results/`
- [ ] Phase 4: React UI working
- [ ] README Intro, architecture(include tech stack in this only table format),setup

## Provenance & coverage

### `data/sources.yaml`
Maintain one entry for every source PDF. Fill fields only from the PDF itself or from the user;
never guess. Use `TBD` for unknown values. `pdf_created` is the PDF file creation date in
`YYYY-MM` format, not the publication year. `verified` is a status: `cover` means title and
publisher were confirmed from the document, `partial` means some fields remain TBD, and `no`
means nothing has been confirmed. Check each SHA-256 at ingest; on mismatch, warn and re-ingest
that document.

Use the exact current filenames:
- `Quality Management Systems for non-laboratory settings.pdf` (`who_buffer_stock`)
- `inventory-control.pdf` (`inventory_control`)
- `Logistics Handbook.pdf` (`usaid_logistics_handbook`)
- `scmhandbook.pdf` (`crs_scm_handbook`)
- `Whole-manual.pdf` (`brc_logistics_manual`)
- `The handbook of logistics and distribution managment.pdf` (`rushton_handbook`)
- `World-Bank-Supply-Chain-Management-Guidance.pdf` (`wb_scm_guidance`)

### Coverage and overlap (`scripts/overlap_report.py`)
Run after a full ingest or when source files change, not after every single-document re-ingest.
Write `data/reports/overlap_report.json` and a readable Markdown report.
1. Embed chunks and find each chunk's nearest neighbor in a different document; flag cosine
   similarity at or above `overlap_sim` from config.
2. Check near-verbatim overlap across documents with normalized text, 8-word shingles, and
   MinHash/Jaccard.
3. Produce a topic coverage matrix using keyword tags such as buffer stock, max-min, reorder
   level, lead time, AMC, FEFO/FIFO, receiving, warehouse layout, transport, procurement, and
   Incoterms. Treat this as a rough diagnostic, not proof of adequate topic coverage. Topics
   appearing in 0–1 documents are potential gaps; avoid evaluation questions that assume depth.
4. Manually review the top 20 flagged pairs and persist labels in `data/reports/overlap_labels.json`; label each `duplicate`, `paraphrase`, or
   `conflict`. Keep paraphrases and conflicts as separate evidence. Add evaluation questions for
   conflicts and present both supported versions with their source-file citations.
5. At retrieval, collapse only pairs confirmed as `duplicate`: keep the higher-scored chunk and
   attach both source-file citations. Never collapse paraphrases or conflicts.
6. First inspect WHO toolkit vs USAID handbook (alert/buffer stock formulas), USAID handbook vs
   CRS handbook (inventory control and max-min), and CRS handbook vs BRC manual (warehousing and
   procurement).
7. `overlap_sim: 0.92` is a starting default; calibrate it by reviewing flagged pairs.
