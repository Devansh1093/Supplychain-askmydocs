from __future__ import annotations

import json
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.config import load_config, require_calibrated_threshold
from src.pipeline import SupplyDocsPipeline


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = "hybrid_rerank"


@lru_cache(maxsize=1)
def get_pipeline() -> SupplyDocsPipeline:
    return SupplyDocsPipeline(load_config())


@asynccontextmanager
async def lifespan(_: FastAPI):
    config = load_config()
    require_calibrated_threshold(config)
    get_pipeline()
    yield


app = FastAPI(title="SupplyDocsQA", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    try:
        return get_pipeline().ask(request.query, request.mode)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503 if isinstance(exc, RuntimeError) else 422, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    def events():
        try:
            response = get_pipeline().ask(request.query, request.mode)
            yield f"event: answer\ndata: {json.dumps(response, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/chunk/{chunk_id}")
def get_chunk(chunk_id: str) -> dict[str, Any]:
    try:
        chunk = get_pipeline().retrieval.get_chunk(chunk_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if chunk is None:
        raise HTTPException(status_code=404, detail="Chunk not found")
    return chunk


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
