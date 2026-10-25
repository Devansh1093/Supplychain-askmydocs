from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import resolve_path
from src.utils.hashing import stable_hash

logger = logging.getLogger(__name__)

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["claims", "unsupported"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "chunk_ids"],
                "properties": {
                    "text": {"type": "string"},
                    "chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "unsupported": {"type": "array", "items": {"type": "string"}},
    },
}


class OpenAIClient:
    def __init__(self, config: dict[str, Any]) -> None:
        load_dotenv(Path(config["_root"]) / ".env")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before starting the application.")
        self.config = config
        llm_config = config["llm"]
        if llm_config.get("provider") != "openai":
            raise ValueError("llm.provider must be 'openai'")
        self.model_name = str(llm_config["model"])
        self.temperature = float(llm_config["temperature"])
        self.max_retries = int(llm_config["max_retries"])
        self.backoff_base = float(llm_config["backoff_base_s"])
        self.backoff_max = float(llm_config["backoff_max_s"])
        self.jitter_low = float(llm_config["backoff_jitter_low"])
        self.jitter_high = float(llm_config["backoff_jitter_high"])
        self.cache_dir = resolve_path(config, "cache_dir")
        self.enabled = bool(config["cache"]["enabled"])
        self.model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_retries=0,
        )
        self.structured_model = self.model.with_structured_output(
            RESPONSE_SCHEMA, method="json_schema", strict=True
        )

    def _cache_key(self, messages: list[tuple[str, str]]) -> str:
        return stable_hash({"provider": "openai", "model": self.model_name, "temperature": self.temperature, "messages": messages})

    def generate_json(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"{key}.json"
        if self.enabled and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        lc_messages = [SystemMessage(content=text) if role == "system" else HumanMessage(content=text) for role, text in messages]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self.structured_model.invoke(lc_messages)
                if not isinstance(data, dict):
                    raise ValueError("OpenAI structured output was not a JSON object")
                if not isinstance(data.get("claims"), list) or not isinstance(data.get("unsupported"), list):
                    raise ValueError("OpenAI response is missing claims or unsupported arrays")
                if self.enabled:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = cache_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                    temporary.replace(cache_path)
                return data
            except Exception as exc:  # API errors, rate limits, and malformed output are retried
                last_error = exc
                if attempt + 1 >= self.max_retries:
                    break
                delay = min(self.backoff_max, self.backoff_base * (2 ** attempt))
                time.sleep(delay * random.uniform(self.jitter_low, self.jitter_high))
                logger.warning("OpenAI attempt %s failed; retrying: %s", attempt + 1, exc)
        raise RuntimeError(f"OpenAI request failed after {self.max_retries} attempts: {last_error}")
