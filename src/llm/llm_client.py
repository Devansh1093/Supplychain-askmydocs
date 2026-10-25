from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.config import resolve_path
from src.utils.hashing import stable_hash

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        llm_config = config["llm"]
        if llm_config.get("provider") != "ollama":
            raise ValueError("llm.provider must be 'ollama'")
        self.model_name = str(llm_config["model"])
        self.base_url = str(llm_config["base_url"])
        self.temperature = float(llm_config["temperature"])
        self.max_retries = int(llm_config["max_retries"])
        self.backoff_base = float(llm_config["backoff_base_s"])
        self.backoff_max = float(llm_config["backoff_max_s"])
        self.jitter_low = float(llm_config["backoff_jitter_low"])
        self.jitter_high = float(llm_config["backoff_jitter_high"])
        self.cache_dir = resolve_path(config, "cache_dir")
        self.enabled = bool(config["cache"]["enabled"])
        self.model = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            format="json",
        )

    def _cache_key(self, messages: list[tuple[str, str]]) -> str:
        return stable_hash({"provider": "ollama", "base_url": self.base_url, "model": self.model_name, "temperature": self.temperature, "messages": messages})

    def generate_json(self, messages: list[tuple[str, str]]) -> dict[str, Any]:
        key = self._cache_key(messages)
        cache_path = self.cache_dir / f"{key}.json"
        if self.enabled and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        lc_messages = [SystemMessage(content=text) if role == "system" else HumanMessage(content=text) for role, text in messages]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.model.invoke(lc_messages)
                content = response.content
                if isinstance(content, list):
                    content = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)
                data = _parse_json(str(content))
                if not isinstance(data.get("claims"), list) or not isinstance(data.get("unsupported"), list):
                    raise ValueError("Ollama response is missing claims or unsupported arrays")
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
                logger.warning("Ollama attempt %s failed; retrying: %s", attempt + 1, exc)
        raise RuntimeError(f"Ollama request failed after {self.max_retries} attempts: {last_error}")


def _parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM output did not contain a JSON object")
    return json.loads(stripped[start:end + 1])
