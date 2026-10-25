from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.load import dumps, loads

from src.utils.hashing import stable_hash


class DiskCache(BaseCache):
    """Persistent cache for LangChain calls without SQLite or SQLAlchemy."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, prompt: str, llm_string: str) -> Path:
        key = stable_hash({"prompt": prompt, "llm": llm_string})
        return self.directory / f"{key}.json"

    def lookup(self, prompt: str, llm_string: str) -> Any | None:
        path = self._path(prompt, llm_string)
        if not path.exists():
            return None
        try:
            return loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(prompt, llm_string)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(dumps(return_val), encoding="utf-8")
        temporary.replace(path)

    def clear(self, **_: Any) -> None:
        return None
