from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: Any
    error: str | None = None


class Tool(Protocol):
    """Simple tool interface (open/closed: add new tools without changing callers)."""

    name: str

    def run(self, **kwargs: Any) -> ToolResult: ...
