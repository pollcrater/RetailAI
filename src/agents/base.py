from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentResult:
    ok: bool
    content: Any
    error: str | None = None


class Agent(Protocol):
    """Agent interface (open/closed: add new agents without changing the graph)."""

    name: str

    def run(self, state: dict[str, Any]) -> AgentResult: ...
