from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from src.agents.base import AgentResult
from src.tools.duckdb_tools import DuckDBRunner


@dataclass(frozen=True)
class SqlExecutorAgent:
    """Agent 2: execute SQL against DuckDB and return a dataframe."""

    name: str = "executor"

    def run(self, state: dict[str, Any]) -> AgentResult:
        logger = logging.getLogger(__name__)
        runner: DuckDBRunner = state["duckdb_runner"]
        queries = state.get("queries") or []
        sql = str(state.get("sql", "")).strip()
        logger.error("EXECUTOR DEBUG - SQL received: %r", sql)
        logger.error("SQL length: %s", len(sql))
        logger.error("First 50 chars: %s", sql[:50] if sql else "EMPTY")
        if not queries and not sql:
            return AgentResult(ok=False, content=None, error="Missing SQL to execute")

        if queries:
            results: list[Any] = []
            for q in queries:
                res = runner.query(q)
                if not res.ok:
                    error = res.error or "Query failed"
                    if "Could not convert string" in error or "Conversion Error" in error:
                        error += " | Hint: use TRY_CAST(...) and filter out null casts for numeric fields."
                    return AgentResult(ok=False, content=None, error=error)
                results.append(res.content)
            return AgentResult(ok=True, content={"dfs": results})

        res = runner.query(sql)
        if not res.ok:
            error = res.error or "Query failed"
            if "Could not convert string" in error or "Conversion Error" in error:
                error += " | Hint: use TRY_CAST(...) and filter out null casts for numeric fields."
            return AgentResult(ok=False, content=None, error=error)

        return AgentResult(ok=True, content={"df": res.content})
