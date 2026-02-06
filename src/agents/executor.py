from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.base import AgentResult
from src.tools.duckdb_tools import DuckDBRunner


@dataclass(frozen=True)
class SqlExecutorAgent:
    """Agent 2: execute SQL against DuckDB and return a dataframe."""

    name: str = "executor"

    def run(self, state: dict[str, Any]) -> AgentResult:
        runner: DuckDBRunner = state["duckdb_runner"]
        queries = state.get("queries") or []
        sql = str(state.get("sql", "")).strip()
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
