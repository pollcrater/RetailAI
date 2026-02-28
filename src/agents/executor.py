from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import logging
import time
from typing import Any

from src.agents.base import AgentResult
from src.tools.duckdb_tools import DuckDBRunner


@dataclass(frozen=True)
class SqlExecutorAgent:
    """Agent 2: execute SQL against DuckDB and return a dataframe."""

    name: str = "executor"
    max_workers: int = 4
    query_timeout_seconds: float = 20.0

    @staticmethod
    def _augment_error(error: str) -> str:
        if "Could not convert string" in error or "Conversion Error" in error:
            return error + " | Hint: use TRY_CAST(...) and filter out null casts for numeric fields."
        return error

    def run(self, state: dict[str, Any]) -> AgentResult:
        logger = logging.getLogger(__name__)
        runner: DuckDBRunner = state["duckdb_runner"]
        queries = state.get("queries") or []
        sql = str(state.get("sql", "")).strip()
        logger.debug("EXECUTOR DEBUG - SQL received: %r", sql)
        logger.debug("SQL length: %s", len(sql))
        logger.debug("First 50 chars: %s", sql[:50] if sql else "EMPTY")
        if not queries and not sql:
            return AgentResult(ok=False, content=None, error="Missing SQL to execute")

        if queries:
            normalized_queries = [str(q).strip() for q in queries if str(q).strip()]
            if not normalized_queries:
                return AgentResult(ok=False, content=None, error="Missing SQL to execute")

            max_workers = max(1, min(self.max_workers, len(normalized_queries)))
            start_times: dict[int, float] = {}
            futures: list[tuple[int, str, Any]] = []
            results_by_index: dict[int, Any] = {}
            execution_meta: list[dict[str, Any]] = []
            query_errors: list[dict[str, Any]] = []

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for index, query in enumerate(normalized_queries):
                    start_times[index] = time.perf_counter()
                    futures.append((index, query, pool.submit(runner.query, query)))

                for index, query, future in futures:
                    try:
                        res = future.result(timeout=self.query_timeout_seconds)
                    except TimeoutError:
                        future.cancel()
                        duration_ms = int((time.perf_counter() - start_times[index]) * 1000)
                        error = f"Query {index + 1} timed out after {self.query_timeout_seconds:.1f}s"
                        query_errors.append({"index": index, "error": error, "timed_out": True})
                        execution_meta.append(
                            {
                                "index": index,
                                "status": "timeout",
                                "row_count": 0,
                                "duration_ms": duration_ms,
                            }
                        )
                        continue

                    duration_ms = int((time.perf_counter() - start_times[index]) * 1000)
                    if not res.ok:
                        error = self._augment_error(res.error or "Query failed")
                        query_errors.append({"index": index, "error": error, "timed_out": False})
                        execution_meta.append(
                            {
                                "index": index,
                                "status": "error",
                                "row_count": 0,
                                "duration_ms": duration_ms,
                            }
                        )
                        continue

                    df = res.content
                    results_by_index[index] = df
                    row_count = int(len(df)) if hasattr(df, "__len__") else 0
                    execution_meta.append(
                        {
                            "index": index,
                            "status": "ok",
                            "row_count": row_count,
                            "duration_ms": duration_ms,
                        }
                    )

            if not results_by_index:
                first_error = query_errors[0]["error"] if query_errors else "All queries failed"
                return AgentResult(
                    ok=False,
                    content={"execution_meta": execution_meta, "query_errors": query_errors},
                    error=str(first_error),
                )

            ordered_indexes = sorted(results_by_index.keys())
            ordered_results = [results_by_index[i] for i in ordered_indexes]
            return AgentResult(
                ok=True,
                content={
                    "dfs": ordered_results,
                    "execution_meta": execution_meta,
                    "query_errors": query_errors,
                },
            )

        res = runner.query(sql)
        if not res.ok:
            error = self._augment_error(res.error or "Query failed")
            return AgentResult(ok=False, content=None, error=error)

        row_count = int(len(res.content)) if hasattr(res.content, "__len__") else 0
        return AgentResult(
            ok=True,
            content={
                "df": res.content,
                "execution_meta": [{"index": 0, "status": "ok", "row_count": row_count, "duration_ms": 0}],
                "query_errors": [],
            },
        )
