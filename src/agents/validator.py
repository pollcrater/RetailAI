from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json
import logging

from src.agents.base import AgentResult
from src.contracts import QueryErrorModel, QueryExecutionMetaModel, build_provenance_json, normalize_preview_json
from src.tools.duckdb_tools import _is_safe_select_sql, dataframe_to_json_preview
from src.utils.llm import generate_text
from src.prompts import prompt_validator


def _build_rows_fallback_answer(preview_json: str, *, max_rows: int = 10) -> str:
    try:
        payload = json.loads(preview_json)
    except Exception:
        return "AI narrative generation is unavailable. SQL ran successfully, but result preview could not be parsed."

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = [r for r in payload["rows"] if isinstance(r, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        for result in payload["results"]:
            if isinstance(result, dict) and isinstance(result.get("rows"), list):
                rows.extend([r for r in result["rows"] if isinstance(r, dict)])

    if not rows:
        return "Query ran successfully, but returned no rows."

    sample = rows[:max_rows]
    first = sample[0]
    lower_to_key = {str(k).lower(): str(k) for k in first.keys()}
    category_key = lower_to_key.get("category")
    value_key = (
        lower_to_key.get("total_sales")
        or lower_to_key.get("sales")
        or lower_to_key.get("amount")
        or lower_to_key.get("total")
    )

    lines = ["AI validator is unavailable (gateway/policy). Showing SQL result preview:"]
    if category_key and value_key:
        for row in sample:
            lines.append(f"- {row.get(category_key)}: {row.get(value_key)}")
    else:
        for row in sample:
            lines.append(f"- {json.dumps(row, default=str)}")

    remaining = len(rows) - len(sample)
    if remaining > 0:
        lines.append(f"- ... and {remaining} more row(s)")

    return "\n".join(lines)


def _compact_preview_json(preview_json: str, *, max_rows: int = 40) -> str:
    """Trim preview payload sent to validator LLM to reduce gateway false positives."""

    return normalize_preview_json(preview_json, max_rows=max_rows)


def _query_summary(sql: str, preview_json: str) -> str:
    """Provide minimal query context without sending raw SQL text."""

    if not sql:
        return "multi-query KPI and trend summary"

    lowered = sql.lower()
    has_group = "group by" in lowered
    has_order = "order by" in lowered
    has_limit = "limit" in lowered

    try:
        payload = json.loads(preview_json)
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list) and payload["rows"]:
            first = payload["rows"][0]
            if isinstance(first, dict):
                cols = list(first.keys())[:5]
                cols_text = ", ".join([str(c) for c in cols])
                return (
                    f"analytical select with columns [{cols_text}]"
                    f"; grouped={has_group}, ordered={has_order}, limited={has_limit}"
                )
    except Exception:
        pass

    return f"read-only analytical query; grouped={has_group}, ordered={has_order}, limited={has_limit}"


def _provenance_payload(execution_meta: Any, query_errors: Any) -> str:
    return build_provenance_json(execution_meta, query_errors)


def _provenance_lines(execution_meta: Any, query_errors: Any) -> list[str]:
    meta: list[QueryExecutionMetaModel] = []
    errs: list[QueryErrorModel] = []

    if isinstance(execution_meta, list):
        for item in execution_meta:
            try:
                meta.append(QueryExecutionMetaModel.model_validate(item))
            except Exception:
                continue

    if isinstance(query_errors, list):
        for item in query_errors:
            try:
                errs.append(QueryErrorModel.model_validate(item))
            except Exception:
                continue

    if not meta and not errs:
        return []

    ok_count = sum(1 for item in meta if item.status == "ok")
    total = len(meta)
    failed = len(errs)
    duration_ms = sum(int(item.duration_ms) for item in meta)
    lines = [
        f"queries executed: {total}, succeeded: {ok_count}, failed: {failed}",
        f"aggregate execution time: {duration_ms} ms",
    ]
    if failed:
        failed_indices = [int(err.index) + 1 for err in errs]
        if failed_indices:
            lines.append(f"failed query indexes: {failed_indices}")
    return lines


@dataclass(frozen=True)
class ResultValidatorAgent:
    """Agent 3: validate results + craft final answer with citations to rows returned."""

    name: str = "validator"

    def run(self, state: dict[str, Any]) -> AgentResult:
        question = str(state.get("question", "")).strip()
        sql = str(state.get("sql", "")).strip()
        queries = state.get("queries") or []
        tool_error = state.get("tool_error")
        df = state.get("query_result_df")
        dfs = state.get("query_result_dfs")
        preview_single = state.get("query_result_preview")
        preview_list = state.get("query_result_previews")
        execution_meta = state.get("query_execution_meta")
        query_errors = state.get("query_errors")

        if tool_error:
            return AgentResult(
                ok=False,
                content={"planner_feedback": f"SQL execution failed: {tool_error}"},
                error="Execution failed",
            )

        has_multi_queries = isinstance(queries, list) and len(queries) > 0
        has_any_results = bool(preview_single) or bool(preview_list) or bool(dfs) or df is not None

        if not sql and not has_multi_queries and not has_any_results:
            return AgentResult(
                ok=False,
                content={"planner_feedback": "Missing SQL for validation."},
                error="Missing SQL",
            )

        if sql and not _is_safe_select_sql(sql):
            return AgentResult(
                ok=False,
                content={"planner_feedback": "Generate a single safe SELECT query (or WITH..SELECT) only."},
                error="Unsafe or missing SQL",
            )

        if preview_list:
            preview = json.dumps(
                {"results": [json.loads(p) for p in preview_list]},
                default=str,
            )
        elif dfs:
            preview = json.dumps(
                {"results": [json.loads(dataframe_to_json_preview(d)) for d in dfs]},
                default=str,
            )
        elif preview_single:
            preview = preview_single
        else:
            preview = dataframe_to_json_preview(df)

        compact_preview = _compact_preview_json(preview, max_rows=40)
        summary = _query_summary(sql, compact_preview)
        prompt = prompt_validator(
            question=question,
            sql_summary=summary,
            results_json=compact_preview,
            provenance_json=_provenance_payload(execution_meta, query_errors),
        )

        try:
            answer = generate_text(prompt)
        except Exception as e:
            logging.getLogger(__name__).warning("Validator AI unavailable, using SQL preview fallback: %s", e)
            answer = _build_rows_fallback_answer(preview)

        prov_lines = _provenance_lines(execution_meta, query_errors)
        if prov_lines:
            answer = answer + "\n\nQuery execution details:\n" + "\n".join([f"- {line}" for line in prov_lines])
        return AgentResult(ok=True, content={"answer": answer})
