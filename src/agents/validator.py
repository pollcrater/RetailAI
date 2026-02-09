from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import json

from src.agents.base import AgentResult
from src.tools.duckdb_tools import _is_safe_select_sql, dataframe_to_json_preview
from src.utils.llm import generate_text
from src.prompts import prompt_validator


@dataclass(frozen=True)
class ResultValidatorAgent:
    """Agent 3: validate results + craft final answer with citations to rows returned."""

    name: str = "validator"

    def run(self, state: dict[str, Any]) -> AgentResult:
        question = str(state.get("question", "")).strip()
        sql = str(state.get("sql", "")).strip()
        tool_error = state.get("tool_error")
        df = state.get("query_result_df")
        dfs = state.get("query_result_dfs")
        preview_single = state.get("query_result_preview")
        preview_list = state.get("query_result_previews")

        if tool_error:
            return AgentResult(
                ok=False,
                content={"planner_feedback": f"SQL execution failed: {tool_error}"},
                error="Execution failed",
            )

        if not sql and not dfs:
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

        prompt = prompt_validator(question=question, sql=sql or "<multiple queries>", results_json=preview)

        answer = generate_text(prompt)
        return AgentResult(ok=True, content={"answer": answer})
