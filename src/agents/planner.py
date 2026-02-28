from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from pydantic import ValidationError

from src.agents.base import AgentResult
from src.contracts import PlannerOutputModel
from src.tools.duckdb_tools import DuckDBRunner
from src.utils.llm import generate_text
from src.prompts import prompt_sql_planner


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction from LLM output."""

    text = text.strip()
    # Prefer fenced ```json
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None

    # Try first {...} block
    m2 = re.search(r"(\{.*\})", text, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            return None

    return None


@dataclass(frozen=True)
class SqlPlannerAgent:
    """Agent 1: turn a question into a safe DuckDB SELECT query."""

    name: str = "planner"

    def run(self, state: dict[str, Any]) -> AgentResult:
        question = str(state.get("question", "")).strip()
        if not question:
            return AgentResult(ok=False, content=None, error="Missing question")

        runner: DuckDBRunner = state["duckdb_runner"]
        schema_md_res = runner.get_schema_markdown(schemas=("raw", "mart"))
        schema_md = schema_md_res.content if schema_md_res.ok else "<schema unavailable>"

        feedback = str(state.get("planner_feedback", "")).strip()
        prompt = prompt_sql_planner(
            schema_md=schema_md,
            question=question,
            feedback=feedback,
            memory=state.get("memory", []) or [],
            resolved_filters=state.get("resolved_filters", {}) or {},
        )

        raw = generate_text(prompt)
        obj = _extract_json(raw)
        if not obj:
            return AgentResult(
                ok=False,
                content={"raw": raw},
                error="Planner did not return valid JSON.",
            )

        try:
            normalized = PlannerOutputModel.model_validate(obj).model_dump()
        except ValidationError as e:
            return AgentResult(
                ok=False,
                content={"raw": raw, "validation_errors": e.errors()},
                error="Planner JSON failed contract validation.",
            )

        return AgentResult(
            ok=True,
            content={
                "sql": normalized["sql"],
                "queries": normalized["queries"],
                "meta": normalized,
            },
        )
