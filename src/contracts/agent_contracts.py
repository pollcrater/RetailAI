from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class PlannerOutputModel(BaseModel):
    sql: str = ""
    queries: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    needed_tables: list[str] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)
    rationale_summary: str = ""
    confidence: float | None = None
    reason_tags: list[str] = Field(default_factory=list)

    @field_validator("sql", mode="before")
    @classmethod
    def _coerce_sql(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("queries", "assumptions", "needed_tables", mode="before")
    @classmethod
    def _coerce_string_list(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        return [text] if text else []

    @field_validator("filters", mode="before")
    @classmethod
    def _coerce_filters(cls, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @field_validator("rationale_summary", mode="before")
    @classmethod
    def _coerce_summary(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("rationale_summary")
    @classmethod
    def _trim_summary(cls, value: str) -> str:
        return value[:280]

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @field_validator("confidence")
    @classmethod
    def _bound_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value < 0:
            return 0.0
        if value > 1:
            return 1.0
        return value

    @field_validator("reason_tags", mode="before")
    @classmethod
    def _coerce_reason_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]

        tags: list[str] = []
        for item in value:
            tag = str(item).strip().lower().replace(" ", "_")
            tag = re.sub(r"[^a-z0-9_\-]", "", tag)[:32]
            if tag and tag not in tags:
                tags.append(tag)
            if len(tags) >= 6:
                break
        return tags

    @model_validator(mode="after")
    def _ensure_executable_query(self) -> "PlannerOutputModel":
        if not self.sql and not self.queries:
            raise ValueError("Planner output must include 'sql' or non-empty 'queries'.")
        return self


class ResultRowsModel(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)


class ResultSetModel(BaseModel):
    results: list[ResultRowsModel] = Field(default_factory=list)


class QueryExecutionMetaModel(BaseModel):
    index: int
    status: Literal["ok", "error", "timeout"]
    row_count: int = 0
    duration_ms: int = 0

    @field_validator("row_count", "duration_ms")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        return max(0, int(value))


class QueryErrorModel(BaseModel):
    index: int
    error: str
    timed_out: bool = False


def normalize_preview_json(preview_json: str, *, max_rows: int = 40) -> str:
    try:
        single = ResultRowsModel.model_validate_json(preview_json)
        return ResultRowsModel(rows=single.rows[:max_rows]).model_dump_json()
    except ValidationError:
        pass

    try:
        multi = ResultSetModel.model_validate_json(preview_json)
        compact = ResultSetModel(
            results=[ResultRowsModel(rows=result.rows[:max_rows]) for result in multi.results]
        )
        return compact.model_dump_json()
    except ValidationError:
        return preview_json


def build_provenance_json(execution_meta: Any, query_errors: Any) -> str:
    valid_meta: list[QueryExecutionMetaModel] = []
    valid_errors: list[QueryErrorModel] = []

    if isinstance(execution_meta, list):
        for item in execution_meta:
            try:
                valid_meta.append(QueryExecutionMetaModel.model_validate(item))
            except ValidationError:
                continue

    if isinstance(query_errors, list):
        for item in query_errors:
            try:
                valid_errors.append(QueryErrorModel.model_validate(item))
            except ValidationError:
                continue

    payload = {
        "queries_total": len(valid_meta),
        "queries_ok": sum(1 for item in valid_meta if item.status == "ok"),
        "queries_failed": len(valid_errors),
        "execution": [m.model_dump() for m in valid_meta],
        "errors": [e.model_dump() for e in valid_errors],
    }
    return json.dumps(payload, default=str)
