from __future__ import annotations

from typing import Any


def prompt_db_summary(*, metrics: dict[str, Any]) -> str:
    return (
        "You are a retail analytics assistant. "
        "Write a concise DB-based summary from the provided metrics JSON. "
        "Highlight KPIs, trends, and deltas when available. "
        "If KPIs are missing, summarize schema, sample rows, and numeric profiles instead. "
        "Include a data coverage line with date range, months included, and filters applied. "
        "If no filters were applied, explicitly say so. "
        "If a metric is missing, explicitly say so.\n\n"
        f"Metrics JSON: {metrics}\n\n"
        "Output rules:\n"
        "- 4-8 bullet points\n"
        "- Include key numbers\n"
        "- Mention last-month delta if present\n"
    )
