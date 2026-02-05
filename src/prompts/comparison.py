from __future__ import annotations

from typing import Any


def prompt_comparison_summary(*, metrics: dict[str, Any]) -> str:
    return (
        "You are a retail operations analyst. "
        "Summarize a vendor price comparison table based on the provided metrics JSON. "
        "Call out which provider is cheaper more often, average prices, and any large gaps. "
        "If some prices are missing or non-numeric, say so.\n\n"
        f"Metrics JSON: {metrics}\n\n"
        "Output rules:\n"
        "- 4-8 bullet points\n"
        "- Highlight critical numbers\n"
        "- Keep it concise and business-friendly\n"
    )
