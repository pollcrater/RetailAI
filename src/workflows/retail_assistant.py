from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.llm import generate_text


@dataclass(frozen=True)
class DataFacts:
    row_count: int
    columns: list[str]
    numeric_summary: dict[str, dict[str, float]]
    group_summaries: dict[str, list[dict[str, Any]]]


def _read_csv_head(csv_path: Path, max_rows: int = 200_000) -> pd.DataFrame:
    # For now we keep it simple and safe for laptops:
    # - load up to max_rows
    # - inference via pandas
    return pd.read_csv(csv_path).head(max_rows)


def _find_dimension_columns(columns: list[str]) -> list[str]:
    wanted = [
        "region",
        "state",
        "store",
        "category",
        "subcategory",
        "product",
        "product_line",
        "brand",
        "channel",
        "segment",
        "year",
        "quarter",
        "month",
        "date",
    ]

    lower = {c.lower(): c for c in columns}
    return [lower[c] for c in wanted if c in lower]


def _find_metric_columns(df: pd.DataFrame) -> list[str]:
    # Prefer common retail metrics if present; otherwise fall back to all numeric.
    common = [
        "sales",
        "revenue",
        "amount",
        "profit",
        "margin",
        "units",
        "quantity",
    ]
    cols_lower = {c.lower(): c for c in df.columns}
    preferred = [cols_lower[c] for c in common if c in cols_lower]
    if preferred:
        return preferred

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    return numeric_cols


def _build_facts(df: pd.DataFrame) -> DataFacts:
    metric_cols = _find_metric_columns(df)
    dim_cols = _find_dimension_columns(df.columns.tolist())

    numeric_summary: dict[str, dict[str, float]] = {}
    if metric_cols:
        desc = df[metric_cols].describe(include="all")
        for col in metric_cols:
            numeric_summary[col] = {
                "count": float(desc.at["count", col]),
                "mean": float(desc.at["mean", col]),
                "min": float(desc.at["min", col]),
                "max": float(desc.at["max", col]),
            }

    group_summaries: dict[str, list[dict[str, Any]]] = {}
    if metric_cols and dim_cols:
        # Summarize top contributors for up to 2 dimensions.
        for dim in dim_cols[:2]:
            metric = metric_cols[0]
            grouped = (
                df.groupby(dim, dropna=False)[metric]
                .sum(numeric_only=True)
                .sort_values(ascending=False)
                .head(10)
            )
            group_summaries[dim] = [
                {"value": str(idx), f"sum_{metric}": float(val)}
                for idx, val in grouped.items()
            ]

    return DataFacts(
        row_count=int(len(df)),
        columns=[str(c) for c in df.columns.tolist()],
        numeric_summary=numeric_summary,
        group_summaries=group_summaries,
    )


def summarize_sales_csv(csv_path: Path) -> str:
    df = _read_csv_head(csv_path)
    facts = _build_facts(df)

    prompt = (
        "You are a retail analytics assistant. "
        "Write a concise business summary based ONLY on the provided facts. "
        "If the facts are insufficient, say what additional fields are needed.\n\n"
        f"Data preview row_count={facts.row_count}\n"
        f"Columns={facts.columns}\n\n"
        f"Numeric summary (count/mean/min/max)={facts.numeric_summary}\n\n"
        f"Top group summaries={facts.group_summaries}\n\n"
        "Output format:\n"
        "- 4-8 bullet points\n"
        "- Call out key drivers (top regions/categories) if available\n"
        "- Call out anomalies or data quality issues if you suspect them\n"
    )
    return generate_text(prompt)


def answer_sales_question_csv(csv_path: Path, question: str) -> str:
    df = _read_csv_head(csv_path)
    facts = _build_facts(df)

    prompt = (
        "You are a retail analytics assistant. "
        "Answer the user question using ONLY the provided facts. "
        "If you cannot answer, explain what query or data is required.\n\n"
        f"Question: {question}\n\n"
        f"Data preview row_count={facts.row_count}\n"
        f"Columns={facts.columns}\n\n"
        f"Numeric summary (count/mean/min/max)={facts.numeric_summary}\n\n"
        f"Top group summaries={facts.group_summaries}\n"
    )
    return generate_text(prompt)
