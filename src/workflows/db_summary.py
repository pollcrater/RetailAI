from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import re
from src.tools.duckdb_tools import DuckDBRunner
from src.utils.llm import generate_text
from src.prompts import prompt_db_summary, prompt_comparison_summary
import logging


@dataclass(frozen=True)
class DbSummaryResult:
    table: str
    metrics: dict[str, Any]
    summary_text: str


def _pick_best_table(tables: list[str]) -> str | None:
    if "mart.sales_clean" in tables:
        return "mart.sales_clean"

    mart_tables = [t for t in tables if t.startswith("mart.")]
    if mart_tables:
        return mart_tables[0]

    raw_tables = [t for t in tables if t.startswith("raw.")]
    if raw_tables:
        return raw_tables[0]

    return tables[0] if tables else None


def _get_columns(runner: DuckDBRunner, table: str) -> list[str]:
    if "." in table:
        schema, name = table.split(".", 1)
    else:
        schema, name = "main", table
    safe_sql = (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}' ORDER BY ordinal_position"
    )
    res = runner.query(safe_sql)
    if res.ok:
        try:
            df = res.content
            cols = df["column_name"].tolist()
            return [str(c) for c in cols]
        except Exception:
            return []
    # Fallback for engines where information_schema may be restricted
    fallback_sql = f"SELECT name FROM pragma_table_info('{schema}.{name}')"
    res2 = runner.query(fallback_sql)
    if res2.ok:
        try:
            df = res2.content
            cols = df["name"].tolist()
            return [str(c) for c in cols]
        except Exception:
            return []
    return []


def _get_schema_info(runner: DuckDBRunner, table: str) -> list[dict[str, Any]]:
    if "." in table:
        schema, name = table.split(".", 1)
    else:
        schema, name = "main", table
    sql = (
        "SELECT column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema = '{schema}' AND table_name = '{name}' ORDER BY ordinal_position"
    )
    res = runner.query(sql)
    if res.ok and res.content is not None and not res.content.empty:
        return res.content.to_dict(orient="records")

    fallback_sql = f"SELECT name AS column_name, type AS data_type FROM pragma_table_info('{schema}.{name}')"
    res2 = runner.query(fallback_sql)
    if res2.ok and res2.content is not None and not res2.content.empty:
        return res2.content.to_dict(orient="records")

    return []


def _get_sample_rows(runner: DuckDBRunner, table: str, limit: int = 3) -> list[dict[str, Any]]:
    res = runner.query(f"SELECT * FROM {table} LIMIT {limit}")
    if not res.ok or res.content is None or res.content.empty:
        return []
    return res.content.to_dict(orient="records")


def _get_numeric_profile(runner: DuckDBRunner, table: str, schema_info: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_types = {"integer", "bigint", "smallint", "double", "float", "real", "decimal", "numeric"}
    numeric_cols = [
        c["column_name"]
        for c in schema_info
        if str(c.get("data_type", "")).lower().split("(")[0] in numeric_types
    ]
    if not numeric_cols:
        return {}

    profiles: dict[str, Any] = {}
    for col in numeric_cols:
        sql = (
            f"SELECT MIN({col}) AS min_val, MAX({col}) AS max_val, AVG({col}) AS avg_val "
            f"FROM {table}"
        )
        res = runner.query(sql)
        if res.ok and res.content is not None and not res.content.empty:
            row = res.content.iloc[0]
            profiles[col] = {
                "min": row.get("min_val"),
                "max": row.get("max_val"),
                "avg": row.get("avg_val"),
            }
    return profiles


def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _detect_columns(columns: list[str]) -> dict[str, str | None]:
    return {
        "date": _find_col(
            columns,
            ["date_parsed", "date", "order_date", "sale_date", "created_at", "invoice_date"],
        ),
        "amount": _find_col(columns, ["amount", "total", "total_amount", "sales", "revenue", "gross"]),
        "quantity": _find_col(columns, ["quantity", "qty", "units", "units_sold"]),
        "category": _find_col(columns, ["category", "product_category", "class", "type"]),
        "channel": _find_col(columns, ["channel", "sales_channel", "platform", "marketplace"]),
        "region": _find_col(columns, ["region", "state", "city", "country"]),
    }


def _extract_price(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    matches = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    if not matches:
        return None
    try:
        return float(matches[0])
    except Exception:
        return None


def _comparison_strategy(runner: DuckDBRunner, table: str, columns: list[str]) -> tuple[dict[str, Any], str] | None:
    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    norm_map = {_norm(c): c for c in columns}
    if not {"shiprocket", "increff"}.issubset(set(norm_map.keys())):
        return None

    res = runner.query(f"SELECT * FROM {table}")
    if not res.ok or res.content is None or res.content.empty:
        return None

    df = res.content.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Drop unnamed/index columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "index" in [c.lower() for c in df.columns]:
        drop_cols += [c for c in df.columns if c.lower() == "index"]
    if drop_cols:
        df = df.drop(columns=list(set(drop_cols)))

    if df.empty or len(df.columns) < 3:
        return None

    label_col = df.columns[0]
    provider_cols = df.columns[1:]

    # Remove header-like row if present
    header_mask = df[label_col].astype(str).str.contains("Heads|Price", case=False, na=False)
    if header_mask.any():
        df = df[~header_mask]

    # Parse prices
    for col in provider_cols:
        df[col] = df[col].map(_extract_price)

    df = df.dropna(subset=provider_cols, how="all")
    if df.empty:
        return None

    ship_col = norm_map["shiprocket"]
    inc_col = norm_map["increff"]
    if ship_col in df.columns and inc_col in df.columns:
        df["diff"] = df[inc_col] - df[ship_col]
        cheaper_ship = int((df["diff"] > 0).sum())
        cheaper_inc = int((df["diff"] < 0).sum())
        ties = int((df["diff"] == 0).sum())
        avg_diff = float(df["diff"].mean()) if not df["diff"].isna().all() else None
    else:
        cheaper_ship = cheaper_inc = ties = 0
        avg_diff = None

    metrics = {
        "table": table,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "row_count": int(len(df)),
        "providers": provider_cols,
        "avg_prices": {
            col: float(df[col].mean()) if not df[col].isna().all() else None for col in provider_cols
        },
        "min_prices": {
            col: float(df[col].min()) if not df[col].isna().all() else None for col in provider_cols
        },
        "max_prices": {
            col: float(df[col].max()) if not df[col].isna().all() else None for col in provider_cols
        },
        "cheaper_counts": {
            "shiprocket": cheaper_ship,
            "increff": cheaper_inc,
            "ties": ties,
        },
        "avg_price_diff_increff_minus_shiprocket": avg_diff,
    }

    return metrics, prompt_comparison_summary(metrics=metrics)


def _query_single_value(runner: DuckDBRunner, sql: str, key: str) -> Any:
    res = runner.query(sql)
    if not res.ok:
        return None
    df = res.content
    if df is None or df.empty:
        return None
    return df.iloc[0][key]


def _compute_trends(runner: DuckDBRunner, table: str, date_col: str, amount_col: str) -> dict[str, Any]:
    date_expr = f"TRY_CAST({date_col} AS DATE)"
    sql = (
        "SELECT date_trunc('month', "
        f"{date_expr}) AS month, SUM({amount_col}) AS total "
        f"FROM {table} "
        f"WHERE {date_expr} IS NOT NULL "
        "GROUP BY 1 ORDER BY 1"
    )
    res = runner.query(sql)
    if not res.ok:
        return {}
    df = res.content
    if df is None or df.empty:
        return {}

    df = df.dropna(subset=["month"]).sort_values("month")
    if df.empty:
        return {}

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else None
    delta = None
    delta_pct = None
    if prev_row is not None and prev_row["total"] not in (0, None):
        delta = float(last_row["total"] - prev_row["total"])
        delta_pct = float(delta / prev_row["total"] * 100)

    months_included = [str(m.date()) for m in df["month"].tolist()]

    return {
        "monthly_trend": df.tail(12).to_dict(orient="records"),
        "months_included": months_included,
        "last_month": str(last_row["month"]) if "month" in df.columns else None,
        "last_month_total": float(last_row["total"]),
        "prev_month_total": float(prev_row["total"]) if prev_row is not None else None,
        "delta": delta,
        "delta_pct": delta_pct,
    }


def _get_top_dimension(
    runner: DuckDBRunner,
    table: str,
    dim_col: str,
    amount_col: str | None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if amount_col:
        sql = (
            f"SELECT {dim_col} AS label, SUM({amount_col}) AS total "
            f"FROM {table} GROUP BY 1 ORDER BY total DESC LIMIT {limit}"
        )
    else:
        sql = f"SELECT {dim_col} AS label, COUNT(*) AS total FROM {table} GROUP BY 1 ORDER BY total DESC LIMIT {limit}"

    res = runner.query(sql)
    if not res.ok:
        return []
    df = res.content
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def run_db_summary(*, db_path: Path, table: str | None = None) -> DbSummaryResult:
    runner = DuckDBRunner(db_path=db_path)
    tables_res = runner.list_tables()
    tables = tables_res.content if tables_res.ok else []
    chosen = table or _pick_best_table(list(tables) if isinstance(tables, list) else [])
    if not chosen:
        return DbSummaryResult(table="", metrics={}, summary_text="No tables found to summarize.")

    logger = logging.getLogger(__name__)
    logger.info("DB summary start: table=%s", chosen)

    columns = _get_columns(runner, chosen)
    schema_info = _get_schema_info(runner, chosen)
    col_map = _detect_columns(columns)

    # Specialized strategies (open/closed: add new strategies without modifying core flow)
    comparison = _comparison_strategy(runner, chosen, columns)
    if comparison is not None:
        metrics, prompt = comparison
        summary = generate_text(prompt)
        logger.info("DB summary complete")
        return DbSummaryResult(table=chosen, metrics=metrics, summary_text=summary)

    metrics: dict[str, Any] = {
        "table": chosen,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "filters_applied": [],
        "schema": schema_info,
        "sample_rows": _get_sample_rows(runner, chosen, limit=3),
    }

    numeric_profile = _get_numeric_profile(runner, chosen, schema_info)
    if numeric_profile:
        metrics["numeric_profile"] = numeric_profile

    metrics["row_count"] = _query_single_value(runner, f"SELECT COUNT(*) AS row_count FROM {chosen}", "row_count")
    logger.info("DB summary metric: row_count")

    amount_col = col_map.get("amount")
    if amount_col:
        metrics["total_sales"] = _query_single_value(
            runner, f"SELECT SUM({amount_col}) AS total_sales FROM {chosen}", "total_sales"
        )
        metrics["avg_order"] = _query_single_value(
            runner, f"SELECT AVG({amount_col}) AS avg_order FROM {chosen}", "avg_order"
        )
        logger.info("DB summary metric: sales_totals")

    qty_col = col_map.get("quantity")
    if qty_col:
        metrics["total_units"] = _query_single_value(
            runner, f"SELECT SUM({qty_col}) AS total_units FROM {chosen}", "total_units"
        )
        logger.info("DB summary metric: units_total")

    date_col = col_map.get("date")
    if date_col and amount_col:
        metrics.update(_compute_trends(runner, chosen, date_col, amount_col))
        logger.info("DB summary metric: monthly_trend")

    if date_col:
        date_expr = f"TRY_CAST({date_col} AS DATE)"
        date_sql = (
            f"SELECT MIN({date_expr}) AS min_date, MAX({date_expr}) AS max_date "
            f"FROM {chosen} WHERE {date_expr} IS NOT NULL"
        )
        res = runner.query(date_sql)
        if res.ok and res.content is not None and not res.content.empty:
            row = res.content.iloc[0]
            metrics["date_range"] = {
                "start": str(row["min_date"]) if row["min_date"] is not None else None,
                "end": str(row["max_date"]) if row["max_date"] is not None else None,
            }
            logger.info("DB summary metric: date_range")

    for dim_key in ("category", "channel", "region"):
        dim_col = col_map.get(dim_key)
        if dim_col:
            metrics[f"top_{dim_key}"] = _get_top_dimension(runner, chosen, dim_col, amount_col)
            logger.info("DB summary metric: top_%s", dim_key)

    prompt = prompt_db_summary(metrics=metrics)

    summary = generate_text(prompt)
    logger.info("DB summary complete")
    return DbSummaryResult(table=chosen, metrics=metrics, summary_text=summary)
