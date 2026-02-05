from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

"""Upload → clean/normalize → load to DuckDB (mart layer)."""

# Canonical cleaning/normalization pipeline lives in scripts/preprocess.py
from scripts.preprocess import DataPreprocessor, save_processed_data
from src.config import get_settings
from src.tools.duckdb_tools import DuckDBRunner


@dataclass(frozen=True)
class PipelineResult:
    processed_csv: Path
    summary_json: Path
    table_name: str
    row_count: int


def _ensure_schema_sql(table_name: str) -> tuple[str, str]:
    if "." in table_name:
        schema, table = table_name.split(".", 1)
        return schema, table
    return "mart", table_name


def process_uploaded_csv(*, csv_path: Path, output_dir: Path | None = None) -> tuple[Path, Path]:
    settings = get_settings()
    if output_dir is None:
        output_dir = settings.project_root / "data" / "processed"

    preprocessor = DataPreprocessor()
    df_processed, summary = preprocessor.process(str(csv_path))

    processed_csv, summary_json = save_processed_data(
        df_processed,
        summary,
        output_dir=str(output_dir),
        output_basename=csv_path.stem,
    )

    return Path(processed_csv), Path(summary_json)


def load_processed_to_duckdb(
    *,
    processed_csv: Path,
    db_path: Path,
    table_name: str = "mart.sales_clean",
) -> int:
    schema, table = _ensure_schema_sql(table_name)
    runner = DuckDBRunner(db_path=db_path)

    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};
    CREATE OR REPLACE TABLE {schema}.{table} AS
    SELECT *
    FROM read_csv_auto('{processed_csv.as_posix()}', header=true);
    """.strip()

    res = runner.execute(sql)
    if not res.ok:
        raise RuntimeError(res.error or "Failed to load processed data into DuckDB")

    count = runner.query(f"SELECT COUNT(*) AS n FROM {schema}.{table}")
    if not count.ok:
        raise RuntimeError(count.error or "Failed to count rows after load")

    df = count.content
    row_count = int(df.iloc[0]["n"]) if df is not None else 0
    return row_count


def run_processing_pipeline(
    *,
    csv_path: Path,
    db_path: Path,
    table_name: str = "mart.sales_clean",
    output_dir: Path | None = None,
) -> PipelineResult:
    
    processed_csv, summary_json = process_uploaded_csv(csv_path=csv_path, output_dir=output_dir)
    row_count = load_processed_to_duckdb(
        processed_csv=processed_csv,
        db_path=db_path,
        table_name=table_name,
    )

    return PipelineResult(
        processed_csv=processed_csv,
        summary_json=summary_json,
        table_name=table_name,
        row_count=row_count,
    )
