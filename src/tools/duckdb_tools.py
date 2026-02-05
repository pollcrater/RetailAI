from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import duckdb
import logging

from src.config import get_settings
from src.tools.base import ToolResult


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_db_path(db_path: Path | None) -> Path:
    settings = get_settings()
    path = db_path or settings.db_path
    if not path.is_absolute():
        path = (settings.project_root / path).resolve()
    return path


def _connect(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    path = _normalize_db_path(db_path)
    _ensure_parent_dir(path)
    return duckdb.connect(str(path))


def _is_safe_select_sql(sql: str) -> bool:
    s = sql.strip().lower()
    if not s:
        return False

    # Allow WITH ... SELECT ...
    if s.startswith("with "):
        return True

    # Only allow SELECT as the main statement.
    if not s.startswith("select "):
        return False

    forbidden = [
        "insert ",
        "update ",
        "delete ",
        "drop ",
        "alter ",
        "create ",
        "attach ",
        "copy ",
        "export ",
        "pragma ",
        "call ",
    ]
    return not any(tok in s for tok in forbidden)


@dataclass(frozen=True)
class DuckDBRunner:
    """Thin wrapper around DuckDB for tools/agents."""

    db_path: Path | None = None

    def execute(self, sql: str) -> ToolResult:
        try:
            logging.getLogger(__name__).debug("DuckDB execute")
            con = _connect(self.db_path)
            try:
                con.execute(sql)
                return ToolResult(ok=True, content=True)
            finally:
                con.close()
        except Exception as e:
            logging.getLogger(__name__).error("DuckDB execute failed: %s", e)
            return ToolResult(ok=False, content=None, error=str(e))

    def query(self, sql: str, *, max_rows: int = 200) -> ToolResult:
        if not _is_safe_select_sql(sql):
            return ToolResult(ok=False, content=None, error="Only SELECT queries are allowed")

        try:
            logging.getLogger(__name__).debug("DuckDB query")
            con = _connect(self.db_path)
            try:
                df = con.execute(sql).fetch_df()
            finally:
                con.close()

            if len(df) > max_rows:
                df = df.head(max_rows)
            return ToolResult(ok=True, content=df)
        except Exception as e:
            logging.getLogger(__name__).error("DuckDB query failed: %s", e)
            return ToolResult(ok=False, content=None, error=str(e))

    def list_tables(self) -> ToolResult:
        try:
            con = _connect(self.db_path)
            try:
                df = con.execute(
                    """
                    SELECT table_schema, table_name
                    FROM information_schema.tables
                    WHERE table_type = 'BASE TABLE'
                    ORDER BY table_schema, table_name
                    """
                ).fetch_df()
            finally:
                con.close()

            tables = [f"{r.table_schema}.{r.table_name}" for r in df.itertuples(index=False)]
            return ToolResult(ok=True, content=tables)
        except Exception as e:
            return ToolResult(ok=False, content=None, error=str(e))

    def get_schema_markdown(self, *, schemas: Iterable[str] = ("raw", "mart")) -> ToolResult:
        try:
            con = _connect(self.db_path)
            try:
                rows = con.execute(
                    """
                    SELECT table_schema, table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = ANY(?::VARCHAR[])
                    ORDER BY table_schema, table_name, ordinal_position
                    """,
                    [list(schemas)],
                ).fetchall()
            finally:
                con.close()

            md = []
            current = None
            for table_schema, table_name, column_name, data_type in rows:
                t = f"{table_schema}.{table_name}"
                if t != current:
                    md.append(f"\n### {t}\n")
                    md.append("| column | type |\n|---|---|\n")
                    current = t
                md.append(f"| {column_name} | {data_type} |\n")
            return ToolResult(ok=True, content="".join(md).strip() or "<no tables>")
        except Exception as e:
            return ToolResult(ok=False, content=None, error=str(e))


def load_raw_from_sql_file(
    *,
    sql_file: Path,
    db_path: Path | None = None,
) -> ToolResult:
    """Execute a SQL file (typically duckdb_load_raw.sql) against the DB."""

    try:
        sql_text = sql_file.read_text(encoding="utf-8")
    except Exception as e:
        return ToolResult(ok=False, content=None, error=f"Failed reading {sql_file}: {e}")

    runner = DuckDBRunner(db_path=db_path)
    return runner.execute(sql_text)


def dataframe_to_json_preview(df: Any, *, max_rows: int = 50) -> str:
    """Convert a pandas df (or compatible) to JSON for LLM consumption."""

    try:
        if hasattr(df, "head"):
            df2 = df.head(max_rows)
        else:
            df2 = df
        return json.dumps({"rows": df2.to_dict(orient="records")}, default=str)  # type: ignore[attr-defined]
    except Exception:
        return json.dumps({"rows": []})
