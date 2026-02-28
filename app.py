from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import get_settings
from src.utils.logging import configure_logging
from src.workflows.db_qna_graph import run_db_qna
from src.workflows.db_summary import run_db_summary
from src.workflows.data_pipeline import run_processing_pipeline


def main() -> int:
    settings = get_settings()
    configure_logging(settings)

    parser = argparse.ArgumentParser(prog="retailai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sumdb = sub.add_parser("summarize-db", help="Generate KPI summary from DuckDB data")
    p_sumdb.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="DuckDB database path (default: settings.db_path)",
    )
    p_sumdb.add_argument(
        "--table",
        default=None,
        help="Table to summarize (default: auto-select mart.sales_clean when available)",
    )

    p_db = sub.add_parser("ask-db", help="Ask a business question over DuckDB using a 3-agent workflow")
    p_db.add_argument("--question", required=True)
    p_db.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="DuckDB database path (default: settings.db_path)",
    )
    p_db.add_argument(
        "--load-raw-sql",
        type=Path,
        default=settings.project_root / "data" / "schema_analysis" / "duckdb_load_raw.sql",
        help="SQL script to rebuild source raw tables (default: data/schema_analysis/duckdb_load_raw.sql)",
    )
    p_db.add_argument(
        "--no-load",
        action="store_true",
        help="Skip rebuilding source raw tables before running the query",
    )
    p_db.add_argument(
        "--checkpoint-db",
        type=Path,
        default=settings.project_root / "data" / "checkpoints.db",
        help="SQLite file used to save conversation context and workflow state",
    )
    p_db.add_argument(
        "--thread-id",
        default="default",
        help="Conversation ID used to resume context across runs",
    )
    p_db.add_argument(
        "--stream",
        choices=["updates", "tasks", "messages", "checkpoints", "debug"],
        default=None,
        help="Optional step-by-step event stream mode",
    )
    p_db.add_argument(
        "--show-planner-meta",
        action="store_true",
        help="Show planning notes, confidence, and tags in output",
    )

    p_ingest = sub.add_parser(
        "ingest",
        help="Clean a CSV file and load analysis-ready data into DuckDB",
    )
    p_ingest.add_argument("--csv", type=Path, required=True)
    p_ingest.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="DuckDB database path (default: settings.db_path)",
    )
    p_ingest.add_argument(
        "--table",
        default="mart.sales_clean",
        help="Target table name for cleaned data (default: mart.sales_clean)",
    )

    args = parser.parse_args()

    if args.cmd == "summarize-db":
        db_path: Path = args.db
        logging.getLogger(__name__).info("Summarizing DuckDB at %s", db_path)
        result = run_db_summary(db_path=db_path, table=args.table)
        print(result.summary_text)
        return 0

    if args.cmd == "ask-db":
        db_path: Path = args.db
        load_sql: Path = args.load_raw_sql
        ensure_sql = None if args.no_load else load_sql
        if ensure_sql is not None and not ensure_sql.exists():
            raise SystemExit(
                f"Raw-table load SQL not found: {ensure_sql}. Run scripts/schema_analyzer.py to generate it."
            )
        logging.getLogger(__name__).info("Answering question using DuckDB at %s", db_path)
        print(
            run_db_qna(
                question=args.question,
                db_path=db_path,
                ensure_raw_sql_file=ensure_sql,
                stream_mode=args.stream,
                checkpoint_db_path=args.checkpoint_db,
                thread_id=args.thread_id,
                show_planner_meta=args.show_planner_meta,
            )
        )
        return 0

    if args.cmd == "ingest":
        csv_path: Path = args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV file not found: {csv_path}")
        db_path: Path = args.db
        logging.getLogger(__name__).info("Processing upload %s", csv_path)
        result = run_processing_pipeline(
            csv_path=csv_path,
            db_path=db_path,
            table_name=str(args.table),
        )
        print(f"Processed CSV: {result.processed_csv}")
        print(f"Summary JSON: {result.summary_json}")
        print(f"Loaded table: {result.table_name} (rows={result.row_count})")
        return 0

    raise SystemExit("Unknown command. Use -h to see available commands.")


if __name__ == "__main__":
    raise SystemExit(main())
