from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import get_settings
from src.utils.logging import configure_logging
from src.utils.llm import generate_text
from src.workflows.retail_assistant import answer_sales_question_csv, summarize_sales_csv
from src.workflows.db_qna_graph import run_db_qna
from src.workflows.db_summary import run_db_summary
from src.workflows.data_pipeline import run_processing_pipeline


def _mask_secret(value: str | None) -> str:
    if not value:
        return "<not set>"
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def main() -> int:
    settings = get_settings()
    configure_logging(settings)

    parser = argparse.ArgumentParser(prog="retailai")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("config", help="Print resolved configuration")

    p_llm = sub.add_parser("llm", help="Send a raw prompt to the LLM")
    p_llm.add_argument("prompt", nargs="+", help="Prompt text")

    p_sum = sub.add_parser("summarize", help="Summarize a sales CSV")
    p_sum.add_argument("--csv", type=Path, required=True)

    p_sumdb = sub.add_parser("summarize-db", help="Summarize data using DuckDB aggregates")
    p_sumdb.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="Path to DuckDB database file (default: settings.db_path)",
    )
    p_sumdb.add_argument(
        "--table",
        default=None,
        help="Target table to summarize (default: auto-pick mart.sales_clean if present)",
    )

    p_ask = sub.add_parser("ask", help="Ask a question about a sales CSV")
    p_ask.add_argument("--csv", type=Path, required=True)
    p_ask.add_argument("--question", required=True)

    p_db = sub.add_parser("ask-db", help="Ask a question using DuckDB + multi-agent LangGraph")
    p_db.add_argument("--question", required=True)
    p_db.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="Path to DuckDB database file (default: settings.db_path)",
    )
    p_db.add_argument(
        "--load-raw-sql",
        type=Path,
        default=settings.project_root / "data" / "schema_analysis" / "duckdb_load_raw.sql",
        help="SQL file to (re)create raw.* tables (default: data/schema_analysis/duckdb_load_raw.sql)",
    )
    p_db.add_argument(
        "--no-load",
        action="store_true",
        help="Skip loading raw tables before answering",
    )
    p_db.add_argument(
        "--memory-file",
        type=Path,
        default=None,
        help="Optional JSON file to persist QnA memory across runs",
    )
    p_db.add_argument(
        "--stream",
        choices=["updates", "tasks", "messages", "checkpoints", "debug"],
        default=None,
        help="Optional LangGraph stream mode for step-by-step events",
    )

    p_ingest = sub.add_parser(
        "ingest",
        help="Process an uploaded CSV and load cleaned data into DuckDB (mart layer)",
    )
    p_ingest.add_argument("--csv", type=Path, required=True)
    p_ingest.add_argument(
        "--db",
        type=Path,
        default=settings.db_path,
        help="Path to DuckDB database file (default: settings.db_path)",
    )
    p_ingest.add_argument(
        "--table",
        default="mart.sales_clean",
        help="Target table name for cleaned data (default: mart.sales_clean)",
    )

    args = parser.parse_args()

    if args.cmd == "config":
        print(f"app_name={settings.app_name}")
        print(f"app_env={settings.app_env}")
        print(f"log_level={settings.log_level}")
        print(f"data_dir={settings.data_dir}")
        print(f"output_dir={settings.output_dir}")
        print(f"db_path={settings.db_path}")
        print(f"openai_model={settings.openai_model}")
        print(f"openai_base_url={settings.openai_base_url or '<default>'}")
        print(f"openai_api_key={_mask_secret(settings.openai_api_key)}")
        return 0

    if args.cmd == "llm":
        prompt = " ".join(args.prompt).strip()
        print(generate_text(prompt))
        return 0

    if args.cmd == "summarize":
        csv_path: Path = args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        logging.getLogger(__name__).info("Summarizing %s", csv_path)
        print(summarize_sales_csv(csv_path))
        return 0

    if args.cmd == "summarize-db":
        db_path: Path = args.db
        logging.getLogger(__name__).info("Summarizing DuckDB at %s", db_path)
        result = run_db_summary(db_path=db_path, table=args.table)
        print(result.summary_text)
        return 0

    if args.cmd == "ask":
        csv_path: Path = args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        logging.getLogger(__name__).info("Answering question for %s", csv_path)
        print(answer_sales_question_csv(csv_path, args.question))
        return 0

    if args.cmd == "ask-db":
        db_path: Path = args.db
        load_sql: Path = args.load_raw_sql
        ensure_sql = None if args.no_load else load_sql
        if ensure_sql is not None and not ensure_sql.exists():
            raise SystemExit(
                f"Raw load SQL not found: {ensure_sql}. Run scripts/schema_analyzer.py to generate it."
            )
        logging.getLogger(__name__).info("Answering question using DuckDB at %s", db_path)
        print(
            run_db_qna(
                question=args.question,
                db_path=db_path,
                ensure_raw_sql_file=ensure_sql,
                memory_file=args.memory_file,
                stream_mode=args.stream,
            )
        )
        return 0

    if args.cmd == "ingest":
        csv_path: Path = args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
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

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
