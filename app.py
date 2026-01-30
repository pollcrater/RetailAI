from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import get_settings
from src.utils.logging import configure_logging
from src.utils.llm import generate_text
from src.workflows.retail_assistant import answer_sales_question_csv, summarize_sales_csv


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

    p_ask = sub.add_parser("ask", help="Ask a question about a sales CSV")
    p_ask.add_argument("--csv", type=Path, required=True)
    p_ask.add_argument("--question", required=True)

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

    if args.cmd == "ask":
        csv_path: Path = args.csv
        if not csv_path.exists():
            raise SystemExit(f"CSV not found: {csv_path}")
        logging.getLogger(__name__).info("Answering question for %s", csv_path)
        print(answer_sales_question_csv(csv_path, args.question))
        return 0

    raise SystemExit("Unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
