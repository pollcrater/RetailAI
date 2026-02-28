# RetailAI
RetailAI is a GenAI-powered retail insights assistant that enables conversational analytics and automated business summaries over structured sales data using a multi-agent architecture.

## Quickstart

1) Create and activate a venv

- PowerShell:
	- `py -m venv venv`
	- `venv\Scripts\Activate.ps1`

2) Install deps

- `pip install -r requirements.txt`

3) Configure environment

- Copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- Optional (behind Zscaler/VPN): set one of `SSL_CERT_FILE` / `REQUESTS_CA_BUNDLE`

4) Test your LLM connectivity

- `py scripts\test_openai.py`
- Azure OpenAI (LangChain):
	- `py scripts\test_azure_openai_langchain.py`

5) Use the shared LLM helper

- Implementation: [src/utils/llm.py](src/utils/llm.py)
- Example:
	- `py scripts\ask_llm.py "Summarize what this project does"`

## App entrypoint

Use [app.py](app.py) as the main command entrypoint:

- Summarize DuckDB (KPIs + coverage):
	- `py app.py summarize-db --db data\retail_sales.duckdb --table mart.sales_clean`
- Conversational Q&A over DuckDB (3-agent workflow):
	- `py app.py ask-db --question "Total sales by category?" --db data\retail_sales.duckdb --no-load`
- Save conversation context across runs (by conversation/thread id):
	- `py app.py ask-db --question "What is total sales?" --checkpoint-db data\checkpoints.db --thread-id user_123 --no-load`
- Optionally show planning metadata (rationale summary + confidence + tags):
	- `py app.py ask-db --question "Top categories" --thread-id user_123 --no-load --show-planner-meta`
- Ingest and clean a source CSV into analysis table:
	- `py app.py ingest --csv data\source\amazon_sale_report.csv --db data\retail_sales.duckdb --table mart.sales_clean`

## Streamlit UI (Chat Interface)

Run the chat-style UI:
- `streamlit run streamlit_app.py`

UI flow:
1) Upload a CSV file.
2) Ingestion pipeline runs automatically.
3) Choose **Summarize Report** or **Q&A**.
4) In Q&A mode, each chat uses a thread id with checkpointed memory.

## Assignment-aligned, streamlined flow

Business flow:
1) **Explore & profile data** → generate schema artifacts.
2) **Materialize raw tables in DuckDB** (one table per file; no schema mixing).
3) **Process uploads** → cleaned/normalized CSV → load to DuckDB (mart layer).
4) **Multi-agent QnA** (planner → executor → validator) over DuckDB.

Commands:
- Generate schema + DuckDB raw load SQL:
	- `py scripts\schema_analyzer.py --data-dir data\source --output-dir data\schema_analysis --model-mode separate`
- Process an uploaded CSV and load cleaned data to DuckDB (mart layer):
	- `py app.py ingest --csv data\source\amazon_sale_report.csv --table mart.sales_clean`
- Ask questions over DuckDB with LangGraph (3 agents):
	- `py app.py ask-db --question "How many rows are in raw.raw_amazon_sale_report?"`

## Monitoring & Evaluation (concise)

Track these per run:
- Accuracy: compare answers to SQL results or known benchmarks.
- Latency: total response time and per-agent timing.
- Cost: token usage and request counts.
- Robustness: count retries, failures, and fallback responses.

Fallback rules:
- If planner/executor fails twice, return a safe error + suggestion.
- If schema/metrics missing, summarize coverage + sample rows only.

## Deliverables Checklist

- Code implementation: multi-agent QnA + summarization
- Runs on sample CSVs
- Dependencies + setup in this README
- Architecture slides in [docs/ppt_slides.md](docs/ppt_slides.md)
- Example outputs (summaries + QnA) captured in terminal logs

## Assumptions, Limitations, Improvements

Assumptions:
- Input data is a CSV or structured report with consistent headers.
- DuckDB can store cleaned datasets locally for demo scale.

Limitations:
- LLM rate limits can throttle summaries in bursts.
- Some datasets lack numeric/date fields; summaries fall back to schema + samples.
- PNG diagram export may be blocked by corporate SSL; Mermaid is provided.

Artifacts:
- DuckDB raw load SQL: [data/schema_analysis/duckdb_load_raw.sql](data/schema_analysis/duckdb_load_raw.sql)
- LangGraph diagram (Mermaid): [data/schema_analysis/langgraph_qna.mmd](data/schema_analysis/langgraph_qna.mmd)
- LangGraph diagram (PNG): [data/schema_analysis/langgraph_qna.png](data/schema_analysis/langgraph_qna.png)
