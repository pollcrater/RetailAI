# PPT Slides — Retail Insights Assistant

## Slide 1 — Title
**Retail Insights Assistant (GenAI + Scalable Data System)**
- Your name / team
- Course / assignment
- Date

## Slide 2 — Problem & Goals
- Retail teams need fast, conversational analytics.
- Goals: summarize performance + answer ad‑hoc questions.
- Must be reliable, scalable, and explainable.

## Slide 3 — Solution Overview
- End‑to‑end pipeline: ingest → clean → DuckDB → LLM insights.
- Two modes: Summarization and QnA.
- Multi‑agent workflow for accuracy and safety.

## Slide 4 — Architecture (Core Flow)
- Input: CSV or report.
- Preprocess + normalize.
- Store in DuckDB (raw + mart).
- LangGraph 3‑agent QnA.
- Output: analyst‑friendly insights.

## Slide 5 — Multi‑Agent Design
- Planner: NL → SQL plan (single or multi‑query).
- Executor: run SQL safely.
- Validator: craft final answer and quality checks.
- Retry loop on failures.

## Slide 6 — Data Layer & Processing
- Pandas + DuckDB for structured analytics.
- Raw tables per file (no schema mixing).
- Cleaned mart tables for QnA and summaries.

## Slide 7 — Summarization Mode
- KPI aggregation, trends, deltas.
- Data coverage + filters.
- Falls back to schema + samples if KPIs missing.

## Slide 8 — QnA Mode (Example)
- User: “Which category drove growth?”
- Planner generates multiple queries.
- Executor runs queries.
- Validator returns concise business answer.

## Slide 9 — Monitoring & Reliability
- Metrics: accuracy, latency, cost, retries.
- Safe SQL enforcement.
- Retry/backoff on transient API errors.
- Memory for multi‑turn context.

## Slide 10 — Scalability (Design Notes)
- Partitioned storage (parquet/lake/warehouse).
- Metadata filtering + vector search.
- Batch/stream preprocessing (Spark/Dask).

## Slide 11 — Demo / Results
- Summaries for amazon_sale_report.csv.
- QnA examples with DuckDB results.
- Comparison‑table summaries.

## Slide 12 — Next Steps
- UI (Streamlit/Gradio).
- Advanced insights (YoY, cohort).
- Production monitoring + caching.
