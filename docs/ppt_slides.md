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

## Slide 4 — LLM Integration Strategy
- OpenAI API via shared LLM wrapper
- Prompt library for consistency
- Retry/backoff for transient failures

## Slide 5 — Architecture (Core Flow)
- Input: CSV or report.
- Preprocess + normalize.
- Store in DuckDB (raw + mart).
- LangGraph 3‑agent QnA.
- Output: analyst‑friendly insights.

## Slide 6 — Multi‑Agent Design
- Planner: NL → SQL plan (single or multi‑query).
- Executor: run SQL safely.
- Validator: craft final answer and quality checks.
- Retry loop on failures.

## Slide 7 — Data Layer & Processing
- Pandas + DuckDB for structured analytics.
- Raw tables per file (no schema mixing).
- Cleaned mart tables for QnA and summaries.

## Slide 8 — Summarization Mode
- KPI aggregation, trends, deltas.
- Data coverage + filters.
- Falls back to schema + samples if KPIs missing.

## Slide 9 — QnA Mode (Example)
- User: “Which category drove growth?”
- Planner generates multiple queries.
- Executor runs queries.
- Validator returns concise business answer.

## Slide 10 — Example Query → Response Pipeline
- User question → planner SQL(s)
- DuckDB execution → results JSON
- Validator → final analyst response

## Slide 11 — Monitoring & Reliability
- Metrics: accuracy, latency, cost, retries.
- Safe SQL enforcement.
- Retry/backoff on transient API errors.
- Memory for multi‑turn context.

## Slide 12 — Code Quality & Review Focus
- Readability and maintainability
- Modularity and reusability
- Linting / standard checks
- Scalability and security

## Slide 13 — Scalability (Design Notes)
- Partitioned storage (parquet/lake/warehouse).
- Metadata filtering + vector search.
- Batch/stream preprocessing (Spark/Dask).

## Slide 14 — Demo / Results
- Summaries for amazon_sale_report.csv.
- QnA examples with DuckDB results.
- Comparison‑table summaries.

## Slide 12 — Next Steps
- UI (Streamlit/Gradio).
- Advanced insights (YoY, cohort).
- Production monitoring + caching.
