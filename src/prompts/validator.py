from __future__ import annotations


def prompt_validator(*, question: str, sql: str, results_json: str) -> str:
    return (
        "You are a retail analytics assistant. "
        "Answer the user's question based ONLY on the query results JSON. "
        "If results are empty or insufficient, say what to query next.\n\n"
        f"Question: {question}\n\n"
        f"SQL: {sql}\n\n"
        f"Results JSON: {results_json}\n\n"
        "Output rules:\n"
        "- 5-10 bullet points max\n"
        "- Include the key numbers you used\n"
        "- If uncertain, explicitly say so\n"
    )
