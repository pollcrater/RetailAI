from __future__ import annotations


def prompt_validator(*, question: str, sql_summary: str, results_json: str, provenance_json: str = "") -> str:
    provenance_block = f"Execution provenance: {provenance_json}\n\n" if provenance_json else ""
    return (
        "You are a retail analytics assistant. "
        "Answer the user's question in clear business language using ONLY the query results JSON. "
        "If results are empty or not enough, clearly say what additional query is needed.\n\n"
        f"Question: {question}\n\n"
        f"Query summary: {sql_summary}\n\n"
        f"{provenance_block}"
        f"Results JSON: {results_json}\n\n"
        "Output rules:\n"
        "- 5-10 bullet points max\n"
        "- Include the key numbers used in the answer\n"
        "- If uncertain, explicitly say so\n"
    )
