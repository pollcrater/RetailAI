from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import logging

from src.agents.planner import SqlPlannerAgent
from src.agents.executor import SqlExecutorAgent
from src.agents.validator import ResultValidatorAgent
from src.tools.duckdb_tools import DuckDBRunner, load_raw_from_sql_file, dataframe_to_json_preview


class QnaState(TypedDict, total=False):
    question: str

    # memory (lightweight)
    memory: list[dict[str, Any]]
    resolved_filters: dict[str, Any]
    memory_size: int

    # planner outputs
    sql: str
    queries: list[str]
    planner_meta: dict[str, Any]
    planner_rationale_summary: str
    planner_confidence: float | None
    planner_reason_tags: list[str]
    planner_feedback: str

    # executor outputs
    query_result_df: Any
    query_result_dfs: list[Any]
    query_result_preview: str
    query_result_previews: list[str]
    query_execution_meta: list[dict[str, Any]]
    query_errors: list[dict[str, Any]]
    tool_error: str

    # validator outputs
    answer: str

    # orchestration
    iterations: int
    done: bool

    # deps
    duckdb_runner: DuckDBRunner


@dataclass(frozen=True)
class DbQnaWorkflow:
    """LangGraph workflow for DB-backed QnA using 3 agents.

    Agents:
    - planner: NL -> SQL
    - executor: SQL -> dataframe
    - validator: dataframe -> final answer (or feedback loop)
    """

    max_iterations: int = 2
    memory_size: int = 5

    def build(self, *, checkpointer: SqliteSaver | None = None):
        planner = SqlPlannerAgent()
        executor = SqlExecutorAgent()
        validator = ResultValidatorAgent()

        logger = logging.getLogger(__name__)

        graph = StateGraph(QnaState)

        def plan_node(state: QnaState) -> QnaState:
            if "memory" not in state:
                state = {**state, "memory": [], "resolved_filters": {}, "memory_size": self.memory_size}
            logger.info("Planner start: question=%s", state.get("question", ""))
            result = planner.run(dict(state))
            if not result.ok:
                logger.error("Planner failed: %s", result.error or "unknown error")
                return {
                    **state,
                    "tool_error": result.error or "Planner failed",
                    "planner_feedback": str(result.content),
                    "done": True,
                }
            logger.info("Planner ok: sql_len=%s", len(result.content.get("sql", "")))
            return {
                **state,
                "sql": result.content["sql"],
                "queries": result.content.get("queries") or [],
                "planner_meta": result.content.get("meta", {}),
                "planner_rationale_summary": str(
                    result.content.get("meta", {}).get("rationale_summary", "")
                ).strip(),
                "planner_confidence": result.content.get("meta", {}).get("confidence"),
                "planner_reason_tags": result.content.get("meta", {}).get("reason_tags") or [],
                "tool_error": "",
            }

        def execute_node(state: QnaState) -> QnaState:
            logger.info("Executor start")
            result = executor.run(dict(state))
            if not result.ok:
                logger.error("Executor failed: %s", result.error or "unknown error")
                return {
                    **state,
                    "tool_error": result.error or "Query failed",
                    "query_result_df": None,
                    "query_result_dfs": None,
                    "query_result_preview": "",
                    "query_result_previews": [],
                    "query_execution_meta": result.content.get("execution_meta", []) if isinstance(result.content, dict) else [],
                    "query_errors": result.content.get("query_errors", []) if isinstance(result.content, dict) else [],
                }
            logger.info("Executor ok")
            if "dfs" in result.content:
                previews = [dataframe_to_json_preview(d) for d in result.content["dfs"]]
                return {
                    **state,
                    "tool_error": "",
                    "query_result_dfs": None,
                    "query_result_df": None,
                    "query_result_preview": "",
                    "query_result_previews": previews,
                    "query_execution_meta": result.content.get("execution_meta", []),
                    "query_errors": result.content.get("query_errors", []),
                }
            preview = dataframe_to_json_preview(result.content["df"])
            return {
                **state,
                "tool_error": "",
                "query_result_df": None,
                "query_result_dfs": None,
                "query_result_preview": preview,
                "query_result_previews": [],
                "query_execution_meta": result.content.get("execution_meta", []),
                "query_errors": result.content.get("query_errors", []),
            }

        def validate_node(state: QnaState) -> QnaState:
            logger.info("Validator start")
            result = validator.run(dict(state))
            if result.ok:
                updated_state = {**state, "answer": result.content["answer"], "done": True}
                memory = list(updated_state.get("memory", []))
                max_size = int(updated_state.get("memory_size", self.memory_size))
                planner_meta = updated_state.get("planner_meta", {}) or {}
                filters = planner_meta.get("filters") or planner_meta.get("resolved_filters") or {}
                memory.append(
                    {
                        "question": updated_state.get("question", ""),
                        "answer": updated_state.get("answer", ""),
                        "sql": updated_state.get("sql", ""),
                        "filters": filters,
                    }
                )
                if len(memory) > max_size:
                    memory = memory[-max_size:]
                updated_state["memory"] = memory
                if isinstance(filters, dict) and filters:
                    merged = dict(updated_state.get("resolved_filters", {}) or {})
                    merged.update(filters)
                    updated_state["resolved_filters"] = merged

                logger.info(
                    "QnA memory updated: size=%s filters=%s",
                    len(memory),
                    updated_state.get("resolved_filters", {}),
                )
                logger.info("Validator ok")
                return updated_state

            # feedback loop
            it = int(state.get("iterations", 0))
            feedback = ""
            if isinstance(result.content, dict):
                feedback = str(result.content.get("planner_feedback", ""))
            logger.warning("Validator requested retry: iteration=%s", it + 1)
            return {
                **state,
                "planner_feedback": feedback,
                "iterations": it + 1,
                "done": False,
            }

        def route_after_validate(state: QnaState) -> Literal["plan", "end"]:
            if state.get("done"):
                return "end"
            if int(state.get("iterations", 0)) >= self.max_iterations:
                return "end"
            return "plan"

        graph.add_node("plan", plan_node)
        graph.add_node("execute", execute_node)
        graph.add_node("validate", validate_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "validate")
        graph.add_conditional_edges("validate", route_after_validate, {"plan": "plan", "end": END})

        return graph.compile(checkpointer=checkpointer)


def ensure_raw_loaded(*, db_path: Path, load_sql_file: Path) -> None:
    """Idempotent-ish raw load: execute load SQL to create raw tables."""

    res = load_raw_from_sql_file(sql_file=load_sql_file, db_path=db_path)
    if not res.ok:
        raise RuntimeError(res.error or "Failed to load raw tables")


def run_db_qna(
    *,
    question: str,
    db_path: Path,
    ensure_raw_sql_file: Path | None = None,
    stream_mode: str | None = None,
    checkpoint_db_path: Path | None = None,
    thread_id: str | None = None,
    show_planner_meta: bool = False,
) -> str:
    """Convenience runner used by CLI."""

    if ensure_raw_sql_file is not None:
        ensure_raw_loaded(db_path=db_path, load_sql_file=ensure_raw_sql_file)

    runner = DuckDBRunner(db_path=db_path)
    workflow = DbQnaWorkflow()

    checkpointer = None
    checkpoint_conn: sqlite3.Connection | None = None
    if checkpoint_db_path is not None:
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        checkpointer = SqliteSaver(checkpoint_conn)

    app = workflow.build(checkpointer=checkpointer)

    payload = {
        "question": question,
        "duckdb_runner": runner,
        "iterations": 0,
        "done": False,
    }

    invoke_config = None
    if thread_id:
        invoke_config = {"configurable": {"thread_id": thread_id}}

    final: dict[str, Any] | None = None
    invoke_error: Exception | None = None

    def _safe_preview(value: Any, *, max_len: int = 160) -> str:
        text = str(value)
        if len(text) > max_len:
            return text[:max_len].rstrip() + "..."
        return text

    def _summarize_state_update(data: dict[str, Any]) -> str:
        parts: list[str] = []
        for node_name, node_payload in data.items():
            if not isinstance(node_payload, dict):
                parts.append(f"{node_name}: <non-dict update>")
                continue

            keys = sorted(node_payload.keys())
            highlights: list[str] = []
            if "sql" in node_payload:
                highlights.append(f"sql_len={len(str(node_payload.get('sql', '')))}")
            if "queries" in node_payload:
                try:
                    highlights.append(f"queries={len(node_payload.get('queries') or [])}")
                except Exception:
                    highlights.append("queries=?")
            if "tool_error" in node_payload and node_payload.get("tool_error"):
                highlights.append(f"tool_error={_safe_preview(node_payload.get('tool_error'))}")
            if "answer" in node_payload and node_payload.get("answer"):
                highlights.append(f"answer={_safe_preview(node_payload.get('answer'))}")
            if "query_result_preview" in node_payload and node_payload.get("query_result_preview"):
                highlights.append(
                    f"query_result_preview={_safe_preview(node_payload.get('query_result_preview'))}"
                )
            if "query_result_previews" in node_payload:
                try:
                    highlights.append(f"query_result_previews={len(node_payload.get('query_result_previews') or [])}")
                except Exception:
                    highlights.append("query_result_previews=?")
            if "iterations" in node_payload:
                highlights.append(f"iterations={node_payload.get('iterations')}")
            if "done" in node_payload:
                highlights.append(f"done={node_payload.get('done')}")

            keys_text = ",".join(keys)
            detail = " | ".join(highlights) if highlights else "no-highlight-fields"
            parts.append(f"{node_name}: keys=[{keys_text}] {detail}")
        return " || ".join(parts)

    try:
        if stream_mode:
            logging.getLogger(__name__).info("QnA stream mode enabled: %s", stream_mode)
            requested_modes = ["values"] if stream_mode == "values" else [stream_mode, "values"]
            for mode, data in app.stream(payload, config=invoke_config, stream_mode=requested_modes):
                if mode == "updates" and isinstance(data, dict):
                    logging.getLogger(__name__).info("Stream update: %s", _summarize_state_update(data))
                elif mode == "values" and isinstance(data, dict):
                    final = data
                elif mode == "tasks":
                    logging.getLogger(__name__).info("Stream task: %s", data)
                elif mode == "checkpoints":
                    logging.getLogger(__name__).info("Stream checkpoint")
                elif mode == "messages":
                    if isinstance(data, tuple) and len(data) == 2:
                        token, _meta = data
                        logging.getLogger(__name__).info("Stream message token: %s", _safe_preview(token, max_len=80))
                    else:
                        logging.getLogger(__name__).info("Stream message event")
                elif mode == "custom":
                    logging.getLogger(__name__).info("Stream custom: %s", _safe_preview(data))
                elif mode == "debug":
                    logging.getLogger(__name__).debug("Stream debug: %s", data)
        else:
            final = app.invoke(payload, config=invoke_config)
    except Exception as e:
        invoke_error = e
    finally:
        if checkpoint_conn is not None:
            checkpoint_conn.close()

    if invoke_error is not None:
        message = str(invoke_error).strip()
        if len(message) > 500:
            message = message[:500].rstrip() + "..."
        logging.getLogger(__name__).error("QnA failed: %s", message)
        return f"Error: {message}"

    if final is None:
        return "Error: Workflow finished without a final result."

    logging.getLogger(__name__).info(
        "QnA completed: memory_size=%s resolved_filters=%s",
        len(final.get("memory", [])),
        final.get("resolved_filters", {}),
    )

    answer = str(final.get("answer", "")).strip()
    if answer:
        if show_planner_meta:
            rationale = str(final.get("planner_rationale_summary", "")).strip() or "<not provided>"
            confidence = final.get("planner_confidence")
            confidence_text = "<not provided>" if confidence is None else f"{float(confidence):.2f}"
            tags = final.get("planner_reason_tags") or []
            tags_text = ", ".join([str(t) for t in tags]) if tags else "<none>"
            return (
                f"{answer}\n\n"
                f"Planner rationale: {rationale}\n"
                f"Planner confidence: {confidence_text}\n"
                f"Planner tags: {tags_text}"
            )
        return answer

    # Fallback if planner died early
    if final.get("tool_error"):
        return f"Error: {final['tool_error']}"

    return "No answer could be generated."


def run_db_qna_stream(
    *,
    question: str,
    db_path: Path,
    ensure_raw_sql_file: Path | None = None,
    stream_mode: str | None = None,
    checkpoint_db_path: Path | None = None,
    thread_id: str | None = None,
    show_planner_meta: bool = False,
) -> Iterator[dict[str, str]]:
    """Run DB QnA and yield streaming events for UI.

    Event types:
    - {"type": "status", "text": "..."}
    - {"type": "final", "answer": "..."}
    - {"type": "error", "error": "..."}
    """

    if ensure_raw_sql_file is not None:
        ensure_raw_loaded(db_path=db_path, load_sql_file=ensure_raw_sql_file)

    runner = DuckDBRunner(db_path=db_path)
    workflow = DbQnaWorkflow()

    checkpointer = None
    checkpoint_conn: sqlite3.Connection | None = None
    if checkpoint_db_path is not None:
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        checkpointer = SqliteSaver(checkpoint_conn)

    app = workflow.build(checkpointer=checkpointer)

    payload = {
        "question": question,
        "duckdb_runner": runner,
        "iterations": 0,
        "done": False,
    }

    invoke_config = None
    if thread_id:
        invoke_config = {"configurable": {"thread_id": thread_id}}

    final: dict[str, Any] | None = None

    def _safe_preview(value: Any, *, max_len: int = 160) -> str:
        text = str(value)
        if len(text) > max_len:
            return text[:max_len].rstrip() + "..."
        return text

    def _summarize_state_update(data: dict[str, Any]) -> str:
        parts: list[str] = []
        for node_name, node_payload in data.items():
            if not isinstance(node_payload, dict):
                parts.append(f"{node_name}: <non-dict update>")
                continue

            keys = sorted(node_payload.keys())
            highlights: list[str] = []
            if "sql" in node_payload:
                highlights.append(f"sql_len={len(str(node_payload.get('sql', '')))}")
            if "queries" in node_payload:
                try:
                    highlights.append(f"queries={len(node_payload.get('queries') or [])}")
                except Exception:
                    highlights.append("queries=?")
            if "tool_error" in node_payload and node_payload.get("tool_error"):
                highlights.append(f"tool_error={_safe_preview(node_payload.get('tool_error'))}")
            if "answer" in node_payload and node_payload.get("answer"):
                highlights.append(f"answer={_safe_preview(node_payload.get('answer'))}")
            if "iterations" in node_payload:
                highlights.append(f"iterations={node_payload.get('iterations')}")
            if "done" in node_payload:
                highlights.append(f"done={node_payload.get('done')}")

            keys_text = ",".join(keys)
            detail = " | ".join(highlights) if highlights else "no-highlight-fields"
            parts.append(f"{node_name}: keys=[{keys_text}] {detail}")
        return " || ".join(parts)

    try:
        requested_modes = ["updates", "values"]
        if stream_mode and stream_mode not in requested_modes:
            requested_modes.append(stream_mode)

        logging.getLogger(__name__).info("QnA stream mode enabled: %s", stream_mode or "updates+values")

        for mode, data in app.stream(payload, config=invoke_config, stream_mode=requested_modes):
            if mode == "updates" and isinstance(data, dict):
                summary = _summarize_state_update(data)
                logging.getLogger(__name__).info("Stream update: %s", summary)
                yield {"type": "status", "text": summary}
            elif mode == "values" and isinstance(data, dict):
                final = data
            elif mode == "tasks":
                logging.getLogger(__name__).info("Stream task: %s", data)
                yield {"type": "status", "text": f"task: {_safe_preview(data)}"}
            elif mode == "checkpoints":
                logging.getLogger(__name__).info("Stream checkpoint")
                yield {"type": "status", "text": "checkpoint saved"}
            elif mode == "messages":
                if isinstance(data, tuple) and len(data) == 2:
                    token, _meta = data
                    yield {"type": "status", "text": f"llm token: {_safe_preview(token, max_len=80)}"}
            elif mode == "custom":
                yield {"type": "status", "text": f"custom: {_safe_preview(data)}"}
            elif mode == "debug":
                logging.getLogger(__name__).debug("Stream debug: %s", data)

    except Exception as e:
        message = str(e).strip()
        if len(message) > 500:
            message = message[:500].rstrip() + "..."
        logging.getLogger(__name__).error("QnA failed: %s", message)
        yield {"type": "error", "error": message}
        return
    finally:
        if checkpoint_conn is not None:
            checkpoint_conn.close()

    if final is None:
        yield {"type": "error", "error": "Workflow finished without a final result."}
        return

    answer = str(final.get("answer", "")).strip()
    if answer and show_planner_meta:
        rationale = str(final.get("planner_rationale_summary", "")).strip() or "<not provided>"
        confidence = final.get("planner_confidence")
        confidence_text = "<not provided>" if confidence is None else f"{float(confidence):.2f}"
        tags = final.get("planner_reason_tags") or []
        tags_text = ", ".join([str(t) for t in tags]) if tags else "<none>"
        answer = (
            f"{answer}\n\n"
            f"Planner rationale: {rationale}\n"
            f"Planner confidence: {confidence_text}\n"
            f"Planner tags: {tags_text}"
        )

    if answer:
        yield {"type": "final", "answer": answer}
        return

    if final.get("tool_error"):
        yield {"type": "error", "error": str(final.get("tool_error"))}
        return

    yield {"type": "error", "error": "No answer could be generated."}
