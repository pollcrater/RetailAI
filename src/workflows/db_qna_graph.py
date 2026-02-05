from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph
import logging

from src.agents.planner import SqlPlannerAgent
from src.agents.executor import SqlExecutorAgent
from src.agents.validator import ResultValidatorAgent
from src.tools.duckdb_tools import DuckDBRunner, load_raw_from_sql_file


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
    planner_feedback: str

    # executor outputs
    query_result_df: Any
    query_result_dfs: list[Any]
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

    def build(self):
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
                }
            logger.info("Executor ok")
            if "dfs" in result.content:
                return {**state, "tool_error": "", "query_result_dfs": result.content["dfs"]}
            return {**state, "tool_error": "", "query_result_df": result.content["df"]}

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

        return graph.compile()


def ensure_raw_loaded(*, db_path: Path, load_sql_file: Path) -> None:
    """Idempotent-ish raw load: execute load SQL to create raw tables."""

    res = load_raw_from_sql_file(sql_file=load_sql_file, db_path=db_path)
    if not res.ok:
        raise RuntimeError(res.error or "Failed to load raw tables")


def _load_memory(memory_file: Path) -> dict[str, Any]:
    try:
        if not memory_file.exists():
            return {"memory": [], "resolved_filters": {}}
        data = json.loads(memory_file.read_text(encoding="utf-8"))
        memory = data.get("memory", []) if isinstance(data, dict) else []
        resolved_filters = data.get("resolved_filters", {}) if isinstance(data, dict) else {}
        if not isinstance(memory, list):
            memory = []
        if not isinstance(resolved_filters, dict):
            resolved_filters = {}
        return {"memory": memory, "resolved_filters": resolved_filters}
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to load memory: %s", e)
        return {"memory": [], "resolved_filters": {}}


def _save_memory(memory_file: Path, *, memory: list[dict[str, Any]], resolved_filters: dict[str, Any]) -> None:
    try:
        memory_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {"memory": memory, "resolved_filters": resolved_filters}
        memory_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save memory: %s", e)


def run_db_qna(
    *,
    question: str,
    db_path: Path,
    ensure_raw_sql_file: Path | None = None,
    memory_file: Path | None = None,
    stream_mode: str | None = None,
) -> str:
    """Convenience runner used by CLI."""

    if ensure_raw_sql_file is not None:
        ensure_raw_loaded(db_path=db_path, load_sql_file=ensure_raw_sql_file)

    runner = DuckDBRunner(db_path=db_path)
    workflow = DbQnaWorkflow()
    app = workflow.build()

    memory_state = {"memory": [], "resolved_filters": {}}
    if memory_file is not None:
        memory_state = _load_memory(memory_file)

    payload = {
        "question": question,
        "duckdb_runner": runner,
        "iterations": 0,
        "memory": memory_state.get("memory", []),
        "resolved_filters": memory_state.get("resolved_filters", {}),
        "memory_size": workflow.memory_size,
    }

    if stream_mode:
        logging.getLogger(__name__).info("QnA stream mode enabled: %s", stream_mode)
        for mode, data in app.stream(payload, stream_mode=[stream_mode]):
            if mode == "updates" and isinstance(data, dict):
                logging.getLogger(__name__).info("Stream update: %s", list(data.keys()))
            elif mode == "tasks":
                logging.getLogger(__name__).info("Stream task: %s", data)
            elif mode == "checkpoints":
                logging.getLogger(__name__).info("Stream checkpoint")
            elif mode == "debug":
                logging.getLogger(__name__).debug("Stream debug: %s", data)
        final = app.invoke(payload)
    else:
        final = app.invoke(payload)

    if memory_file is not None:
        _save_memory(
            memory_file,
            memory=final.get("memory", []),
            resolved_filters=final.get("resolved_filters", {}),
        )

    logging.getLogger(__name__).info(
        "QnA completed: memory_size=%s resolved_filters=%s",
        len(final.get("memory", [])),
        final.get("resolved_filters", {}),
    )

    answer = str(final.get("answer", "")).strip()
    if answer:
        return answer

    # Fallback if planner died early
    if final.get("tool_error"):
        return f"Error: {final['tool_error']}"

    return "No answer produced."
