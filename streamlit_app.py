from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import streamlit as st

from src.config import get_settings
from src.ui.session_state import (
    append_message,
    create_chat,
    ensure_session_state,
    get_active_chat,
    set_active_chat,
    update_active_chat,
)
from src.utils.logging import configure_logging
from src.workflows.data_pipeline import run_processing_pipeline
from src.workflows.db_qna_graph import run_db_qna_stream
from src.workflows.db_summary import run_db_summary


st.set_page_config(page_title="Retail Insights Assistant", page_icon="📊", layout="wide")
logger = logging.getLogger(__name__)


def _ensure_logging() -> None:
    if st.session_state.get("logging_ready"):
        return
    settings = get_settings()
    configure_logging(settings)
    st.session_state.logging_ready = True
    logger.info("Streamlit logging initialized: level=%s", settings.log_level)


def _save_uploaded_csv(file_bytes: bytes, file_name: str) -> Path:
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file_name).name
    target = upload_dir / safe_name
    target.write_bytes(file_bytes)
    return target


def _render_chat_list() -> None:
    st.sidebar.header("Chats")
    if st.sidebar.button("+ New Chat", use_container_width=True):
        create_chat()
        st.rerun()

    for thread_id, chat in st.session_state.chat_sessions.items():
        label = f"{chat.title} ({chat.mode.upper()})"
        if st.sidebar.button(label, key=f"chat_{thread_id}", use_container_width=True):
            set_active_chat(thread_id)
            st.rerun()


def _render_monitor_controls() -> None:
    st.sidebar.subheader("State Monitor")
    selected = st.sidebar.selectbox(
        "LangGraph stream mode",
        options=["off", "updates", "values", "checkpoints", "tasks", "messages", "debug", "custom"],
        index=0,
        help="Choose a stream mode to print LangGraph state/event updates in the terminal.",
    )
    st.session_state.qna_stream_mode = None if selected == "off" else selected


def _render_status() -> None:
    state = st.session_state.app_state
    if state == "ingesting":
        st.info("Ingesting uploaded file... Please wait.")
    elif state == "running_query":
        st.info("Processing your request... Please wait.")
    elif state == "ready":
        st.success("Ready. You can summarize the report or ask questions.")
    else:
        st.caption("Upload a CSV file to begin.")


def _run_ingestion(uploaded_name: str, uploaded_bytes: bytes) -> None:
    st.session_state.app_state = "ingesting"
    logger.info("Ingestion start: file=%s", uploaded_name)
    with st.spinner("Running ingestion pipeline..."):
        csv_path = _save_uploaded_csv(uploaded_bytes, uploaded_name)
        result = run_processing_pipeline(
            csv_path=csv_path,
            db_path=Path(st.session_state.db_path),
            table_name="mart.sales_clean",
        )

    st.session_state.last_ingestion = {
        "processed_csv": str(result.processed_csv),
        "summary_json": str(result.summary_json),
        "table_name": result.table_name,
        "row_count": result.row_count,
    }
    update_active_chat(table_name=result.table_name)
    append_message(
        role="assistant",
        content=(
            "Upload complete. Data ingestion succeeded.\n"
            f"- Table: {result.table_name}\n"
            f"- Rows: {result.row_count}\n"
            "Choose next step: Summarize Report or Q&A."
        ),
    )
    st.session_state.app_state = "ready"
    logger.info("Ingestion complete: table=%s rows=%s", result.table_name, result.row_count)


def _handle_uploaded_file() -> None:
    st.subheader("1) Upload sales data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is None:
        return

    file_bytes = uploaded.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.uploaded_file_hash == digest:
        st.caption(f"Using uploaded file: {uploaded.name}")
        return

    st.session_state.uploaded_file_hash = digest
    st.session_state.uploaded_file_name = uploaded.name

    try:
        _run_ingestion(uploaded.name, file_bytes)
        st.rerun()
    except Exception as e:
        st.session_state.app_state = "idle"
        logger.exception("Ingestion failed")
        st.error(f"Ingestion failed: {e}")


def _run_summary() -> None:
    st.session_state.app_state = "running_query"
    logger.info("Summary start: table=mart.sales_clean")
    with st.spinner("Generating summary..."):
        result = run_db_summary(
            db_path=Path(st.session_state.db_path),
            table="mart.sales_clean",
        )
    append_message(role="assistant", content=result.summary_text)
    st.session_state.app_state = "ready"
    logger.info("Summary complete")


def _run_qna(question: str) -> None:
    st.session_state.app_state = "running_query"
    append_message(role="user", content=question)
    logger.info("QnA start: thread_id=%s question=%s", get_active_chat().thread_id, question)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()

        final_answer = ""
        with st.spinner("Getting answer..."):
            for event in run_db_qna_stream(
                question=question,
                db_path=Path(st.session_state.db_path),
                ensure_raw_sql_file=None,
                stream_mode=st.session_state.get("qna_stream_mode"),
                checkpoint_db_path=Path(st.session_state.checkpoint_db_path),
                thread_id=get_active_chat().thread_id,
                show_planner_meta=False,
            ):
                event_type = event.get("type")
                if event_type == "status":
                    status_placeholder.caption(f"⏳ {event.get('text', '')}")
                elif event_type == "error":
                    final_answer = f"Error: {event.get('error', 'Unknown error')}"
                    status_placeholder.empty()
                    answer_placeholder.markdown(final_answer)
                    break
                elif event_type == "final":
                    final_answer = event.get("answer", "")
                    status_placeholder.empty()
                    streamed = ""
                    for token in final_answer.split(" "):
                        streamed = (streamed + " " + token).strip()
                        answer_placeholder.markdown(streamed)
                        time.sleep(0.01)
                    break

        if not final_answer:
            final_answer = "Error: No answer could be generated."
            status_placeholder.empty()
            answer_placeholder.markdown(final_answer)

    append_message(role="assistant", content=final_answer)
    st.session_state.app_state = "ready"
    logger.info("QnA complete: thread_id=%s", get_active_chat().thread_id)


def _render_mode_controls() -> None:
    st.subheader("2) Choose analysis mode")

    col1, col2 = st.columns(2)
    running = st.session_state.app_state in {"ingesting", "running_query"}

    with col1:
        if st.button("Summarize Report", disabled=running, use_container_width=True):
            update_active_chat(mode="summary")
            try:
                _run_summary()
            except Exception as e:
                st.session_state.app_state = "ready"
                logger.exception("Summary failed")
                st.error(f"Summary failed: {e}")
            st.rerun()

    with col2:
        if st.button("Q&A", disabled=running, use_container_width=True):
            update_active_chat(mode="qna")
            append_message(role="assistant", content="Q&A mode is active. Ask your question below.")
            st.rerun()


def _render_chat_window() -> None:
    st.subheader("3) Chat")
    chat = get_active_chat()

    for msg in chat.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    running = st.session_state.app_state in {"ingesting", "running_query"}
    qna_enabled = st.session_state.app_state == "ready" and chat.mode == "qna"

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Message",
            placeholder="Ask a business question about the uploaded sales data...",
            disabled=not qna_enabled,
            label_visibility="collapsed",
        )
        send = st.form_submit_button("Send", disabled=not qna_enabled or running)

    if send and question.strip():
        try:
            _run_qna(question.strip())
        except Exception as e:
            st.session_state.app_state = "ready"
            logger.exception("QnA failed")
            append_message(role="assistant", content=f"Error: {e}")
        st.rerun()


def main() -> None:
    ensure_session_state()
    _ensure_logging()

    st.title("Retail Insights Assistant")
    st.caption("Upload sales data, run ingestion, then use Summarization or Q&A in a chat interface.")

    _render_chat_list()
    _render_monitor_controls()
    _render_status()
    _handle_uploaded_file()

    if st.session_state.last_ingestion:
        _render_mode_controls()

    _render_chat_window()


if __name__ == "__main__":
    main()
