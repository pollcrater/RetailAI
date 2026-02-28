from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

import streamlit as st


@dataclass
class ChatSession:
    thread_id: str
    title: str
    created_at: str
    mode: str = "qna"
    messages: list[dict[str, str]] = field(default_factory=list)
    table_name: str | None = None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def new_chat_session(title: str = "New Chat") -> ChatSession:
    return ChatSession(
        thread_id=str(uuid4()),
        title=title,
        created_at=_now_iso(),
        messages=[],
    )


def ensure_session_state() -> None:
    if "chat_sessions" not in st.session_state:
        first = new_chat_session("Chat 1")
        st.session_state.chat_sessions = {first.thread_id: first}
        st.session_state.active_thread_id = first.thread_id

    st.session_state.setdefault("app_state", "idle")
    st.session_state.setdefault("db_path", "data/retail_sales.duckdb")
    st.session_state.setdefault("checkpoint_db_path", "data/checkpoints.db")
    st.session_state.setdefault("uploaded_file_hash", None)
    st.session_state.setdefault("uploaded_file_name", None)
    st.session_state.setdefault("last_ingestion", None)


def get_active_chat() -> ChatSession:
    active_id = st.session_state.active_thread_id
    return st.session_state.chat_sessions[active_id]


def set_active_chat(thread_id: str) -> None:
    if thread_id in st.session_state.chat_sessions:
        st.session_state.active_thread_id = thread_id


def create_chat(title: str | None = None) -> None:
    chat_number = len(st.session_state.chat_sessions) + 1
    session = new_chat_session(title or f"Chat {chat_number}")
    st.session_state.chat_sessions[session.thread_id] = session
    st.session_state.active_thread_id = session.thread_id


def append_message(*, role: str, content: str) -> None:
    chat = get_active_chat()
    chat.messages.append({"role": role, "content": content})


def update_active_chat(**kwargs: Any) -> None:
    chat = get_active_chat()
    for key, value in kwargs.items():
        if hasattr(chat, key):
            setattr(chat, key, value)
