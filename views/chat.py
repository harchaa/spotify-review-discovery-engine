from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.chat import answer_chat_question  # noqa: E402
from analysis.llm_client import LLMClient  # noqa: E402
from ui_common import load_reviews, load_summary_tables  # noqa: E402

STARTER_QUESTIONS = [
    "Why do users struggle to discover new music?",
    "What frustrates people most about recommendations?",
    "What are users trying to achieve?",
    "Why do people keep replaying the same songs?",
    "Which user segments struggle the most?",
    "What's the biggest unmet need?",
]


@st.cache_resource
def get_client() -> LLMClient:
    return LLMClient()


st.title("Chat")
st.caption("Ask about what real Spotify users are saying — answers are grounded in the analyzed reviews, not invented.")

reviews_df = load_reviews()
tables = load_summary_tables()

if reviews_df is None or reviews_df.empty:
    st.info("No review data yet. Run `python run_pipeline.py` first to scrape and process real Spotify feedback.")
    st.stop()

if tables is None or not tables.get("total_relevant"):
    st.warning(
        "LLM tagging hasn't run yet, so there's nothing to chat about. Add an API key to `.env` "
        "and run `python run_pipeline.py`, then come back here."
    )
    st.stop()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if not st.session_state.chat_history:
    st.write("Try one of these, or ask your own:")
    cols = st.columns(3)
    for i, question in enumerate(STARTER_QUESTIONS):
        if cols[i % 3].button(question, width="stretch"):
            st.session_state.pending_question = question

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"], avatar="🎧" if turn["role"] == "assistant" else None):
        st.write(turn["content"])

user_question = st.chat_input("Ask a question about the reviews...")
if not user_question and "pending_question" in st.session_state:
    user_question = st.session_state.pop("pending_question")

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant", avatar="🎧"):
        with st.spinner("Reading through the reviews..."):
            try:
                answer = answer_chat_question(get_client(), user_question, tables, st.session_state.chat_history[:-1])
            except Exception as exc:  # unexpected, non-quota failure - surface plainly, don't crash the page
                answer = f"Something went wrong answering that: {exc}"
        st.write(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})

if st.session_state.chat_history and st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()
