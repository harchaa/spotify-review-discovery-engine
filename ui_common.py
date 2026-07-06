from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


DATA_DIR = Path(__file__).resolve().parent / "data"
REVIEWS_PATH = DATA_DIR / "reviews.csv"
SUMMARY_PATH = DATA_DIR / "summary_tables.json"
ANSWERS_PATH = DATA_DIR / "six_question_answers.json"

THEME_LABELS = {
    "stale_recommendations": "Stale recommendations",
    "no_control_over_algorithm": "No control over algorithm",
    "discovery_buried_in_ui": "Discovery buried in UI",
    "filter_bubble_overpersonalization": "Filter bubble / overpersonalization",
    "discovery_breaks_focus": "Discovery breaks focus",
    "repetition_fatigue": "Repetition fatigue",
    "poor_new_release_surfacing": "Poor new-release surfacing",
    "trust_in_recs": "Trust in recommendations",
    "other": "Other",
}
SEGMENT_LABELS = {
    "focus_work_listener": "Focus / work listener",
    "discovery_seeker_distrusts_algo": "Discovery seeker (distrusts algorithm)",
    "workout_tempo_listener": "Workout / tempo listener",
    "mood_context_listener": "Mood / context listener",
    "genre_explorer_filter_bubble": "Genre explorer (filter bubble)",
    "other": "Other",
}


def is_dark_theme() -> bool:
    return st.get_option("theme.base") == "dark"


def bar_color(dark: bool) -> str:
    return "#3987e5" if dark else "#2a78d6"


def accent_color(dark: bool) -> str:
    return "#199e70" if dark else "#1baf7a"


def ink_color(dark: bool) -> str:
    return "#c3c2b7" if dark else "#52514e"


def gridline_color(dark: bool) -> str:
    return "#2c2c2a" if dark else "#e1e0d9"


@st.cache_data
def load_json(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_csv(path_str: str):
    path = Path(path_str)
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_reviews() -> pd.DataFrame | None:
    return load_csv(str(REVIEWS_PATH))


def load_summary_tables() -> dict | None:
    return load_json(str(SUMMARY_PATH))


def load_six_question_answers() -> dict | None:
    return load_json(str(ANSWERS_PATH))


def ranked_bar_chart(records: list[dict], label_col: str, value_col: str, labels: dict, color: str, dark: bool, x_title: str):
    df = pd.DataFrame(records)
    df["_label"] = df[label_col].map(lambda k: labels.get(k, k))
    df = df.sort_values(value_col, ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df[value_col],
            y=df["_label"],
            orientation="h",
            marker=dict(color=color),
            text=df[value_col],
            textposition="outside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ink_color(dark)),
        xaxis=dict(title=x_title, gridcolor=gridline_color(dark), zeroline=False),
        yaxis=dict(title=None),
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(240, 42 * len(df)),
        showlegend=False,
    )
    return fig


def render_quote(quote: dict) -> None:
    text = quote.get("text", "")
    if len(text) > 320:
        text = text[:320] + "..."
    st.markdown(f"> {text}")
    source = quote.get("source", "unknown")
    url = quote.get("url")
    if url and isinstance(url, str) and url.lower() != "nan":
        st.caption(f"— {source} · [source]({url})")
    else:
        st.caption(f"— {source}")


def quotes_for_themes(quotes_by_theme: dict, themes: list[str] | None, limit: int = 3) -> list[dict]:
    if not quotes_by_theme:
        return []
    pools = [quotes_by_theme[t] for t in (themes or quotes_by_theme.keys()) if t in quotes_by_theme]
    interleaved: list[dict] = []
    for i in range(limit):
        for pool in pools:
            if i < len(pool):
                interleaved.append(pool[i])
            if len(interleaved) >= limit:
                return interleaved
    return interleaved[:limit]


def quotes_for_segments(quotes_by_segment: dict, limit: int = 3) -> list[dict]:
    if not quotes_by_segment:
        return []
    picked = []
    for segment_quotes in quotes_by_segment.values():
        if segment_quotes:
            picked.append(segment_quotes[0])
        if len(picked) >= limit:
            break
    return picked
