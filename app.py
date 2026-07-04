from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis.cleaning import parse_date
from config import RECENCY_MONTHS

DATA_DIR = Path(__file__).resolve().parent / "data"
REVIEWS_PATH = DATA_DIR / "reviews.csv"
SUMMARY_PATH = DATA_DIR / "summary_tables.json"
ANSWERS_PATH = DATA_DIR / "six_question_answers.json"

SIX_QUESTIONS = [
    ("why_struggle_to_discover", "Why do users struggle to discover new music?", ["discovery_buried_in_ui", "poor_new_release_surfacing", "filter_bubble_overpersonalization"]),
    ("recommendation_frustrations", "What are the most common frustrations with recommendations?", ["stale_recommendations", "no_control_over_algorithm", "trust_in_recs"]),
    ("desired_listening_behaviors", "What listening behaviors are users trying to achieve?", None),
    ("repetition_causes", "What causes users to repeatedly listen to the same content?", ["repetition_fatigue", "discovery_breaks_focus"]),
    ("segment_challenges", "Which user segments experience different discovery challenges?", None),
    ("unmet_needs", "What unmet needs emerge consistently across reviews?", None),
]

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
SENTIMENT_COLORS = {"positive": "#0ca30c", "neutral": "#898781", "negative": "#d03b3b"}


def is_dark_theme() -> bool:
    return st.get_option("theme.base") == "dark"


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


def overview_date_range(reviews_df: pd.DataFrame) -> str:
    # Row-wise parsing, not pd.to_datetime(..., utc=True): the raw "date" column mixes
    # naive and tz-aware ISO strings across sources, and pandas' vectorized parser
    # silently coerces every format but the first-seen one to NaT.
    dates = reviews_df["date"].apply(parse_date).dropna()
    if dates.empty:
        return "n/a"
    return f"{dates.min()} — {dates.max()}"


def main() -> None:
    st.set_page_config(page_title="Spotify Review Discovery Engine", layout="wide")
    dark_theme = is_dark_theme()
    bar_color = "#3987e5" if dark_theme else "#2a78d6"
    accent_color = "#199e70" if dark_theme else "#1baf7a"

    st.title("Spotify Review Discovery Engine")
    st.caption("AI-powered analysis of real Spotify user feedback about music discovery and recommendations.")

    reviews_df = load_csv(str(REVIEWS_PATH))

    if reviews_df is None or reviews_df.empty:
        st.info("No review data yet. Run `python run_pipeline.py` first to scrape and process real Spotify feedback.")
        st.stop()

    # ---------- Overview ----------
    st.header("Overview")
    cols = st.columns(4)
    cols[0].metric("Items analyzed", f"{len(reviews_df):,}")
    source_counts = reviews_df["source"].value_counts()
    cols[1].metric("Sources", len(source_counts))
    cols[2].metric("Date range", overview_date_range(reviews_df))
    cols[3].metric("Recency window", f"{RECENCY_MONTHS} months")

    st.caption("Source mix: " + " · ".join(f"{src}: {n:,}" for src, n in source_counts.items()))

    with st.expander("Limitations & scope", expanded=False):
        st.markdown(
            "- **Reddit is excluded**: its Data API now requires a moderation-use-case approval "
            "that doesn't fit this project's timeline.\n"
            "- **App Store coverage is capped** by the public customer-reviews RSS feed "
            "(roughly 500 reviews/country); Apple's documented `sortby=mostrecent` parameter "
            "currently returns zero entries, so the scraper uses the feed's default ordering instead.\n"
            "- **Non-English reviews are kept**, not dropped, and language-tagged; the LLM reasons "
            "over the original text.\n"
            f"- **Recency filter**: Google Play / App Store reviews older than {RECENCY_MONTHS} months "
            "are excluded so the analysis reflects current app behavior, not complaints about removed "
            "features. Spotify Community ideas are kept if recent, OR if they're high-vote (top quartile) "
            "and not older than 2023 — otherwise long-implemented ideas from 2013-2019 would dominate "
            "the 'unmet needs' view even though Spotify already shipped them."
        )

    summary = load_json(str(SUMMARY_PATH))
    answers = load_json(str(ANSWERS_PATH))

    if summary is None or answers is None:
        st.warning(
            "LLM tagging and aggregation haven't been run yet (no `GEMINI_API_KEY` was set during the "
            "last pipeline run). Add the key to `.env` and re-run `python run_pipeline.py` to see themes, "
            "segments, and the six answers below."
        )
        st.stop()

    themes = summary.get("themes", [])
    segments = summary.get("segments", [])
    segment_details = summary.get("segment_details", {})
    unmet_needs = summary.get("unmet_needs", [])
    quotes_by_theme = summary.get("quotes_by_theme", {})
    quotes_by_segment = summary.get("quotes_by_segment", {})

    st.caption(f"{summary.get('total_relevant', 0):,} of {summary.get('total_rows', 0):,} items were tagged as relevant to discovery, recommendations, or repetition.")

    # ---------- Six questions ----------
    st.header("The six questions")
    for key, question, related_themes in SIX_QUESTIONS:
        with st.container(border=True):
            st.subheader(question)
            st.write(answers.get(key, "_No answer generated._"))

            if key == "segment_challenges" and segments:
                st.caption("Segment sizes: " + " · ".join(f"{SEGMENT_LABELS.get(s['use_case_segment'], s['use_case_segment'])}: {s['count']}" for s in segments[:5]))
                for quote in quotes_for_segments(quotes_by_segment):
                    render_quote(quote)
            elif key == "unmet_needs" and unmet_needs:
                st.caption("Top unmet needs (weighted by frequency, severity, and community votes):")
                for need in unmet_needs[:3]:
                    st.markdown(f"- **{need['job_to_be_done']}** ({THEME_LABELS.get(need['theme'], need['theme'])}, {need['count']} mentions)")
            else:
                if themes:
                    st.caption("Supporting counts: " + " · ".join(f"{THEME_LABELS.get(t['theme'], t['theme'])}: {t['count']}" for t in themes[:5]))
                for quote in quotes_for_themes(quotes_by_theme, related_themes):
                    render_quote(quote)

    # ---------- Themes ----------
    st.header("Themes")
    if themes:
        st.plotly_chart(
            ranked_bar_chart(themes, "theme", "weighted_count", THEME_LABELS, bar_color, dark_theme, "Weighted mentions (community items weighted by votes)"),
            width="stretch",
        )
        theme_choice = st.selectbox("Drill down into a theme", options=[t["theme"] for t in themes], format_func=lambda t: THEME_LABELS.get(t, t))
        for quote in quotes_by_theme.get(theme_choice, [])[:3]:
            render_quote(quote)
    else:
        st.info("No themes tagged yet.")

    # ---------- Segments ----------
    st.header("Segments")
    if segments:
        st.plotly_chart(
            ranked_bar_chart(segments, "use_case_segment", "count", SEGMENT_LABELS, accent_color, dark_theme, "Items"),
            width="stretch",
        )
        for segment in segments:
            seg_key = segment["use_case_segment"]
            details = segment_details.get(seg_key, {})
            with st.expander(f"{SEGMENT_LABELS.get(seg_key, seg_key)} — {segment['count']} items"):
                top_themes = details.get("top_themes", [])
                if top_themes:
                    st.markdown("**Top frustrations:** " + ", ".join(THEME_LABELS.get(t["theme"], t["theme"]) for t in top_themes))
                top_jobs = details.get("top_jobs", [])
                if top_jobs:
                    st.markdown("**Jobs-to-be-done:** " + "; ".join(top_jobs))
                for quote in quotes_by_segment.get(seg_key, [])[:2]:
                    render_quote(quote)
    else:
        st.info("No segments tagged yet.")

    # ---------- Unmet needs ----------
    st.header("Unmet needs")
    if unmet_needs:
        st.caption("Ranked by frequency × severity, with community items additionally weighted by kudos/votes.")
        unmet_df = pd.DataFrame(unmet_needs)
        unmet_df["theme"] = unmet_df["theme"].map(lambda t: THEME_LABELS.get(t, t))
        st.dataframe(
            unmet_df.rename(columns={"job_to_be_done": "Unmet need", "theme": "Theme", "count": "Mentions", "weighted_score": "Weighted score"}),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No unmet needs identified yet.")


if __name__ == "__main__":
    main()
