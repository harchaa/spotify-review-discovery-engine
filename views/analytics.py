from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RECENCY_MONTHS  # noqa: E402
from ui_common import (  # noqa: E402
    SEGMENT_LABELS,
    THEME_LABELS,
    accent_color,
    bar_color,
    is_dark_theme,
    load_reviews,
    load_summary_tables,
    overview_date_range,
    ranked_bar_chart,
    render_quote,
)

dark_theme = is_dark_theme()

st.title("Analytics")
st.caption("Aggregated insights from real Spotify user feedback — Google Play, App Store, and Spotify Community.")

reviews_df = load_reviews()
if reviews_df is None or reviews_df.empty:
    st.info("No review data yet. Run `python run_pipeline.py` first to scrape and process real Spotify feedback.")
    st.stop()

with st.container(border=True):
    cols = st.columns(4)
    cols[0].metric("Items analyzed", f"{len(reviews_df):,}")
    source_counts = reviews_df["source"].value_counts()
    cols[1].metric("Sources", len(source_counts))
    cols[2].metric("Date range", overview_date_range(reviews_df))
    cols[3].metric("Recency window", f"{RECENCY_MONTHS} months")
    st.caption("Source mix: " + " · ".join(f"{src}: {n:,}" for src, n in source_counts.items()))

summary = load_summary_tables()
if summary is None:
    st.warning(
        "LLM tagging and aggregation haven't run yet. Add an API key to `.env` and run "
        "`python run_pipeline.py` to see themes, segments, and unmet needs here."
    )
    st.stop()

themes = summary.get("themes", [])
segments = summary.get("segments", [])
segment_details = summary.get("segment_details", {})
unmet_needs = summary.get("unmet_needs", [])
quotes_by_theme = summary.get("quotes_by_theme", {})
quotes_by_segment = summary.get("quotes_by_segment", {})

total_rows = summary.get("total_rows", 0) or 1
total_relevant = summary.get("total_relevant", 0)
st.caption(f"{total_relevant:,} of {total_rows:,} items ({round(100 * total_relevant / total_rows)}%) were relevant to discovery, recommendations, or repetition.")

st.info("💬 Want a specific answer, not just charts? Ask it on the **Chat** page — it's grounded in this same data.", icon="💬")

tab_themes, tab_segments, tab_needs = st.tabs(["Themes", "Segments", "Unmet needs"])

with tab_themes:
    if themes:
        st.plotly_chart(
            ranked_bar_chart(themes, "theme", "weighted_count", THEME_LABELS, bar_color(dark_theme), dark_theme, "Weighted mentions (community items weighted by votes)"),
            width="stretch",
        )
        theme_choice = st.selectbox("Drill down into a theme", options=[t["theme"] for t in themes], format_func=lambda t: THEME_LABELS.get(t, t))
        for quote in quotes_by_theme.get(theme_choice, [])[:3]:
            render_quote(quote)
    else:
        st.info("No themes tagged yet.")

with tab_segments:
    if segments:
        st.plotly_chart(
            ranked_bar_chart(segments, "use_case_segment", "count", SEGMENT_LABELS, accent_color(dark_theme), dark_theme, "Items"),
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

with tab_needs:
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

st.divider()
with st.expander("Methodology & limitations"):
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
