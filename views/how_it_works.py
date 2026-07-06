from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui_common import (  # noqa: E402
    accent_color,
    bar_color,
    gridline_color,
    ink_color,
    is_dark_theme,
    load_json,
)

DATA_DIR = ROOT / "data"

dark_theme = is_dark_theme()
scrape_c = bar_color(dark_theme)
llm_c = accent_color(dark_theme)
ink = ink_color(dark_theme)
grid = gridline_color(dark_theme)
card_bg = "rgba(255,255,255,0.035)" if dark_theme else "rgba(0,0,0,0.02)"
muted_bg = "rgba(255,255,255,0.02)" if dark_theme else "rgba(0,0,0,0.015)"

st.title("How it works")
st.caption("From live scrape to the numbers on the Analytics page — every step, and how the weekly refresh triggers itself.")

# ---------------------------------------------------------------------------
# Live snapshot of the most recent pipeline run, so the diagram below isn't
# just an abstract claim — it's grounded in the same data/ files the app reads.
# ---------------------------------------------------------------------------
pipeline_summary = load_json(str(DATA_DIR / "pipeline_summary.json"))

if pipeline_summary:
    merge = pipeline_summary.get("merge", {})
    recency = pipeline_summary.get("recency", {})
    skipped_llm = pipeline_summary.get("skipped_llm")
    rows_per_source = merge.get("rows_per_source", {})
    rows_lost = recency.get("rows_lost_per_source", {})

    # st.metric ellipsizes a value that doesn't fit its default 2.25rem font in a
    # narrow column (e.g. "2024-07-14" in a quarter-width column below ~1400px) -
    # shrink and un-truncate just the date value rather than widen the whole card.
    st.markdown(
        """
        <style>
        .st-key-cutoff-date-metric [data-testid="stMetricValue"],
        .st-key-cutoff-date-metric [data-testid="stMetricValue"] * {
            font-size: 1.5rem !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("**Last pipeline run**")
        cols = st.columns(4)
        cols[0].metric("Rows scraped", f"{merge.get('total_rows', 0):,}")
        cols[1].metric("Rows kept", f"{recency.get('rows_kept', 0):,}")
        with cols[2]:
            with st.container(key="cutoff-date-metric"):
                st.metric("Cutoff date", str(recency.get("cutoff_date", "n/a")))
        cols[3].metric("LLM tagging", "Skipped" if skipped_llm else "Ran")
        source_bits = " · ".join(f"{src}: {n:,}" for src, n in rows_per_source.items())
        st.caption(f"Rows scraped per source: {source_bits} · \"Rows kept\" is after the recency filter below")
        zero_sources = [src for src, n in rows_per_source.items() if n == 0]
        for src in zero_sources:
            st.caption(f"⚠️ {src} returned 0 rows this run — see the App Store card below for why this happens on CI runners.")
else:
    st.info("No `data/pipeline_summary.json` yet — run `python run_pipeline.py` once to populate this snapshot.")

st.divider()

# ---------------------------------------------------------------------------
# Shared HTML/CSS for the flow diagram
# ---------------------------------------------------------------------------
STYLE = f"""
<style>
.flow-wrap {{ font-size: 0.92rem; color: {ink}; }}
.flow-row {{ display: flex; gap: 14px; flex-wrap: wrap; justify-content: center; margin: 0 auto; max-width: 1000px; }}
.flow-card {{
  background: {card_bg};
  border: 1px solid {grid};
  border-left: 3px solid var(--accent, {scrape_c});
  border-radius: 10px;
  padding: 12px 16px;
  flex: 1 1 260px;
  min-width: 220px;
  max-width: 460px;
}}
.flow-card.branch {{
  border-left: 3px dashed {ink};
  background: {muted_bg};
  opacity: 0.85;
}}
.flow-card h4 {{ margin: 0 0 6px 0; font-size: 0.98rem; color: inherit; }}
.flow-card p {{ margin: 2px 0; font-size: 0.82rem; opacity: 0.85; line-height: 1.35; }}
.flow-card .file {{
  display: inline-block; margin-top: 6px; font-family: ui-monospace, monospace;
  font-size: 0.72rem; padding: 2px 6px; border-radius: 5px;
  background: {muted_bg}; border: 1px solid {grid};
}}
.flow-arrow {{ text-align: center; font-size: 1.3rem; opacity: 0.45; margin: 2px 0; line-height: 1; }}
.flow-label {{ text-align: center; font-size: 0.76rem; opacity: 0.65; max-width: 640px; margin: -4px auto 4px auto; }}
/* The data lineage table below is too wide for a narrow window - scroll it
   horizontally within itself instead of letting it overflow the page. */
div[data-testid="stMarkdownContainer"] table {{ display: block; overflow-x: auto; max-width: 100%; }}
</style>
"""

st.markdown(STYLE, unsafe_allow_html=True)


def card(title: str, lines: list[str], files: list[str] | None = None, color: str | None = None, branch: bool = False) -> str:
    body = "".join(f"<p>{line}</p>" for line in lines)
    file_tags = "".join(f'<span class="file">{f}</span> ' for f in (files or []))
    style_attr = f' style="--accent:{color}"' if color else ""
    cls = "flow-card branch" if branch else "flow-card"
    return f'<div class="{cls}"{style_attr}><h4>{title}</h4>{body}{file_tags}</div>'


def row(*cards: str) -> str:
    return f'<div class="flow-row">{"".join(cards)}</div>'


def arrow(label: str | None = None) -> str:
    label_html = f'<div class="flow-label">{label}</div>' if label else ""
    return f'<div class="flow-arrow">↓</div>{label_html}'


st.subheader("The pipeline: scrape → clean → filter → tag → aggregate")

diagram = ['<div class="flow-wrap">']

diagram.append(
    row(
        card(
            "Google Play",
            [
                "google-play-scraper · com.spotify.music",
                "Locales: US, GB, IN, BR",
                "~842 real reviews (below the 1,000/locale target — Google's endpoint just returns fewer for some locales)",
            ],
            color=scrape_c,
        ),
        card(
            "App Store",
            [
                "Apple iTunes RSS feed · app id 324684580",
                "4 countries, up to 10 pages each (~500/country cap)",
                "~1,468 rows locally — returns 0 rows when run from a GitHub Actions IP (Apple-side, undocumented)",
            ],
            color=scrape_c,
        ),
        card(
            "Spotify Community",
            [
                "Khoros LiQL search API",
                "6 boards (incl. discovery_and_promo, music_discussion) — \"implemented\" ideas excluded at the source",
                "~1,500 ideas/posts, with kudos (vote) counts",
            ],
            color=scrape_c,
        ),
    )
)
diagram.append(arrow("merge_sources() unifies all three into one schema: id, source, text, rating, votes, locale, date, url, language"))

diagram.append(
    row(
        card(
            "Merge &amp; clean",
            [
                "analysis/merge.py",
                "Dedupe near-identical text (normalized: lowercase, punctuation stripped)",
                "Detect language per row (langdetect)",
            ],
            files=["data/reviews_raw.csv/json — not committed"],
        )
    )
)
diagram.append(arrow())

diagram.append(
    row(
        card(
            "Recency filter",
            [
                "analysis/cleaning.py + config.py — RECENCY_MONTHS = 24",
                "Google Play / App Store: drop anything older than the cutoff",
                "Community: keep if recent, OR old but top vote-quartile and not older than 2023",
            ],
            files=["data/reviews.csv/json"],
        )
    )
)
diagram.append(
    row(
        card(
            "No LLM_PROVIDER key set?",
            [
                "Pipeline stops right here.",
                "Analytics shows the Overview stats only; Chat says there's nothing to chat about yet.",
            ],
            branch=True,
        )
    )
)
diagram.append(arrow())

diagram.append(
    row(
        card(
            "LLM relevance filter + structured tagging",
            [
                "analysis/tagging.py + analysis/llm_client.py",
                "Gemini gemini-2.5-flash (default) or Groq llama-3.1-8b-instant — swappable via LLM_PROVIDER",
                "Batched 25 rows/request; every response cached on disk per row id",
                "Tags: theme (9 options), user segment (6 options), sentiment, severity 1–5",
            ],
            files=["data/tagged.csv/json"],
            color=llm_c,
        )
    )
)
diagram.append(arrow())

diagram.append(
    row(
        card(
            "Aggregate",
            [
                "analysis/aggregate.py",
                "Theme + segment tables, vote/severity-weighted unmet needs, representative quotes",
                "The six starter questions are re-answered from the current tables on every run",
            ],
            files=["data/summary_tables.json", "data/six_question_answers.json", "data/pipeline_summary.json"],
            color=llm_c,
        )
    )
)
diagram.append(arrow())

diagram.append(
    row(
        card(
            "Committed to main",
            [
                "git commit + push (locally, or by the weekly GitHub Action below)",
                "data/reviews_raw.* and data/llm_cache/ stay out of git — everything else above is checked in",
            ],
        )
    )
)
diagram.append(arrow())

diagram.append(
    row(
        card(
            "Chat page",
            [
                "analysis/chat.py",
                "Every answer is generated live, grounded in summary_tables.json — it never scrapes or re-tags",
            ],
            color=scrape_c,
        ),
        card(
            "Analytics page",
            [
                "views/analytics.py",
                "Charts, tables, and quote drill-downs from reviews.csv + summary_tables.json — also never scrapes",
            ],
            color=scrape_c,
        ),
    )
)
diagram.append("</div>")

st.markdown("".join(diagram), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# GitHub Actions automation
# ---------------------------------------------------------------------------
st.subheader("Staying fresh: the weekly GitHub Action")
st.caption(".github/workflows/weekly-scrape.yml — runs the exact same run_pipeline.py you'd run locally, on a schedule.")

auto = ['<div class="flow-wrap">']
auto.append(
    row(
        card("Every Monday, 06:00 UTC", ["cron: \"0 6 * * 1\""]),
        card("...or on demand", ["Actions tab → \"Weekly Spotify Review Scrape\" → Run workflow", "(workflow_dispatch)"]),
    )
)
auto.append(arrow("concurrency group \"weekly-scrape\" — a second trigger queues instead of racing the first"))
auto.append(row(card("Checkout repo + set up Python 3.12", ["actions/checkout, actions/setup-python (pip cache)"])))
auto.append(arrow())
auto.append(
    row(
        card(
            "Restore the LLM response cache",
            ["actions/cache, keyed by run id, falling back to the last cache", "avoids re-paying for rows already tagged in a previous run"],
        )
    )
)
auto.append(arrow())
auto.append(row(card("pip install -r requirements.txt", [])))
auto.append(arrow())
auto.append(
    row(
        card(
            "python run_pipeline.py",
            [
                "LLM_PROVIDER + GEMINI_API_KEY/GROQ_API_KEY come from repository secrets",
                "A quota exhaustion stops tagging early and keeps partial results — the job doesn't fail",
            ],
            color=llm_c,
        )
    )
)
auto.append(arrow())
auto.append(
    row(
        card(
            "Commit the refreshed data files",
            ["message: \"Weekly data refresh: &lt;date&gt;\"", "No changes this run? Exits cleanly, no empty commit"],
        )
    )
)
auto.append(arrow())
auto.append(
    row(
        card(
            "Merge origin/main, then push",
            [
                "git fetch + merge -X ours before pushing",
                "Handles main having moved since checkout (e.g. re-running an old job) — this run's fresh data wins on conflict, unrelated changes on main pass through untouched",
            ],
        )
    )
)
auto.append("</div>")

st.markdown("".join(auto), unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# Data lineage table
# ---------------------------------------------------------------------------
st.subheader("Data lineage")
st.markdown(
    """
| File | Produced by | Read by | Committed? |
|---|---|---|---|
| `data/reviews_raw.csv` / `.json` | `merge_clean_and_tag_language` (merge.py) | — (intermediate dump) | No |
| `data/reviews.csv` / `.json` | `apply_recency_filter` (cleaning.py) | Analytics page, Chat page (data-present check) | Yes |
| `data/tagged.csv` / `.json` | `apply_structured_tagging` (tagging.py) | — (same rows as reviews.csv once tagging has run) | Yes |
| `data/summary_tables.json` | `build_summary_tables` (aggregate.py) | Chat page (grounding), Analytics page (charts) | Yes |
| `data/six_question_answers.json` | `answer_six_questions` (aggregate.py) | Generated every run; not yet surfaced in either page | Yes |
| `data/pipeline_summary.json` | `run_pipeline.py` (merge + recency stats) | This page's "Last pipeline run" snapshot above | Yes |
| `data/llm_cache/` | `LLMClient` (llm_client.py) | Reused by the next pipeline run; persisted across CI runs via `actions/cache` | No |
"""
)

with st.expander("Resilience notes"):
    st.markdown(
        "- **Per-row LLM caching** — every relevance/tagging call is cached on disk keyed by row id, so a run "
        "that stops partway through (rate limit, quota) doesn't re-pay for rows it already tagged.\n"
        "- **Graceful quota handling** — `tagging.py` and `aggregate.py` detect a permanent quota exhaustion "
        "(vs. a transient 429) and stop that stage early with whatever was completed, instead of crashing the "
        "whole pipeline and losing it.\n"
        "- **No API key, no crash** — without a working `LLM_PROVIDER` key, the pipeline still scrapes, merges, "
        "and recency-filters real data; it just skips tagging and aggregation and logs a warning."
    )
