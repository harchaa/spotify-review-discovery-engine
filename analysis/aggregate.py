from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from analysis.llm_client import LLMClient, is_quota_exhausted

logger = logging.getLogger(__name__)

QUOTES_PER_GROUP = 3
DEFAULT_SEVERITY = 3
TOP_UNMET_NEEDS = 10

SIX_QUESTIONS = [
    ("why_struggle_to_discover", "Why do users struggle to discover new music?"),
    ("recommendation_frustrations", "What are the most common frustrations with recommendations?"),
    ("desired_listening_behaviors", "What listening behaviors are users trying to achieve?"),
    ("repetition_causes", "What causes users to repeatedly listen to the same content?"),
    ("segment_challenges", "Which user segments experience different discovery challenges?"),
    ("unmet_needs", "What unmet needs emerge consistently across reviews?"),
]


def _relevant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "relevant" not in df.columns:
        return df
    return df[df["relevant"] == True]  # noqa: E712 (explicit compare is NaN-safe)


def _weight(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    votes = pd.to_numeric(df["votes"], errors="coerce").fillna(0) if "votes" in df.columns else pd.Series(0, index=df.index)
    is_community = (df["source"] == "community") if "source" in df.columns else pd.Series(False, index=df.index)
    return 1 + votes.where(is_community, 0)


def theme_frequency(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["theme", "count", "weighted_count", "share"]
    if df.empty or "theme" not in df.columns:
        return pd.DataFrame(columns=columns)
    tagged = _relevant(df).dropna(subset=["theme"]).copy()
    if tagged.empty:
        return pd.DataFrame(columns=columns)
    tagged["_weight"] = _weight(tagged)
    grouped = tagged.groupby("theme").agg(count=("id", "count"), weighted_count=("_weight", "sum")).reset_index()
    total = grouped["weighted_count"].sum()
    grouped["share"] = (grouped["weighted_count"] / total).round(4) if total else 0.0
    return grouped.sort_values("weighted_count", ascending=False).reset_index(drop=True)[columns]


def segment_sizes(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["use_case_segment", "count", "share"]
    if df.empty or "use_case_segment" not in df.columns:
        return pd.DataFrame(columns=columns)
    tagged = _relevant(df).dropna(subset=["use_case_segment"])
    if tagged.empty:
        return pd.DataFrame(columns=columns)
    grouped = tagged.groupby("use_case_segment").size().reset_index(name="count")
    total = grouped["count"].sum()
    grouped["share"] = (grouped["count"] / total).round(4) if total else 0.0
    return grouped.sort_values("count", ascending=False).reset_index(drop=True)[columns]


def top_unmet_needs(df: pd.DataFrame, top_n: int = TOP_UNMET_NEEDS) -> pd.DataFrame:
    columns = ["job_to_be_done", "theme", "count", "weighted_score"]
    if df.empty or "job_to_be_done" not in df.columns or "theme" not in df.columns:
        return pd.DataFrame(columns=columns)
    tagged = _relevant(df).dropna(subset=["theme", "job_to_be_done"]).copy()
    tagged = tagged[tagged["job_to_be_done"] != ""]
    if tagged.empty:
        return pd.DataFrame(columns=columns)
    severity = pd.to_numeric(tagged.get("severity"), errors="coerce").fillna(DEFAULT_SEVERITY)
    tagged["_weight"] = _weight(tagged) * severity
    grouped = (
        tagged.groupby(["job_to_be_done", "theme"])
        .agg(count=("id", "count"), weighted_score=("_weight", "sum"))
        .reset_index()
    )
    return grouped.sort_values("weighted_score", ascending=False).head(top_n).reset_index(drop=True)[columns]


def representative_quotes(df: pd.DataFrame, group_col: str, quotes_per_group: int = QUOTES_PER_GROUP) -> dict[str, list[dict]]:
    if df.empty or group_col not in df.columns:
        return {}
    tagged = _relevant(df).dropna(subset=[group_col]).copy()
    if tagged.empty:
        return {}
    tagged["_weight"] = _weight(tagged)
    result: dict[str, list[dict]] = {}
    for key, group in tagged.groupby(group_col):
        top = group.sort_values("_weight", ascending=False).head(quotes_per_group)
        result[key] = [
            {"text": row["text"], "source": row["source"], "url": row.get("url")} for _, row in top.iterrows()
        ]
    return result


def segment_details(df: pd.DataFrame, top_n: int = 3) -> dict[str, dict[str, list]]:
    if df.empty or "use_case_segment" not in df.columns or "theme" not in df.columns:
        return {}
    tagged = _relevant(df).dropna(subset=["use_case_segment"])
    if tagged.empty:
        return {}
    result: dict[str, dict[str, list]] = {}
    for segment, group in tagged.groupby("use_case_segment"):
        theme_counts = group["theme"].dropna().value_counts().head(top_n)
        jobs = group.get("job_to_be_done", pd.Series(dtype=str)).dropna()
        jobs = jobs[jobs != ""]
        result[segment] = {
            "top_themes": [{"theme": t, "count": int(c)} for t, c in theme_counts.items()],
            "top_jobs": jobs.value_counts().head(top_n).index.tolist(),
        }
    return result


def build_summary_tables(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "themes": theme_frequency(df).to_dict(orient="records"),
        "segments": segment_sizes(df).to_dict(orient="records"),
        "segment_details": segment_details(df),
        "unmet_needs": top_unmet_needs(df).to_dict(orient="records"),
        "quotes_by_theme": representative_quotes(df, "theme"),
        "quotes_by_segment": representative_quotes(df, "use_case_segment"),
        "total_relevant": int(len(_relevant(df))),
        "total_rows": int(len(df)),
    }


def _answer_prompt(question: str, tables: dict[str, Any]) -> str:
    return (
        "You are a product analyst. Using ONLY the aggregated evidence tables below "
        "(do not invent facts beyond them), write a short, evidence-grounded answer "
        f'(2-4 sentences) to this question: "{question}"\n\n'
        f"Theme frequency (count, weighted_count, share): {tables['themes']}\n"
        f"Segment sizes (count, share): {tables['segments']}\n"
        f"Top unmet needs: {tables['unmet_needs']}\n\n"
        'Respond ONLY with JSON of the exact shape {"answer": "..."}.'
    )


def answer_six_questions(client: LLMClient, tables: dict[str, Any]) -> dict[str, str]:
    # force=True: these answers must reflect the CURRENT aggregated tables, not
    # whatever was true the first time this ran. The cache key is fixed per
    # question (not content-addressed), so without this a recurring pipeline
    # run (e.g. a weekly scrape) would keep serving the first run's answers
    # forever even as the underlying data changes week to week.
    answers: dict[str, str] = {}
    for key, question in SIX_QUESTIONS:
        try:
            parsed = client.generate_json(f"answer:{key}", _answer_prompt(question, tables), force=True)
        except Exception as exc:
            if not is_quota_exhausted(exc):
                raise
            logger.warning(
                "answer_six_questions: stopping early at '%s' - API quota appears exhausted (%s). "
                "Re-running later (e.g. after the daily quota resets) will fill in the rest.",
                key,
                exc,
            )
            break
        answers[key] = parsed.get("answer", "")
    return answers
