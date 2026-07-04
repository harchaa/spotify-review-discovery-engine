from __future__ import annotations

import re
from typing import Any

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from analysis.cleaning import parse_date

DetectorFactory.seed = 0  # deterministic langdetect results across runs

COLUMNS = ["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"]
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCTUATION_RE = re.compile(r"[^\w\s]")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lowered = text.lower().strip()
    no_punct = _PUNCTUATION_RE.sub("", lowered)
    return _WHITESPACE_RE.sub(" ", no_punct).strip()


def merge_sources(*frames: pd.DataFrame) -> pd.DataFrame:
    non_empty = [f for f in frames if f is not None and not f.empty]
    if not non_empty:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(non_empty, ignore_index=True)[COLUMNS]


def dedupe_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["_norm_text"] = result["text"].apply(normalize_text)
    result = result[result["_norm_text"] != ""]
    result = result.drop_duplicates(subset=["_norm_text"], keep="first")
    return result.drop(columns=["_norm_text"])


def detect_language(text: str) -> str:
    normalized = text.strip() if isinstance(text, str) else ""
    if len(normalized) < 3:
        return "und"
    try:
        return detect(normalized)
    except LangDetectException:
        return "und"


def add_languages(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["language"] = result["text"].apply(detect_language)
    return result


def build_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_rows": int(len(df)),
        "rows_per_source": {},
        "date_range": None,
        "pct_non_english": 0.0,
    }
    if df.empty:
        return summary

    summary["rows_per_source"] = df.groupby("source").size().to_dict()

    # Row-wise parsing, not pd.to_datetime(..., utc=True): the raw "date" column mixes
    # naive (Google Play) and tz-aware (App Store, Community) ISO strings, and pandas'
    # vectorized parser locks onto the first format it sees, silently coercing every
    # other format to NaT instead of falling back to per-row parsing.
    parsed_dates = df["date"].apply(parse_date).dropna()
    if not parsed_dates.empty:
        summary["date_range"] = [parsed_dates.min().isoformat(), parsed_dates.max().isoformat()]

    if "language" in df.columns and df["language"].notna().any():
        non_english = (df["language"] != "en").sum()
        summary["pct_non_english"] = round(100.0 * non_english / len(df), 1)

    return summary


def merge_clean_and_tag_language(*frames: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = merge_sources(*frames)
    deduped = dedupe_reviews(merged)
    tagged = add_languages(deduped)
    summary = build_summary(tagged)
    return tagged, summary
