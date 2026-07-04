from __future__ import annotations

from datetime import datetime, date
from typing import Any

import pandas as pd

from config import RECENCY_CUTOFF, COMMUNITY_ANCIENT_CUTOFF


def parse_date(value: Any) -> date | None:
    if pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


def apply_recency_filter(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "date" not in result.columns:
        result["date"] = pd.NaT
    if "votes" not in result.columns:
        result["votes"] = 0
    if "source" not in result.columns:
        result["source"] = "unknown"

    result["parsed_date"] = result["date"].apply(parse_date)
    result["recency_kept"] = False

    for source in ["play_store", "app_store"]:
        mask = result["source"] == source
        if not mask.any():
            continue
        dates = result.loc[mask, "parsed_date"]
        keep = dates.notna() & (dates >= RECENCY_CUTOFF)
        result.loc[mask, "recency_kept"] = keep.fillna(False).astype(object).map(bool)

    community_mask = result["source"] == "community"
    if community_mask.any():
        community_dates = result.loc[community_mask, "parsed_date"]
        community_votes = pd.to_numeric(result.loc[community_mask, "votes"], errors="coerce").fillna(0)
        recent_mask = community_dates.notna() & (community_dates >= RECENCY_CUTOFF)
        old_mask = community_dates.notna() & (community_dates < RECENCY_CUTOFF) & (community_dates >= COMMUNITY_ANCIENT_CUTOFF)
        threshold = community_votes.quantile(0.75)
        high_vote_mask = old_mask & (community_votes >= threshold)
        keep = recent_mask | high_vote_mask
        result.loc[community_mask, "recency_kept"] = keep.fillna(False).astype(object).map(bool)

    result = result.loc[result["recency_kept"]].copy()
    result["parsed_date"] = pd.to_datetime(result["parsed_date"], errors="coerce")
    return result


def get_recency_summary(df: pd.DataFrame) -> dict[str, Any]:
    filtered = apply_recency_filter(df)
    summary = {
        "cutoff_date": RECENCY_CUTOFF.isoformat(),
        "rows_kept": int(len(filtered)),
        "rows_lost_per_source": {},
        "date_range": None,
        "source_date_ranges": {},
    }
    if not filtered.empty:
        parsed_dates = filtered["parsed_date"].dropna()
        if not parsed_dates.empty:
            summary["date_range"] = [parsed_dates.min().date().isoformat(), parsed_dates.max().date().isoformat()]

    for source in ["play_store", "app_store", "community"]:
        source_rows = df[df["source"] == source]
        if source_rows.empty:
            summary["rows_lost_per_source"][source] = 0
            summary["source_date_ranges"][source] = None
            continue
        kept = filtered[filtered["source"] == source]
        summary["rows_lost_per_source"][source] = int(len(source_rows) - len(kept))
        source_dates = pd.to_datetime(kept["parsed_date"], errors="coerce").dropna()
        if not source_dates.empty:
            summary["source_date_ranges"][source] = [source_dates.min().date().isoformat(), source_dates.max().date().isoformat()]
        else:
            summary["source_date_ranges"][source] = None
    return summary
