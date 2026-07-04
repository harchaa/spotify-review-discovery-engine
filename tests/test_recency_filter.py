from datetime import date, timedelta

import pandas as pd

from analysis.cleaning import apply_recency_filter, get_recency_summary
from config import RECENCY_MONTHS, RECENCY_CUTOFF, COMMUNITY_ANCIENT_CUTOFF


def test_play_store_and_app_store_use_recent_cutoff():
    df = pd.DataFrame(
        [
            {"id": "a", "source": "play_store", "date": (date.today() - timedelta(days=30 * RECENCY_MONTHS)).isoformat()},
            {"id": "b", "source": "play_store", "date": (date.today() - timedelta(days=30 * RECENCY_MONTHS + 1)).isoformat()},
            {"id": "c", "source": "app_store", "date": (date.today() - timedelta(days=10)).isoformat()},
            {"id": "d", "source": "app_store", "date": (date.today() - timedelta(days=30 * RECENCY_MONTHS + 100)).isoformat()},
        ]
    )

    result = apply_recency_filter(df)
    ids = sorted(result["id"].tolist())

    assert ids == ["a", "c"]
    assert bool(result.loc[result["id"] == "a", "recency_kept"].iloc[0]) is True
    assert "b" not in result["id"].tolist()


def test_community_allows_recent_or_high_vote_old_items():
    df = pd.DataFrame(
        [
            {"id": "old-low", "source": "community", "date": "2023-06-01", "votes": 1},
            {"id": "old-high", "source": "community", "date": "2023-06-01", "votes": 100},
            {"id": "recent", "source": "community", "date": date.today().isoformat(), "votes": 1},
            {"id": "ancient-low", "source": "community", "date": "2020-01-01", "votes": 1},
        ]
    )

    result = apply_recency_filter(df)

    kept_ids = set(result.loc[result["recency_kept"], "id"])
    assert kept_ids == {"old-high", "recent"}


def test_recency_summary_reports_lost_counts_and_cutoff():
    df = pd.DataFrame(
        [
            {"id": "p1", "source": "play_store", "date": (date.today() - timedelta(days=30 * RECENCY_MONTHS)).isoformat()},
            {"id": "p2", "source": "play_store", "date": (date.today() - timedelta(days=30 * RECENCY_MONTHS + 100)).isoformat()},
            {"id": "c1", "source": "community", "date": "2022-01-01", "votes": 1},
            {"id": "c2", "source": "community", "date": date.today().isoformat(), "votes": 1},
        ]
    )

    summary = get_recency_summary(df)

    assert summary["cutoff_date"] == RECENCY_CUTOFF.isoformat()
    assert summary["rows_lost_per_source"]["play_store"] == 1
    assert summary["rows_lost_per_source"]["community"] == 1
    assert summary["source_date_ranges"]["play_store"] is not None
    assert summary["source_date_ranges"]["community"] is not None


def test_recency_config_is_single_source_of_truth():
    assert RECENCY_MONTHS >= 1
    assert RECENCY_CUTOFF <= date.today()
    assert COMMUNITY_ANCIENT_CUTOFF < RECENCY_CUTOFF
