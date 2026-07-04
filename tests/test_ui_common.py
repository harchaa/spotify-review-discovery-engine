import pandas as pd

import ui_common


def test_ranked_bar_chart_sorts_ascending_for_horizontal_display():
    records = [
        {"theme": "stale_recommendations", "weighted_count": 5},
        {"theme": "repetition_fatigue", "weighted_count": 20},
        {"theme": "other", "weighted_count": 1},
    ]
    fig = ui_common.ranked_bar_chart(records, "theme", "weighted_count", ui_common.THEME_LABELS, "#2a78d6", False, "count")
    bar = fig.data[0]
    # ascending order so the largest value renders at the top of a horizontal bar chart
    assert list(bar.x) == [1, 5, 20]
    assert list(bar.y) == ["Other", "Stale recommendations", "Repetition fatigue"]


def test_ranked_bar_chart_unmapped_key_falls_back_to_raw_value():
    records = [{"theme": "totally_unknown_theme", "weighted_count": 3}]
    fig = ui_common.ranked_bar_chart(records, "theme", "weighted_count", ui_common.THEME_LABELS, "#2a78d6", False, "count")
    assert fig.data[0].y[0] == "totally_unknown_theme"


def test_quotes_for_themes_interleaves_across_related_themes():
    pool = {
        "stale_recommendations": [{"text": "a1"}, {"text": "a2"}],
        "trust_in_recs": [{"text": "b1"}],
    }
    result = ui_common.quotes_for_themes(pool, ["stale_recommendations", "trust_in_recs"], limit=3)
    assert result == [{"text": "a1"}, {"text": "b1"}, {"text": "a2"}]


def test_quotes_for_themes_defaults_to_all_themes_when_none_given():
    pool = {"other": [{"text": "x"}]}
    result = ui_common.quotes_for_themes(pool, None, limit=3)
    assert result == [{"text": "x"}]


def test_quotes_for_themes_empty_pool_returns_empty_list():
    assert ui_common.quotes_for_themes({}, ["stale_recommendations"]) == []


def test_quotes_for_themes_ignores_themes_not_present_in_pool():
    pool = {"stale_recommendations": [{"text": "a1"}]}
    result = ui_common.quotes_for_themes(pool, ["stale_recommendations", "nonexistent_theme"], limit=3)
    assert result == [{"text": "a1"}]


def test_quotes_for_segments_picks_one_representative_per_segment():
    pool = {
        "focus_work_listener": [{"text": "f1"}, {"text": "f2"}],
        "genre_explorer_filter_bubble": [{"text": "g1"}],
    }
    result = ui_common.quotes_for_segments(pool, limit=5)
    assert result == [{"text": "f1"}, {"text": "g1"}]


def test_quotes_for_segments_respects_limit():
    pool = {f"segment_{i}": [{"text": f"q{i}"}] for i in range(5)}
    result = ui_common.quotes_for_segments(pool, limit=2)
    assert len(result) == 2


def test_quotes_for_segments_skips_segments_with_no_quotes():
    pool = {"focus_work_listener": [], "genre_explorer_filter_bubble": [{"text": "g1"}]}
    result = ui_common.quotes_for_segments(pool, limit=5)
    assert result == [{"text": "g1"}]


def test_overview_date_range_spans_mixed_naive_and_tz_aware_dates():
    # Regression test: pd.to_datetime(..., utc=True) on a column mixing naive
    # (Google Play) and tz-aware (App Store, Community) ISO strings previously
    # coerced every format but the first-seen one to NaT, understating the range.
    df = pd.DataFrame(
        {
            "date": [
                "2014-03-06T16:35:50.919+01:00",
                "2026-06-15T18:19:19-07:00",
                "2026-07-02T14:50:06",
            ]
        }
    )
    assert ui_common.overview_date_range(df) == "2014-03-06 — 2026-07-02"


def test_overview_date_range_empty_or_unparseable_returns_na():
    df = pd.DataFrame({"date": ["not-a-date", None]})
    assert ui_common.overview_date_range(df) == "n/a"
