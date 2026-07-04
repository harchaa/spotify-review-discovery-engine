import pandas as pd

from analysis.merge import (
    COLUMNS,
    add_languages,
    build_summary,
    dedupe_reviews,
    detect_language,
    merge_clean_and_tag_language,
    merge_sources,
    normalize_text,
)


def _row(**kwargs) -> dict:
    base = {
        "id": "x",
        "source": "play_store",
        "text": "sample",
        "rating": 5,
        "votes": 0,
        "locale": "us",
        "date": "2026-06-01T00:00:00",
        "url": "https://example.com",
        "language": None,
    }
    base.update(kwargs)
    return base


def test_merge_sources_skips_empty_frames_and_keeps_column_order():
    df1 = pd.DataFrame([_row(id="a")])
    empty = pd.DataFrame(columns=COLUMNS)
    df2 = pd.DataFrame([_row(id="b", source="app_store")])

    merged = merge_sources(df1, empty, df2)

    assert list(merged.columns) == COLUMNS
    assert sorted(merged["id"]) == ["a", "b"]


def test_merge_sources_all_empty_returns_empty_with_columns():
    merged = merge_sources(pd.DataFrame(columns=COLUMNS), pd.DataFrame(columns=COLUMNS))
    assert merged.empty
    assert list(merged.columns) == COLUMNS


def test_normalize_text_collapses_case_punctuation_and_whitespace():
    assert normalize_text("Great App!!  Love it.") == normalize_text("great app love it")


def test_dedupe_removes_near_identical_text_keeping_first():
    df = pd.DataFrame(
        [
            _row(id="a", text="Recommendations are stale."),
            _row(id="b", text="recommendations are stale"),
            _row(id="c", text="Totally different complaint about ads"),
        ]
    )

    result = dedupe_reviews(df)

    assert sorted(result["id"]) == ["a", "c"]


def test_dedupe_drops_blank_text_rows():
    df = pd.DataFrame([_row(id="a", text="   "), _row(id="b", text="real feedback")])
    result = dedupe_reviews(df)
    assert list(result["id"]) == ["b"]


def test_detect_language_english_text():
    assert detect_language("This app has really helped me discover new music every week") == "en"


def test_detect_language_non_english_text():
    assert detect_language("Esta aplicacion es muy buena para escuchar musica todos los dias") == "es"


def test_detect_language_short_text_is_undetermined():
    assert detect_language("hi") == "und"
    assert detect_language("") == "und"
    assert detect_language(None) == "und"


def test_add_languages_fills_column_for_all_rows():
    df = pd.DataFrame([_row(id="a", text="I love discovering new music with this app")])
    result = add_languages(df)
    assert result.iloc[0]["language"] == "en"


def test_build_summary_counts_and_date_range():
    df = pd.DataFrame(
        [
            _row(id="a", source="play_store", date="2026-01-01T00:00:00", language="en"),
            _row(id="b", source="app_store", date="2026-03-01T00:00:00", language="es"),
            _row(id="c", source="community", date="2026-02-01T00:00:00", language="en"),
        ]
    )

    summary = build_summary(df)

    assert summary["total_rows"] == 3
    assert summary["rows_per_source"] == {"play_store": 1, "app_store": 1, "community": 1}
    assert summary["date_range"] == ["2026-01-01", "2026-03-01"]
    assert summary["pct_non_english"] == round(100 / 3, 1)


def test_build_summary_date_range_spans_mixed_naive_and_tz_aware_dates():
    # Regression test: play_store dates are naive ("...T00:00:00"), app_store and
    # community dates carry a UTC offset ("...+01:00"). pd.to_datetime(..., utc=True)
    # on a column mixing both silently coerces every format but the first-seen one
    # to NaT, which previously made the reported date range span only one source.
    df = pd.DataFrame(
        [
            _row(id="old", source="community", date="2014-03-06T16:35:50.919+01:00"),
            _row(id="mid", source="app_store", date="2026-06-15T18:19:19-07:00"),
            _row(id="new", source="play_store", date="2026-07-02T14:50:06"),
        ]
    )

    summary = build_summary(df)

    assert summary["date_range"] == ["2014-03-06", "2026-07-02"]


def test_build_summary_handles_empty_dataframe():
    summary = build_summary(pd.DataFrame(columns=COLUMNS))
    assert summary["total_rows"] == 0
    assert summary["rows_per_source"] == {}
    assert summary["date_range"] is None
    assert summary["pct_non_english"] == 0.0


def test_build_summary_handles_unparseable_dates():
    df = pd.DataFrame([_row(id="a", date="not-a-date")])
    summary = build_summary(df)
    assert summary["date_range"] is None


def test_merge_clean_and_tag_language_end_to_end():
    df_play = pd.DataFrame([_row(id="p1", source="play_store", text="Recommendations feel stale lately")])
    df_app = pd.DataFrame([_row(id="a1", source="app_store", text="recommendations feel stale lately")])  # near-dup of p1
    df_community = pd.DataFrame([_row(id="c1", source="community", text="Please add a way to reset my taste profile")])

    merged, summary = merge_clean_and_tag_language(df_play, df_app, df_community)

    assert sorted(merged["id"]) == ["c1", "p1"]  # a1 deduped away as near-identical to p1
    assert summary["total_rows"] == 2
    assert all(lang == "en" for lang in merged["language"])
