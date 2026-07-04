import socket
from datetime import datetime
from unittest.mock import patch

from scrapers import play_store


def _fake_review(review_id, content, score=4, thumbs=0, at=None):
    return {
        "reviewId": review_id,
        "content": content,
        "score": score,
        "thumbsUpCount": thumbs,
        "at": at or datetime(2026, 1, 1, 12, 0, 0),
    }


def test_maps_fields_into_unified_schema():
    fake_batch = [_fake_review("r1", "Great app but recs are stale", score=3, thumbs=5)]
    with patch.object(play_store, "reviews", return_value=(fake_batch, None)):
        df = play_store.scrape_google_play_reviews(locales=["us"], target_per_locale=10)

    assert list(df.columns) == play_store.COLUMNS
    assert len(df) == 1
    row = df.iloc[0]
    assert row["id"] == "play_store:r1"
    assert row["source"] == "play_store"
    assert row["text"] == "Great app but recs are stale"
    assert row["rating"] == 3
    assert row["votes"] == 5
    assert row["locale"] == "us"


def test_blank_content_rows_are_dropped():
    fake_batch = [_fake_review("r1", "   "), _fake_review("r2", "Real feedback here")]
    with patch.object(play_store, "reviews", return_value=(fake_batch, None)):
        df = play_store.scrape_google_play_reviews(locales=["us"], target_per_locale=10)

    assert len(df) == 1
    assert df.iloc[0]["id"] == "play_store:r2"


def test_network_failure_returns_empty_dataframe_without_raising():
    with patch.object(play_store, "reviews", side_effect=RuntimeError("boom")):
        with patch.object(play_store.time, "sleep", return_value=None):
            df = play_store.scrape_google_play_reviews(locales=["us"], target_per_locale=10)

    assert df.empty
    assert list(df.columns) == play_store.COLUMNS


def test_multiple_locales_are_all_collected():
    def fake_reviews(app_id, lang, country, sort, count):
        return [_fake_review(f"{country}-1", f"feedback from {country}")], None

    with patch.object(play_store, "reviews", side_effect=fake_reviews):
        df = play_store.scrape_google_play_reviews(locales=["us", "gb"], target_per_locale=10)

    assert sorted(df["locale"].unique()) == ["gb", "us"]
    assert len(df) == 2


def test_missing_thumbs_up_defaults_to_zero():
    fake_batch = [{"reviewId": "r1", "content": "no thumbs field", "score": 5, "at": datetime(2026, 1, 1)}]
    with patch.object(play_store, "reviews", return_value=(fake_batch, None)):
        df = play_store.scrape_google_play_reviews(locales=["us"], target_per_locale=10)

    assert df.iloc[0]["votes"] == 0


def test_bounded_socket_timeout_sets_and_restores_default():
    original = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(None)
        with play_store._bounded_socket_timeout(5):
            assert socket.getdefaulttimeout() == 5
        assert socket.getdefaulttimeout() is None
    finally:
        socket.setdefaulttimeout(original)


def test_bounded_socket_timeout_restores_previous_value_even_on_exception():
    original = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(9)
        try:
            with play_store._bounded_socket_timeout(2):
                raise ValueError("boom")
        except ValueError:
            pass
        assert socket.getdefaulttimeout() == 9
    finally:
        socket.setdefaulttimeout(original)


def test_stalled_connection_times_out_and_retries_instead_of_hanging_forever():
    # google-play-scraper's urlopen() call has no timeout of its own, so a
    # stalled server would hang the pipeline forever without this guard.
    # This simulates that failure mode via a real socket.timeout.
    with patch.object(play_store, "reviews", side_effect=socket.timeout("timed out")):
        with patch.object(play_store.time, "sleep", return_value=None):
            df = play_store.scrape_google_play_reviews(locales=["us"], target_per_locale=10)

    assert df.empty
    assert list(df.columns) == play_store.COLUMNS
