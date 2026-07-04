from unittest.mock import patch

from scrapers import app_store


def _entry(entry_id, rating, content, title="", vote_sum=None, updated="2026-01-01T00:00:00-07:00", link=None):
    entry = {
        "id": {"label": entry_id},
        "im:rating": {"label": str(rating)},
        "title": {"label": title},
        "content": {"label": content},
        "updated": {"label": updated},
        "link": link or {"attributes": {"rel": "related", "href": f"https://itunes.apple.com/review?id={entry_id}"}},
    }
    if vote_sum is not None:
        entry["im:voteSum"] = {"label": str(vote_sum)}
    return entry


def test_entries_without_rating_are_skipped():
    # first entry in some Apple RSS variants is app metadata, not a review
    app_metadata_entry = {"title": {"label": "App info, not a review"}}
    review_entry = _entry("1", 4, "Solid app")
    with patch.object(app_store, "_fetch_page", side_effect=[[app_metadata_entry, review_entry], []]):
        df = app_store.scrape_app_store_reviews(countries=["us"], max_pages=2)

    assert len(df) == 1
    assert df.iloc[0]["text"] == "Solid app"


def test_pagination_stops_when_page_has_no_qualifying_entries():
    calls = []

    def fake_fetch(country, page):
        calls.append(page)
        if page == 1:
            return [_entry("1", 5, "Page one review")]
        return []

    with patch.object(app_store, "_fetch_page", side_effect=fake_fetch):
        df = app_store.scrape_app_store_reviews(countries=["us"], max_pages=10)

    assert calls == [1, 2]
    assert len(df) == 1


def test_extract_url_handles_list_and_dict_links():
    list_link = [
        {"attributes": {"rel": "alternate", "href": ""}},
        {"attributes": {"rel": "related", "href": "https://example.com/review"}},
    ]
    entry = _entry("1", 3, "Review text", link=list_link)
    assert app_store._extract_url(entry) == "https://example.com/review"

    dict_entry = _entry("2", 3, "Another review")
    assert app_store._extract_url(dict_entry).startswith("https://itunes.apple.com")


def test_extract_text_combines_title_and_content_without_duplication():
    entry = _entry("1", 4, "the app crashes on launch", title="Buggy")
    assert app_store._extract_text(entry) == "Buggy. the app crashes on launch"

    entry_dup = _entry("2", 4, "Buggy app, crashes constantly", title="Buggy")
    assert app_store._extract_text(entry_dup) == "Buggy app, crashes constantly"


def test_malformed_rating_does_not_crash():
    entry = _entry("1", "not-a-number", "weird payload")
    with patch.object(app_store, "_fetch_page", side_effect=[[entry], []]):
        df = app_store.scrape_app_store_reviews(countries=["us"], max_pages=2)

    assert df.iloc[0]["rating"] is None


def test_network_failure_returns_empty_dataframe_without_raising():
    with patch("scrapers.app_store.requests.get", side_effect=RuntimeError("boom")):
        with patch.object(app_store.time, "sleep", return_value=None):
            df = app_store.scrape_app_store_reviews(countries=["us"], max_pages=2)

    assert df.empty
    assert list(df.columns) == app_store.COLUMNS
