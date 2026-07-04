from unittest.mock import patch

from scrapers import community


def _item(item_id, subject, body, votes=0, board="ideas_live", post_time="2026-01-01T00:00:00+01:00"):
    return {
        "id": item_id,
        "subject": subject,
        "body": body,
        "kudos": {"sum": {"weight": votes}},
        "board": {"id": board},
        "post_time": post_time,
        "view_href": f"https://community.spotify.com/t5/x/idi-p/{item_id}",
    }


def test_html_is_stripped_from_body():
    item = _item("1", "Bring back shuffle", "<P>Please <B>bring back</B> the old shuffle</P>", votes=42)
    with patch.object(community, "_fetch_page", side_effect=[[item], []]):
        df = community.scrape_community_ideas(target_total=10, page_size=5)

    assert len(df) == 1
    assert "<P>" not in df.iloc[0]["text"]
    assert "bring back" in df.iloc[0]["text"].lower()
    assert df.iloc[0]["votes"] == 42


def test_pagination_stops_when_page_smaller_than_page_size():
    def fake_fetch(offset, page_size):
        if offset == 0:
            return [_item(str(i), f"idea {i}", "body") for i in range(page_size)]
        return [_item("last", "final idea", "body")]

    with patch.object(community, "_fetch_page", side_effect=fake_fetch):
        df = community.scrape_community_ideas(target_total=100, page_size=5)

    assert len(df) == 6  # first full page (5) + partial page (1), then stop


def test_missing_kudos_defaults_votes_to_zero():
    item = {
        "id": "1",
        "subject": "No votes field",
        "body": "plain text",
        "board": {"id": "ideas_live"},
        "post_time": "2026-01-01T00:00:00+01:00",
        "view_href": "https://community.spotify.com/t5/x/idi-p/1",
    }
    with patch.object(community, "_fetch_page", side_effect=[[item], []]):
        df = community.scrape_community_ideas(target_total=10, page_size=5)

    assert df.iloc[0]["votes"] == 0


def test_blank_subject_and_body_are_dropped():
    item = _item("1", "   ", "<P>   </P>")
    with patch.object(community, "_fetch_page", side_effect=[[item], []]):
        df = community.scrape_community_ideas(target_total=10, page_size=5)

    assert df.empty


def test_api_error_status_returns_empty_without_raising():
    with patch("scrapers.community.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"status": "error", "message": "Invalid query syntax"}
        df = community.scrape_community_ideas(target_total=10, page_size=5)

    assert df.empty
    assert list(df.columns) == community.COLUMNS


def test_target_total_caps_result_even_with_more_available():
    def fake_fetch(offset, page_size):
        return [_item(f"{offset}-{i}", f"idea {offset}-{i}", "body") for i in range(page_size)]

    with patch.object(community, "_fetch_page", side_effect=fake_fetch):
        df = community.scrape_community_ideas(target_total=7, page_size=5)

    assert len(df) == 7
