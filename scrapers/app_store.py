from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

APP_ID = "324684580"
COUNTRIES = ["us", "gb", "in", "br"]
MAX_PAGES = 10
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}
COLUMNS = ["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"]


def _page_url(country: str, page: int) -> str:
    # NOTE: the documented .../sortby=mostrecent/json path returns zero entries as of
    # this writing (Apple silently ignores/breaks on that segment); omitting sortby
    # returns real reviews, so we deliberately drop it. See README limitations.
    return f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={APP_ID}/json"


def _fetch_page(country: str, page: int) -> list[dict]:
    url = _page_url(country, page)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            return data.get("feed", {}).get("entry") or []
        except Exception as exc:
            logger.warning("app_store country=%s page=%s attempt=%s failed: %s", country, page, attempt, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(2 * attempt)
    return []


def _safe_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_url(entry: dict) -> str | None:
    link = entry.get("link")
    if isinstance(link, list):
        for item in link:
            href = (item.get("attributes") or {}).get("href")
            if href:
                return href
        return None
    if isinstance(link, dict):
        return (link.get("attributes") or {}).get("href")
    return None


def _extract_text(entry: dict) -> str:
    title = ((entry.get("title") or {}).get("label") or "").strip()
    content = ((entry.get("content") or {}).get("label") or "").strip()
    if title and content and title.lower() not in content.lower():
        return f"{title}. {content}"
    return content or title


def scrape_app_store_reviews(countries: list[str] | None = None, max_pages: int = MAX_PAGES) -> pd.DataFrame:
    countries = countries or COUNTRIES
    rows: list[dict] = []
    for country in countries:
        country_rows = 0
        for page in range(1, max_pages + 1):
            entries = [e for e in _fetch_page(country, page) if "im:rating" in e]
            if not entries:
                break
            for entry in entries:
                text = _extract_text(entry)
                if not text:
                    continue
                review_id = (entry.get("id") or {}).get("label") or f"{country}-{page}-{country_rows}"
                rows.append(
                    {
                        "id": f"app_store:{review_id}",
                        "source": "app_store",
                        "text": text,
                        "rating": _safe_int((entry.get("im:rating") or {}).get("label")),
                        "votes": _safe_int((entry.get("im:voteSum") or {}).get("label")) or 0,
                        "locale": country,
                        "date": (entry.get("updated") or {}).get("label"),
                        "url": _extract_url(entry) or f"https://apps.apple.com/{country}/app/id{APP_ID}",
                        "language": None,
                    }
                )
                country_rows += 1
            time.sleep(0.5)
        logger.info("app_store country=%s rows=%s", country, country_rows)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)
