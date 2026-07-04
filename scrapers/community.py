from __future__ import annotations

import logging
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

SEARCH_URL = "https://community.spotify.com/api/2.0/search"
# Boards most likely to contain discovery/recommendation/repetition feedback.
# "ideas_implemented" is deliberately excluded: those needs are already shipped,
# so they can't represent *unmet* needs (the recency filter would drop most of
# them anyway since they skew old, but excluding at the source saves API calls).
BOARDS = [
    "ideas_live",
    "ideas_no",
    "discovery_and_promo",
    "music_discussion",
    "app_and_features",
    "ongoing_issues",
]
PAGE_SIZE = 500
TARGET_TOTAL = 1500
MAX_RETRIES = 3
REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
}
COLUMNS = ["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"]


def _build_query(offset: int, page_size: int) -> str:
    board_list = ", ".join(f"'{b}'" for b in BOARDS)
    return (
        "SELECT id, subject, body, kudos.sum(weight), post_time, board.id, view_href "
        "FROM messages WHERE depth = 0 AND board.id IN "
        f"({board_list}) ORDER BY kudos.sum(weight) DESC LIMIT {page_size} OFFSET {offset}"
    )


def _fetch_page(offset: int, page_size: int) -> list[dict]:
    params = {"q": _build_query(offset, page_size)}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                logger.warning("community offset=%s api error: %s", offset, data.get("message"))
                return []
            return data.get("data", {}).get("items") or []
        except Exception as exc:
            logger.warning("community offset=%s attempt=%s failed: %s", offset, attempt, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(2 * attempt)
    return []


def _strip_html(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "lxml").get_text(separator=" ", strip=True)


def scrape_community_ideas(target_total: int = TARGET_TOTAL, page_size: int = PAGE_SIZE) -> pd.DataFrame:
    rows: list[dict] = []
    offset = 0
    while len(rows) < target_total:
        items = _fetch_page(offset, page_size)
        if not items:
            break
        for item in items:
            subject = (item.get("subject") or "").strip()
            body = _strip_html(item.get("body") or "")
            text = f"{subject}. {body}".strip(". ").strip()
            if not text:
                continue
            votes = ((item.get("kudos") or {}).get("sum") or {}).get("weight") or 0
            rows.append(
                {
                    "id": f"community:{item.get('id')}",
                    "source": "community",
                    "text": text,
                    "rating": None,
                    "votes": votes,
                    "locale": "global",
                    "date": item.get("post_time"),
                    "url": item.get("view_href"),
                    "language": None,
                }
            )
        logger.info("community offset=%s fetched=%s total=%s", offset, len(items), len(rows))
        if len(items) < page_size:
            break
        offset += page_size
        time.sleep(0.5)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows[:target_total], columns=COLUMNS)
