from __future__ import annotations

import logging
import socket
import time
from contextlib import contextmanager

import pandas as pd
from google_play_scraper import Sort, reviews

logger = logging.getLogger(__name__)

APP_ID = "com.spotify.music"
LOCALES = ["us", "gb", "in", "br"]
TARGET_PER_LOCALE = 1000
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30
COLUMNS = ["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"]


@contextmanager
def _bounded_socket_timeout(seconds: float):
    # google-play-scraper calls urlopen() with no timeout, so a stalled
    # connection blocks forever; urlopen falls back to the global socket
    # default when none is passed explicitly, so setting it here is the only
    # way to bound the call without patching the library itself.
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(seconds)
    try:
        yield
    finally:
        socket.setdefaulttimeout(previous)


def _fetch_locale(country: str, target: int) -> list[dict]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with _bounded_socket_timeout(REQUEST_TIMEOUT_SECONDS):
                batch, _ = reviews(
                    APP_ID,
                    lang="en",
                    country=country,
                    sort=Sort.NEWEST,
                    count=target,
                )
            return batch
        except Exception as exc:
            logger.warning("play_store locale=%s attempt=%s failed: %s", country, attempt, exc)
            if attempt == MAX_RETRIES:
                return []
            time.sleep(2 * attempt)
    return []


def scrape_google_play_reviews(locales: list[str] | None = None, target_per_locale: int = TARGET_PER_LOCALE) -> pd.DataFrame:
    locales = locales or LOCALES
    rows: list[dict] = []
    for country in locales:
        raw = _fetch_locale(country, target_per_locale)
        logger.info("play_store locale=%s rows=%s", country, len(raw))
        for r in raw:
            text = (r.get("content") or "").strip()
            if not text:
                continue
            at = r.get("at")
            rows.append(
                {
                    "id": f"play_store:{r.get('reviewId')}",
                    "source": "play_store",
                    "text": text,
                    "rating": r.get("score"),
                    "votes": r.get("thumbsUpCount") or 0,
                    "locale": country,
                    "date": at.isoformat() if at else None,
                    "url": f"https://play.google.com/store/apps/details?id={APP_ID}&reviewId={r.get('reviewId')}",
                    "language": None,
                }
            )

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(rows, columns=COLUMNS)
