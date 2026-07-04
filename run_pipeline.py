from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from analysis.aggregate import answer_six_questions, build_summary_tables
from analysis.cleaning import apply_recency_filter, get_recency_summary
from analysis.llm_client import LLMClient
from analysis.merge import merge_clean_and_tag_language
from analysis.tagging import apply_relevance_filter, apply_structured_tagging
from scrapers.app_store import scrape_app_store_reviews
from scrapers.community import scrape_community_ideas
from scrapers.play_store import scrape_google_play_reviews

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_pipeline")

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _write_csv_json(df, stem: str) -> None:
    df.to_csv(DATA_DIR / f"{stem}.csv", index=False)
    df.to_json(DATA_DIR / f"{stem}.json", orient="records", indent=2, date_format="iso")


def run_pipeline(skip_llm: bool | None = None):
    logger.info("Scraping Google Play...")
    df_play = scrape_google_play_reviews()
    logger.info("Scraping App Store...")
    df_app = scrape_app_store_reviews()
    logger.info("Scraping Spotify Community...")
    df_community = scrape_community_ideas()

    merged, merge_summary = merge_clean_and_tag_language(df_play, df_app, df_community)
    _write_csv_json(merged, "reviews_raw")

    recency_summary = get_recency_summary(merged)
    filtered = apply_recency_filter(merged)

    print("\n=== Merge summary (before recency filter) ===")
    print(f"- total rows: {merge_summary['total_rows']}")
    print(f"- rows per source: {merge_summary['rows_per_source']}")
    print(f"- date range: {merge_summary['date_range']}")
    print(f"- % non-English: {merge_summary['pct_non_english']}")

    print("\n=== Recency filter summary ===")
    print(f"- cutoff date: {recency_summary['cutoff_date']}")
    print(f"- rows kept: {recency_summary['rows_kept']}")
    print(f"- rows lost per source: {recency_summary['rows_lost_per_source']}")
    print(f"- filtered date range: {recency_summary['date_range']}")
    print(f"- per-source date ranges: {recency_summary['source_date_ranges']}")

    skip_llm = skip_llm if skip_llm is not None else not os.environ.get("GEMINI_API_KEY")
    tables = None
    if skip_llm:
        logger.warning(
            "GEMINI_API_KEY is not set - skipping relevance filtering, tagging, and aggregation. "
            "Add the key to .env and re-run to complete the analysis steps."
        )
        _write_csv_json(filtered, "reviews")
    else:
        client = LLMClient()
        logger.info("Running relevance filter on %s rows...", len(filtered))
        filtered = apply_relevance_filter(client, filtered)
        logger.info("Running structured tagging on relevant rows...")
        filtered = apply_structured_tagging(client, filtered)
        _write_csv_json(filtered, "reviews")
        _write_csv_json(filtered, "tagged")

        logger.info("Building aggregate summary tables...")
        tables = build_summary_tables(filtered)
        (DATA_DIR / "summary_tables.json").write_text(json.dumps(tables, indent=2, default=str))

        logger.info("Generating grounded answers to the six questions...")
        answers = answer_six_questions(client, tables)
        (DATA_DIR / "six_question_answers.json").write_text(json.dumps(answers, indent=2))

        print("\n=== Analysis summary ===")
        print(f"- relevant rows: {tables['total_relevant']} / {tables['total_rows']}")
        print(f"- themes found: {len(tables['themes'])}")
        print(f"- segments found: {len(tables['segments'])}")

    summary_path = DATA_DIR / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps({"merge": merge_summary, "recency": recency_summary, "skipped_llm": skip_llm}, indent=2, default=str)
    )
    logger.info("Wrote %s", summary_path)

    return filtered


if __name__ == "__main__":
    run_pipeline()
