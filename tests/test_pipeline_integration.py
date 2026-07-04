import json
from datetime import date, timedelta

import pandas as pd

import run_pipeline
from config import RECENCY_MONTHS


def _row(id_, source, text, date_str, votes=0, rating=5):
    return {
        "id": id_,
        "source": source,
        "text": text,
        "rating": rating,
        "votes": votes,
        "locale": "us",
        "date": date_str,
        "url": "https://example.com",
        "language": None,
    }


def test_pipeline_scrapes_merges_filters_and_writes_outputs(tmp_path, monkeypatch):
    recent = date.today().isoformat()
    stale = (date.today() - timedelta(days=30 * RECENCY_MONTHS + 30)).isoformat()

    df_play = pd.DataFrame(
        [
            _row("p1", "play_store", "Discovery feels stale lately", recent),
            _row("p2", "play_store", "Old complaint about a removed feature", stale),
        ]
    )
    df_app = pd.DataFrame([_row("a1", "app_store", "Recommendations repeat the same songs", recent)])
    df_community = pd.DataFrame(
        [
            _row("c1", "community", "Add a way to reset my taste profile", recent, votes=5),
            _row("c2", "community", "Ancient but hugely popular idea", "2020-01-01", votes=9999),
        ]
    )

    monkeypatch.setattr(run_pipeline, "scrape_google_play_reviews", lambda: df_play)
    monkeypatch.setattr(run_pipeline, "scrape_app_store_reviews", lambda: df_app)
    monkeypatch.setattr(run_pipeline, "scrape_community_ideas", lambda: df_community)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", tmp_path)

    result = run_pipeline.run_pipeline(skip_llm=True)

    # stale play_store row and the pre-2023 community row are dropped
    assert sorted(result["id"]) == ["a1", "c1", "p1"]

    raw_csv = tmp_path / "reviews_raw.csv"
    filtered_csv = tmp_path / "reviews.csv"
    summary_json = tmp_path / "pipeline_summary.json"
    assert raw_csv.exists() and filtered_csv.exists() and summary_json.exists()

    raw_df = pd.read_csv(raw_csv)
    assert len(raw_df) == 5  # nothing deduped here, all texts distinct

    filtered_df = pd.read_csv(filtered_csv)
    assert sorted(filtered_df["id"]) == ["a1", "c1", "p1"]

    summary = json.loads(summary_json.read_text())
    assert summary["merge"]["total_rows"] == 5
    assert summary["recency"]["rows_kept"] == 3
    assert summary["skipped_llm"] is True
    # no LLM outputs should be produced when the analysis steps are skipped
    assert not (tmp_path / "tagged.csv").exists()
    assert not (tmp_path / "summary_tables.json").exists()
    assert not (tmp_path / "six_question_answers.json").exists()


def test_pipeline_handles_all_scrapers_returning_empty(tmp_path, monkeypatch):
    empty = pd.DataFrame(columns=["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"])
    monkeypatch.setattr(run_pipeline, "scrape_google_play_reviews", lambda: empty)
    monkeypatch.setattr(run_pipeline, "scrape_app_store_reviews", lambda: empty)
    monkeypatch.setattr(run_pipeline, "scrape_community_ideas", lambda: empty)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", tmp_path)

    result = run_pipeline.run_pipeline(skip_llm=True)

    assert result.empty
    assert (tmp_path / "reviews.csv").exists()


def test_pipeline_runs_llm_steps_when_key_present(tmp_path, monkeypatch):
    recent = date.today().isoformat()
    df_play = pd.DataFrame([_row("p1", "play_store", "Recommendations feel stale", recent)])
    empty = pd.DataFrame(columns=["id", "source", "text", "rating", "votes", "locale", "date", "url", "language"])

    monkeypatch.setattr(run_pipeline, "scrape_google_play_reviews", lambda: df_play)
    monkeypatch.setattr(run_pipeline, "scrape_app_store_reviews", lambda: empty)
    monkeypatch.setattr(run_pipeline, "scrape_community_ideas", lambda: empty)
    monkeypatch.setattr(run_pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(run_pipeline, "LLMClient", lambda: object())  # never actually called by the fakes below

    def fake_relevance(client, df):
        tagged = df.copy()
        tagged["relevant"] = True
        return tagged

    def fake_tagging(client, df):
        tagged = df.copy()
        tagged["theme"] = "stale_recommendations"
        tagged["use_case_segment"] = "discovery_seeker_distrusts_algo"
        tagged["job_to_be_done"] = "find new music"
        tagged["sentiment"] = "negative"
        tagged["severity"] = 4
        return tagged

    monkeypatch.setattr(run_pipeline, "apply_relevance_filter", fake_relevance)
    monkeypatch.setattr(run_pipeline, "apply_structured_tagging", fake_tagging)
    monkeypatch.setattr(run_pipeline, "build_summary_tables", lambda df: {"themes": [], "segments": [], "unmet_needs": [], "total_relevant": len(df), "total_rows": len(df)})
    monkeypatch.setattr(run_pipeline, "answer_six_questions", lambda client, tables: {"why_struggle_to_discover": "answer"})

    result = run_pipeline.run_pipeline(skip_llm=False)

    assert result.iloc[0]["theme"] == "stale_recommendations"
    assert (tmp_path / "tagged.csv").exists()
    assert (tmp_path / "summary_tables.json").exists()
    assert (tmp_path / "six_question_answers.json").exists()

    summary = json.loads((tmp_path / "pipeline_summary.json").read_text())
    assert summary["skipped_llm"] is False

    answers = json.loads((tmp_path / "six_question_answers.json").read_text())
    assert answers["why_struggle_to_discover"] == "answer"
