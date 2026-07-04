import json

import pandas as pd
import pytest

from analysis.aggregate import (
    SIX_QUESTIONS,
    answer_six_questions,
    build_summary_tables,
    representative_quotes,
    segment_details,
    segment_sizes,
    theme_frequency,
    top_unmet_needs,
)
from analysis.llm_client import LLMClient


def _row(id_, source="play_store", text="text", votes=0, relevant=True, theme=None, segment=None, job=None, severity=None, url="u"):
    return {
        "id": id_,
        "source": source,
        "text": text,
        "rating": 3,
        "votes": votes,
        "locale": "us",
        "date": "2026-01-01",
        "url": url,
        "language": "en",
        "relevant": relevant,
        "theme": theme,
        "use_case_segment": segment,
        "job_to_be_done": job,
        "severity": severity,
    }


def test_theme_frequency_counts_and_weights_community_votes():
    df = pd.DataFrame(
        [
            _row("p1", theme="stale_recommendations"),
            _row("p2", theme="stale_recommendations"),
            _row("c1", source="community", votes=99, theme="repetition_fatigue"),
        ]
    )

    result = theme_frequency(df)

    stale = result.set_index("theme").loc["stale_recommendations"]
    repetition = result.set_index("theme").loc["repetition_fatigue"]
    assert stale["count"] == 2
    assert stale["weighted_count"] == 2  # 1 + 1, no community boost
    assert repetition["weighted_count"] == 100  # 1 + 99 votes
    assert abs(result["share"].sum() - 1.0) < 1e-6


def test_theme_frequency_excludes_irrelevant_rows():
    df = pd.DataFrame([_row("p1", theme="stale_recommendations", relevant=False)])
    result = theme_frequency(df)
    assert result.empty


def test_theme_frequency_missing_column_returns_empty_with_schema():
    df = pd.DataFrame([{"id": "p1", "source": "play_store", "relevant": True}])
    result = theme_frequency(df)
    assert result.empty
    assert list(result.columns) == ["theme", "count", "weighted_count", "share"]


def test_theme_frequency_empty_dataframe():
    result = theme_frequency(pd.DataFrame())
    assert result.empty


def test_segment_sizes_basic():
    df = pd.DataFrame(
        [
            _row("p1", segment="focus_work_listener"),
            _row("p2", segment="focus_work_listener"),
            _row("p3", segment="genre_explorer_filter_bubble"),
        ]
    )
    result = segment_sizes(df)
    assert result.set_index("use_case_segment").loc["focus_work_listener", "count"] == 2
    assert abs(result["share"].sum() - 1.0) < 1e-6


def test_top_unmet_needs_ranks_by_weighted_severity_and_respects_top_n():
    df = pd.DataFrame(
        [
            _row("p1", theme="stale_recommendations", job="find fresh music", severity=5),
            _row("p2", theme="stale_recommendations", job="find fresh music", severity=5),
            _row("p3", theme="repetition_fatigue", job="avoid repeats", severity=1),
        ]
    )
    result = top_unmet_needs(df, top_n=1)
    assert len(result) == 1
    assert result.iloc[0]["job_to_be_done"] == "find fresh music"
    assert result.iloc[0]["count"] == 2
    assert result.iloc[0]["weighted_score"] == 10  # (1+1)*5


def test_top_unmet_needs_ignores_blank_job_to_be_done():
    df = pd.DataFrame([_row("p1", theme="other", job="", severity=3)])
    result = top_unmet_needs(df)
    assert result.empty


def test_representative_quotes_groups_sorts_and_limits():
    df = pd.DataFrame(
        [
            _row("p1", theme="stale_recommendations", text="low vote quote", votes=0),
            _row("c1", source="community", theme="stale_recommendations", text="high vote quote", votes=500),
            _row("p2", theme="stale_recommendations", text="third quote", votes=0),
            _row("p3", theme="stale_recommendations", text="fourth quote (should be cut)", votes=0),
        ]
    )
    result = representative_quotes(df, "theme", quotes_per_group=3)
    quotes = result["stale_recommendations"]
    assert len(quotes) == 3
    assert quotes[0]["text"] == "high vote quote"
    assert quotes[0]["source"] == "community"


def test_representative_quotes_missing_column_returns_empty_dict():
    df = pd.DataFrame([_row("p1")])
    df = df.drop(columns=["use_case_segment"])
    assert representative_quotes(df, "use_case_segment") == {}


def test_segment_details_reports_top_themes_and_jobs():
    df = pd.DataFrame(
        [
            _row("p1", segment="focus_work_listener", theme="discovery_breaks_focus", job="stay in flow while working"),
            _row("p2", segment="focus_work_listener", theme="discovery_breaks_focus", job="stay in flow while working"),
            _row("p3", segment="focus_work_listener", theme="repetition_fatigue", job="hear something new"),
            _row("p4", segment="genre_explorer_filter_bubble", theme="filter_bubble_overpersonalization", job="escape the bubble"),
        ]
    )
    details = segment_details(df, top_n=2)

    focus = details["focus_work_listener"]
    assert focus["top_themes"][0] == {"theme": "discovery_breaks_focus", "count": 2}
    assert focus["top_jobs"][0] == "stay in flow while working"
    assert "genre_explorer_filter_bubble" in details


def test_segment_details_missing_columns_returns_empty_dict():
    df = pd.DataFrame([{"id": "p1", "relevant": True}])
    assert segment_details(df) == {}


def test_segment_details_empty_dataframe():
    assert segment_details(pd.DataFrame()) == {}


def test_build_summary_tables_includes_segment_details():
    df = pd.DataFrame([_row("p1", segment="focus_work_listener", theme="repetition_fatigue", job="hear something new")])
    summary = build_summary_tables(df)
    assert "focus_work_listener" in summary["segment_details"]


def test_build_summary_tables_reports_relevant_vs_total_counts():
    df = pd.DataFrame(
        [
            _row("p1", theme="stale_recommendations", segment="focus_work_listener", job="x", severity=3),
            _row("p2", relevant=False),
        ]
    )
    summary = build_summary_tables(df)
    assert summary["total_rows"] == 2
    assert summary["total_relevant"] == 1
    assert len(summary["themes"]) == 1
    assert "quotes_by_theme" in summary and "quotes_by_segment" in summary


def test_answer_six_questions_calls_model_once_per_question(tmp_path, monkeypatch):
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return json.dumps({"answer": "Evidence-grounded answer."})

    monkeypatch.setattr(client, "_call_model", fake_call)

    answers = answer_six_questions(client, {"themes": [], "segments": [], "unmet_needs": []})

    assert len(calls) == len(SIX_QUESTIONS)
    assert set(answers.keys()) == {key for key, _ in SIX_QUESTIONS}
    assert all(a == "Evidence-grounded answer." for a in answers.values())


def test_answer_six_questions_always_regenerates_instead_of_caching(tmp_path, monkeypatch):
    # The cache key is fixed per question (not content-addressed), so caching
    # here would mean a recurring pipeline run (e.g. a weekly scrape) keeps
    # serving the first run's answers forever even as the underlying data
    # changes - force=True is required so every call reflects current tables.
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    calls = []
    monkeypatch.setattr(client, "_call_model", lambda p: (calls.append(p), json.dumps({"answer": "x"}))[1])

    tables = {"themes": [], "segments": [], "unmet_needs": []}
    answer_six_questions(client, tables)
    answer_six_questions(client, tables)

    assert len(calls) == 2 * len(SIX_QUESTIONS)  # both passes hit the model, no caching


def test_answer_six_questions_missing_answer_key_defaults_to_empty_string(tmp_path, monkeypatch):
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda p: json.dumps({}))

    answers = answer_six_questions(client, {"themes": [], "segments": [], "unmet_needs": []})

    assert all(a == "" for a in answers.values())


def test_answer_six_questions_stops_early_on_quota_exhaustion_keeping_partial_answers(tmp_path, monkeypatch):
    # Regression test: without this, hitting the free-tier daily quota partway
    # through the six questions would crash the whole pipeline run and lose the
    # tagging/aggregation work that already completed earlier in the same run.
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    calls = []

    def flaky_call(prompt):
        # Quota exhaustion is permanent for the rest of the run, not a one-off blip:
        # the first 2 questions succeed, then every call (including retries) fails,
        # so generate_json exhausts its retries and raises for question 3 onward.
        calls.append(prompt)
        if len(calls) > 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. quota exceeded, retry in 33s")
        return json.dumps({"answer": "a real answer"})

    monkeypatch.setattr(client, "_call_model", flaky_call)

    answers = answer_six_questions(client, {"themes": [], "segments": [], "unmet_needs": []})

    assert len(answers) == 2  # first two questions succeeded before the third failed
    assert all(a == "a real answer" for a in answers.values())


def test_answer_six_questions_non_quota_error_still_propagates(tmp_path, monkeypatch):
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda p: (_ for _ in ()).throw(RuntimeError("totally unrelated failure")))

    with pytest.raises(RuntimeError, match="Gemini call failed after"):
        answer_six_questions(client, {"themes": [], "segments": [], "unmet_needs": []})
