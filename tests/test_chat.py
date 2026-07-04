import pytest

from analysis.chat import (
    NO_DATA_REPLY,
    QUOTA_REPLY,
    answer_chat_question,
    build_chat_prompt,
)
from analysis.llm_client import LLMClient

SAMPLE_TABLES = {
    "total_rows": 2905,
    "total_relevant": 871,
    "themes": [
        {"theme": "filter_bubble_overpersonalization", "count": 49, "weighted_count": 31639, "share": 0.4231},
        {"theme": "discovery_buried_in_ui", "count": 103, "weighted_count": 9311, "share": 0.1245},
    ],
    "segments": [
        {"use_case_segment": "focus_work_listener", "count": 256, "share": 0.2939},
    ],
    "unmet_needs": [
        {"job_to_be_done": "discover authentic music", "theme": "filter_bubble_overpersonalization", "count": 1, "weighted_score": 53144.0},
    ],
    "quotes_by_theme": {
        "filter_bubble_overpersonalization": [
            {"text": "Mark / Disable AI Generated Songs, it floods Release Radar", "source": "community", "url": "https://example.com"},
        ]
    },
}


def make_client(tmp_path) -> LLMClient:
    return LLMClient(api_key="fake-key", cache_dir=tmp_path)


def test_build_chat_prompt_includes_evidence_and_question():
    prompt = build_chat_prompt("what frustrates users most?", SAMPLE_TABLES)
    assert "filter_bubble_overpersonalization" in prompt
    assert "focus_work_listener" in prompt
    assert "discover authentic music" in prompt
    assert "Mark / Disable AI Generated Songs" in prompt
    assert "User: what frustrates users most?" in prompt
    assert prompt.strip().endswith("Assistant:")


def test_build_chat_prompt_includes_recent_history_capped():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(20)]
    prompt = build_chat_prompt("follow up question", SAMPLE_TABLES, history)
    assert "turn 19" in prompt  # most recent turn present
    assert "turn 0" not in prompt  # oldest turns dropped by the cap


def test_answer_chat_question_empty_question_short_circuits_without_calling_model(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "generate_text", lambda p: (_ for _ in ()).throw(AssertionError("should not be called")))

    result = answer_chat_question(client, "   ", SAMPLE_TABLES)

    assert "ask me something" in result.lower()


def test_answer_chat_question_no_tables_returns_no_data_reply_without_calling_model(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "generate_text", lambda p: (_ for _ in ()).throw(AssertionError("should not be called")))

    assert answer_chat_question(client, "what do users want?", None) == NO_DATA_REPLY
    assert answer_chat_question(client, "what do users want?", {"total_relevant": 0}) == NO_DATA_REPLY


def test_answer_chat_question_calls_model_and_strips_whitespace(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    captured = {}

    def fake_generate_text(prompt):
        captured["prompt"] = prompt
        return "  Users mostly complain about repetitive recommendations.  \n"

    monkeypatch.setattr(client, "generate_text", fake_generate_text)

    result = answer_chat_question(client, "what's the biggest complaint?", SAMPLE_TABLES)

    assert result == "Users mostly complain about repetitive recommendations."
    assert "what's the biggest complaint?" in captured["prompt"]


def test_answer_chat_question_passes_through_history(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    captured = {}
    monkeypatch.setattr(client, "generate_text", lambda p: captured.setdefault("prompt", p) or "ok")

    history = [{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer"}]
    answer_chat_question(client, "a follow-up", SAMPLE_TABLES, history)

    assert "earlier question" in captured["prompt"]
    assert "earlier answer" in captured["prompt"]


def test_answer_chat_question_returns_friendly_message_on_quota_exhaustion(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "generate_text", lambda p: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")))

    assert answer_chat_question(client, "a question", SAMPLE_TABLES) == QUOTA_REPLY


def test_answer_chat_question_non_quota_error_propagates(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "generate_text", lambda p: (_ for _ in ()).throw(RuntimeError("totally unrelated failure")))

    with pytest.raises(RuntimeError, match="totally unrelated failure"):
        answer_chat_question(client, "a question", SAMPLE_TABLES)


def test_format_tables_truncates_long_quotes():
    tables = {
        "total_rows": 10,
        "total_relevant": 5,
        "themes": [],
        "segments": [],
        "unmet_needs": [],
        "quotes_by_theme": {"other": [{"text": "x" * 500, "source": "play_store"}]},
    }
    prompt = build_chat_prompt("q", tables)
    # the 500-char quote should be truncated, not dumped in full
    assert "x" * 500 not in prompt
    assert "x" * 220 in prompt
