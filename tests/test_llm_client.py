import json
from unittest.mock import patch

import pytest

from analysis.llm_client import REQUEST_TIMEOUT_MS, REQUEST_TIMEOUT_SECONDS, LLMClient, is_quota_exhausted


def make_client(tmp_path, api_key="fake-key"):
    return LLMClient(api_key=api_key, cache_dir=tmp_path)


def test_generate_json_calls_model_and_writes_cache(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    calls = []

    def fake_call(prompt):
        calls.append(prompt)
        return json.dumps({"theme": "stale_recommendations"})

    monkeypatch.setattr(client, "_call_model", fake_call)

    result = client.generate_json("row-1", "classify this review")

    assert result == {"theme": "stale_recommendations"}
    assert len(calls) == 1
    assert client._cache_path("row-1").exists()


def test_cache_hit_skips_model_call(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client._cache_path("row-1").write_text(json.dumps({"theme": "cached"}))

    def fail_call(prompt):
        raise AssertionError("should not call the model on a cache hit")

    monkeypatch.setattr(client, "_call_model", fail_call)

    result = client.generate_json("row-1", "classify this review")

    assert result == {"theme": "cached"}


def test_force_bypasses_and_overwrites_cache(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    client._cache_path("row-1").write_text(json.dumps({"theme": "stale-cache"}))
    monkeypatch.setattr(client, "_call_model", lambda prompt: json.dumps({"theme": "fresh"}))

    result = client.generate_json("row-1", "classify this review", force=True)

    assert result == {"theme": "fresh"}
    assert json.loads(client._cache_path("row-1").read_text()) == {"theme": "fresh"}


def test_retries_on_transient_failure_then_succeeds(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    attempts = {"count": 0}

    def flaky_call(prompt):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return json.dumps({"theme": "recovered"})

    monkeypatch.setattr(client, "_call_model", flaky_call)

    result = client.generate_json("row-1", "classify this review")

    assert result == {"theme": "recovered"}
    assert attempts["count"] == 3


def test_raises_after_exhausting_retries_and_does_not_cache(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda prompt: (_ for _ in ()).throw(RuntimeError("still failing")))

    with pytest.raises(RuntimeError, match="Gemini call failed after"):
        client.generate_json("row-1", "classify this review")

    assert not client._cache_path("row-1").exists()


def test_malformed_json_response_triggers_retry_and_eventually_raises(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda prompt: "not valid json")

    with pytest.raises(RuntimeError, match="Gemini call failed after"):
        client.generate_json("row-1", "classify this review")


def test_get_client_sets_a_bounded_request_timeout(tmp_path):
    # Regression test: google-genai's HttpOptions.timeout defaults to None (no
    # timeout), so a stalled connection to the API would hang the pipeline
    # forever without this — the same failure mode hit with google-play-scraper.
    client = make_client(tmp_path)
    with patch("google.genai.Client") as mock_client_cls:
        client._get_client()

    _, kwargs = mock_client_cls.call_args
    assert kwargs["http_options"].timeout == REQUEST_TIMEOUT_MS


def test_missing_api_key_raises_only_when_model_is_actually_called(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = LLMClient(api_key=None, cache_dir=tmp_path)  # should not raise yet

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        client._get_client()


def test_cache_key_with_special_characters_is_sanitized_to_a_safe_filename(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda prompt: json.dumps({"ok": True}))

    client.generate_json("play_store:abc/123", "prompt")

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert "/" not in files[0].name and ":" not in files[0].name


def test_different_cache_keys_do_not_collide(tmp_path, monkeypatch):
    # "a:1" and "a_1" sanitize to the same characters, so this specifically
    # exercises the hash suffix that keeps them from clobbering each other's cache.
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda prompt: json.dumps({"prompt": prompt}))

    client.generate_json("a:1", "prompt-a")
    client.generate_json("a_1", "prompt-b")

    assert len(list(tmp_path.glob("*.json"))) == 2
    assert client.generate_json("a:1", "unused") == {"prompt": "prompt-a"}
    assert client.generate_json("a_1", "unused") == {"prompt": "prompt-b"}


def test_is_quota_exhausted_checks_the_chained_cause_not_just_the_outer_message():
    # generate_json raises a generic RuntimeError ("Gemini call failed after N
    # attempts...") and chains the real API error via `raise ... from exc`, so the
    # 429/RESOURCE_EXHAUSTED text only lives on __cause__, not the outer message.
    try:
        try:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. quota exceeded")
        except RuntimeError as cause:
            raise RuntimeError("Gemini call failed after 5 attempts for cache_key=x") from cause
    except RuntimeError as exc:
        assert is_quota_exhausted(exc) is True


def test_is_quota_exhausted_is_false_for_unrelated_errors():
    assert is_quota_exhausted(RuntimeError("some unrelated network error")) is False


def test_is_quota_exhausted_handles_a_cyclical_cause_chain_without_hanging():
    exc = RuntimeError("unrelated")
    exc.__cause__ = exc  # pathological but must not infinite-loop
    assert is_quota_exhausted(exc) is False


def test_is_quota_exhausted_detects_groq_style_status_code_attribute():
    # groq.RateLimitError (and OpenAI-SDK-style errors generally) expose
    # status_code=429 as an attribute rather than embedding it in the message.
    class FakeRateLimitError(Exception):
        status_code = 429

    assert is_quota_exhausted(FakeRateLimitError("rate limited")) is True


def test_default_provider_is_gemini_with_its_default_model(tmp_path):
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    assert client.provider == "gemini"
    assert client.model == "gemini-2.5-flash"


def test_groq_provider_uses_groq_default_model_and_api_key_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "groq-fake-key")

    client = LLMClient(provider="groq", cache_dir=tmp_path)

    assert client.model == "llama-3.1-8b-instant"
    assert client.api_key == "groq-fake-key"


def test_llm_provider_env_var_selects_provider_when_not_passed_explicitly(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    client = LLMClient(api_key="fake", cache_dir=tmp_path)
    assert client.provider == "groq"


def test_unknown_provider_raises_immediately(tmp_path):
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="not-a-real-provider", cache_dir=tmp_path)


def test_missing_groq_api_key_error_names_the_right_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = LLMClient(provider="groq", api_key=None, cache_dir=tmp_path)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        client._get_client()


def test_groq_get_client_passes_api_key_and_bounded_timeout(tmp_path):
    client = LLMClient(provider="groq", api_key="fake", cache_dir=tmp_path)
    with patch("groq.Groq") as mock_groq_cls:
        client._get_client()

    _, kwargs = mock_groq_cls.call_args
    assert kwargs["api_key"] == "fake"
    assert kwargs["timeout"] == REQUEST_TIMEOUT_SECONDS


def test_groq_call_model_uses_chat_completions_with_json_response_format(tmp_path):
    client = LLMClient(provider="groq", api_key="fake", cache_dir=tmp_path)

    fake_response = type(
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": '{"ok": true}'})()})()]},
    )()

    with patch("groq.Groq") as mock_groq_cls:
        mock_groq_cls.return_value.chat.completions.create.return_value = fake_response
        result = client._call_model("classify this")

    assert result == '{"ok": true}'
    _, kwargs = mock_groq_cls.return_value.chat.completions.create.call_args
    assert kwargs["model"] == "llama-3.1-8b-instant"
    assert kwargs["messages"] == [{"role": "user", "content": "classify this"}]
    assert kwargs["response_format"] == {"type": "json_object"}


def test_groq_call_model_omits_response_format_when_not_json_mode(tmp_path):
    client = LLMClient(provider="groq", api_key="fake", cache_dir=tmp_path)
    fake_response = type(
        "FakeResponse",
        (),
        {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": "a conversational reply"})()})()]},
    )()

    with patch("groq.Groq") as mock_groq_cls:
        mock_groq_cls.return_value.chat.completions.create.return_value = fake_response
        result = client._call_model("chat about this", json_mode=False)

    assert result == "a conversational reply"
    _, kwargs = mock_groq_cls.return_value.chat.completions.create.call_args
    assert "response_format" not in kwargs


def test_generate_text_returns_plain_string_and_is_never_cached(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    calls = []
    monkeypatch.setattr(client, "_call_model", lambda prompt, json_mode=True: (calls.append(json_mode), "a natural language answer")[1])

    result = client.generate_text("what do users think about discovery?")
    result2 = client.generate_text("what do users think about discovery?")

    assert result == "a natural language answer"
    assert calls == [False, False]  # json_mode=False both times, no caching to skip the second call
    assert list(tmp_path.glob("*.json")) == []  # generate_text never writes to the response cache


def test_generate_text_retries_then_raises_after_exhausting_attempts(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    monkeypatch.setattr(client, "_call_model", lambda prompt, json_mode=True: (_ for _ in ()).throw(RuntimeError("down")))

    with pytest.raises(RuntimeError, match="Gemini call failed after"):
        client.generate_text("a question")


def test_generate_text_recovers_from_a_transient_failure(tmp_path, monkeypatch):
    client = make_client(tmp_path)
    attempts = {"count": 0}

    def flaky(prompt, json_mode=True):
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return "recovered answer"

    monkeypatch.setattr(client, "_call_model", flaky)

    assert client.generate_text("a question") == "recovered answer"
    assert attempts["count"] == 2
