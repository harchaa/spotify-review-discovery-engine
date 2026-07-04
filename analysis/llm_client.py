from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    # llama-3.1-8b-instant, not the 70b model: Groq's free tier caps
    # llama-3.3-70b-versatile at 100K tokens/day, which our batched structured
    # tagging burns through fast. The 8b model has a separate, far larger daily
    # budget and is more than adequate for fixed-enum classification.
    "groq": "llama-3.1-8b-instant",
}
API_KEY_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "llm_cache"
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 2
REQUEST_TIMEOUT_MS = 30_000  # google-genai's HttpOptions.timeout defaults to None (no timeout)
REQUEST_TIMEOUT_SECONDS = REQUEST_TIMEOUT_MS / 1000


def is_quota_exhausted(exc: BaseException) -> bool:
    """True if exc (or anything in its __cause__ chain) looks like a 429/quota
    error. generate_json's own RuntimeError message is generic ("Gemini call
    failed after N attempts...") - the real API error text only lives on the
    chained __cause__, so that has to be checked too, not just str(exc).

    Checks provider-specific status attributes first (Groq's RateLimitError.
    status_code, Gemini's ClientError.code) since those don't depend on
    message wording, then falls back to substring matching for providers
    that only expose the code in their string representation.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "status_code", None) == 429 or getattr(current, "code", None) == 429:
            return True
        text = str(current)
        if "RESOURCE_EXHAUSTED" in text or "429" in text:
            return True
        current = current.__cause__
    return False


class LLMClient:
    """Single wrapper around the LLM provider: one place to swap providers,
    handle retry/backoff, and cache responses on disk keyed by row id.

    Provider defaults to Gemini (the brief's specified provider) but can be
    swapped to Groq via provider="groq" or the LLM_PROVIDER env var - added
    because a specific Google AI Studio project/key was capped at 20
    requests/day, far below Gemini's documented 1,500/day free tier, and
    Groq's free tier was immediately available with no such restriction.
    """

    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | str | None = None,
        model: str | None = None,
    ):
        self.provider = provider or os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider {self.provider!r}; expected one of {sorted(DEFAULT_MODELS)}")
        self.api_key = api_key or os.environ.get(API_KEY_ENV_VARS[self.provider])
        self.model = model or DEFAULT_MODELS[self.provider]
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError(f"{API_KEY_ENV_VARS[self.provider]} is not set. Add it to your .env file.")

            if self.provider == "groq":
                from groq import Groq

                self._client = Groq(api_key=self.api_key, timeout=REQUEST_TIMEOUT_SECONDS)
            else:
                from google import genai
                from google.genai import types

                self._client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
                )
        return self._client

    def _call_model(self, prompt: str) -> str:
        client = self._get_client()
        if self.provider == "groq":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
        return response.text

    def _cache_path(self, cache_key: str) -> Path:
        # Sanitizing collapses distinct keys (e.g. "a:1" and "a_1"), so a content
        # hash suffix guarantees uniqueness while the prefix stays human-readable.
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in cache_key)[:80]
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{safe_key}_{digest}.json"

    def _read_cache(self, cache_key: str) -> Any | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return None

    def _write_cache(self, cache_key: str, value: Any) -> None:
        self._cache_path(cache_key).write_text(json.dumps(value))

    def get_cached(self, cache_key: str) -> Any | None:
        """Public read-only cache lookup for callers (e.g. tagging.py) that manage
        their own per-row cache keys around a single batched model call."""
        return self._read_cache(cache_key)

    def set_cached(self, cache_key: str, value: Any) -> None:
        self._write_cache(cache_key, value)

    def generate_json(self, cache_key: str, prompt: str, force: bool = False) -> Any:
        if not force:
            cached = self._read_cache(cache_key)
            if cached is not None:
                return cached

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self._call_model(prompt)
                parsed = json.loads(raw)
            except Exception as exc:  # covers API errors, rate limits, and malformed JSON
                last_exc = exc
                if attempt == MAX_RETRIES:
                    break
                wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning("cache_key=%s attempt=%s failed (%s); retrying in %ss", cache_key, attempt, exc, wait)
                time.sleep(wait)
                continue
            self._write_cache(cache_key, parsed)
            return parsed

        raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts for cache_key={cache_key}") from last_exc
