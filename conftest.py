import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)


@pytest.fixture(autouse=True)
def _isolated_llm_env(monkeypatch):
    # A real .env (LLM_PROVIDER, GEMINI_API_KEY, GROQ_API_KEY) must never leak into
    # tests: it would make "default provider" tests pass or fail depending on
    # whatever a developer happens to have configured locally, and risks tests
    # accidentally making real API calls with real credentials.
    for var in ("LLM_PROVIDER", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
