"""Shared fixtures: tests never need a real API key or network access."""
import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch):
    monkeypatch.setenv("LEO_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LEO_AGENT_MEMORY", "false")
    monkeypatch.setenv("LEO_VERBOSE", "false")
    monkeypatch.setenv("LEO_PASS_THRESHOLD", "70")
    monkeypatch.setenv("LEO_MAX_RETEACH_LOOPS", "2")
    from src.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
