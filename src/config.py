"""
Central configuration for Leo.

All secrets come from environment variables (loaded from `.env` via python-dotenv).
Nothing sensitive is hardcoded anywhere in the codebase.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from crewai import LLM
from dotenv import load_dotenv

load_dotenv()  # no-op if .env is absent (e.g. in CI where real env vars are set)

# Maps the provider prefix of LEO_MODEL to the env var that must hold its key.
_PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

# Groq exposes an OpenAI-compatible API. Routing through CrewAI's native OpenAI client
# avoids the LiteLLM path, which currently sends a message field ("cache_breakpoint")
# that Groq rejects.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str
    temperature: float
    pass_threshold: float
    max_reteach_loops: int
    agent_memory: bool
    verbose: bool
    serper_enabled: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        model=os.getenv("LEO_MODEL", "openai/gpt-4o-mini"),
        temperature=float(os.getenv("LEO_TEMPERATURE", "0.3")),
        pass_threshold=float(os.getenv("LEO_PASS_THRESHOLD", "70")),
        max_reteach_loops=int(os.getenv("LEO_MAX_RETEACH_LOOPS", "2")),
        agent_memory=_env_bool("LEO_AGENT_MEMORY", True),
        verbose=_env_bool("LEO_VERBOSE", True),
        serper_enabled=bool(os.getenv("SERPER_API_KEY")),
    )


def build_llm(settings: Settings | None = None, temperature: float | None = None) -> LLM:
    """
    Create the CrewAI LLM. The provider key is read from the environment by
    LiteLLM, so we only verify that it exists and fail early with a clear message.
    """
    settings = settings or get_settings()
    provider = settings.model.split("/", 1)[0].lower()
    key_env = _PROVIDER_KEY_ENV.get(provider)
    if key_env and not os.getenv(key_env):
        raise EnvironmentError(
            f"Missing {key_env}. Copy .env.example to .env and set it "
            f"(LEO_MODEL is '{settings.model}')."
        )
    temp = settings.temperature if temperature is None else temperature

    if provider == "groq":
        model_name = settings.model.split("/", 1)[1]
        return LLM(
            model=f"openai/{model_name}",
            base_url=os.getenv("GROQ_BASE_URL", GROQ_BASE_URL),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temp,
        )

    return LLM(model=settings.model, temperature=temp)