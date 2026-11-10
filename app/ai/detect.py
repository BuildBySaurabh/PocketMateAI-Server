from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detected:
    """Which service an API key belongs to, and how to talk to it."""

    name: str
    kind: str  # "anthropic" or "openai" (any OpenAI-compatible chat API)
    base_url: str
    model: str


# Default model names change over time. Set AI_MODEL in .env to use a different one.
_OPENAI = Detected("OpenAI", "openai", "https://api.openai.com/v1", "gpt-4o-mini")
_GROQ = Detected("Groq", "openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
_OPENROUTER = Detected("OpenRouter", "openai", "https://openrouter.ai/api/v1", "openrouter/auto")
_GEMINI = Detected("Google Gemini", "openai", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash")
_ANTHROPIC = Detected("Anthropic", "anthropic", "https://api.anthropic.com", "claude-sonnet-5-5")


def detect_key(key: str) -> Optional[Detected]:
    """Guess the service from the key's prefix. Returns None for a key we do not recognise."""
    k = key.strip()
    if k.startswith("sk-ant-"):
        return _ANTHROPIC
    if k.startswith("sk-or-"):
        return _OPENROUTER
    if k.startswith("gsk_"):
        return _GROQ
    if k.startswith("AIza"):
        return _GEMINI
    if k.startswith("sk-"):  # sk-proj-..., sk-svcacct-..., classic sk-...
        return _OPENAI
    return None
