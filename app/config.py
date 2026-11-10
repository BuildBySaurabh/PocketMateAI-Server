import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_env_file(path: Path) -> int:
    """Reads KEY=VALUE lines from a .env file into the environment.

    - Lines starting with # and blank lines are ignored.
    - Quotes around a value are removed.
    - A variable that is already set in the environment is never overridden, so real
      environment variables (for example on a hosting service) always win over the file.
    Returns how many variables were loaded. A missing file is fine.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError):
        return 0

    loaded = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def _default_provider() -> str:
    """AI_PROVIDER if set; otherwise "auto" when a generic AI_API_KEY is present; otherwise "mock"."""
    explicit = os.getenv("AI_PROVIDER", "").strip().lower()
    if explicit:
        return explicit
    return "auto" if os.getenv("AI_API_KEY", "").strip() else "mock"


@dataclass(frozen=True)
class Settings:
    ai_provider: str
    anthropic_api_key: str
    anthropic_model: str
    app_token: str
    rate_limit_per_minute: int
    # Optional web search for answers (the user must also switch it on in the app). "none" or "brave".
    search_provider: str = "none"
    search_api_key: str = ""
    # One generic key for any supported service. The provider is detected from the key unless AI_PROVIDER says otherwise.
    ai_api_key: str = ""
    ai_base_url: str = ""
    ai_model: str = ""

    @staticmethod
    def from_env(env_file: Optional[Path] = Path(".env")) -> "Settings":
        if env_file is not None:
            load_env_file(env_file)
        return Settings(
            ai_provider=_default_provider(),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5").strip(),
            app_token=os.getenv("APP_TOKEN", "").strip(),
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "20") or "20"),
            search_provider=os.getenv("SEARCH_PROVIDER", "none").strip().lower(),
            search_api_key=os.getenv("SEARCH_API_KEY", "").strip(),
            ai_api_key=os.getenv("AI_API_KEY", "").strip(),
            ai_base_url=os.getenv("AI_BASE_URL", "").strip().rstrip("/"),
            ai_model=os.getenv("AI_MODEL", "").strip(),
        )
