from typing import Optional

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.ai.anthropic_provider import AnthropicProvider
from app.ai.base import AIProvider, ChatExtras, ProviderError
from app.ai.detect import detect_key
from app.ai.mock import MockProvider
from app.ai.openai_compat import OpenAICompatProvider
from app.config import Settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.security.auth import token_is_valid
from app.search import SearchClient, SearchError, build_search
from app.security.ratelimit import RateLimiter


MIN_TOKEN_LENGTH = 16
# Values that appeared in examples. They are public, so they must never be accepted.
PLACEHOLDER_TOKENS = {"change-me", "paste-a-long-random-value-here", "your-token-here"}


def build_provider(settings: Settings) -> AIProvider:
    kind = settings.ai_provider
    if kind == "mock":
        return MockProvider()

    # Legacy setting: AI_PROVIDER=anthropic with ANTHROPIC_API_KEY.
    if kind == "anthropic":
        key = settings.ai_api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError("AI_PROVIDER=anthropic needs ANTHROPIC_API_KEY (or AI_API_KEY) to be set.")
        return AnthropicProvider(key, settings.ai_model or settings.anthropic_model)

    # Generic: AI_PROVIDER=auto (or openai / compatible) with AI_API_KEY.
    if kind in ("auto", "openai", "compatible"):
        key = settings.ai_api_key
        if not key:
            raise RuntimeError(f"AI_PROVIDER={kind} needs AI_API_KEY to be set.")
        found = detect_key(key)
        if kind == "auto" and found is None and not settings.ai_base_url:
            raise RuntimeError(
                "I don't recognise that API key. Set AI_BASE_URL (an OpenAI-compatible address, for example "
                "https://api.example.com/v1) and AI_MODEL as well, or set AI_PROVIDER=anthropic."
            )
        if found is not None and found.kind == "anthropic" and kind == "auto" and not settings.ai_base_url:
            return AnthropicProvider(key, settings.ai_model or found.model)
        base_url = settings.ai_base_url or (found.base_url if found and found.kind == "openai" else "")
        model = settings.ai_model or (found.model if found and found.kind == "openai" else "")
        if not base_url or not model:
            raise RuntimeError("Set AI_BASE_URL and AI_MODEL for this key.")
        return OpenAICompatProvider(key, base_url, model)

    raise RuntimeError(f"Unknown AI_PROVIDER '{kind}'. Use 'auto', 'mock', 'anthropic', 'openai' or 'compatible'.")


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "message": message})


def create_app(
    settings: Optional[Settings] = None,
    provider: Optional[AIProvider] = None,
    search: Optional[SearchClient] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    token = settings.app_token
    if not token or token.lower() in PLACEHOLDER_TOKENS or len(token) < MIN_TOKEN_LENGTH:
        raise RuntimeError(
            "APP_TOKEN must be set to a long random value (at least "
            f"{MIN_TOKEN_LENGTH} characters) so strangers cannot use your AI key. Generate one with: "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    provider = provider or build_provider(settings)
    search = search or build_search(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)

    app = FastAPI(title="PocketMate backend", docs_url=None, redoc_url=None)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, __: RequestValidationError):
        return _error(400, "bad_request", "The request was not valid.")

    @app.get("/health")
    async def health():
        return {"status": "ok", "provider": settings.ai_provider, "web_search": search is not None}

    @app.post("/v1/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        request: Request,
        authorization: Optional[str] = Header(default=None),
    ):
        if not token_is_valid(authorization, settings.app_token):
            return _error(401, "unauthorized", "The app token is missing or wrong.")

        client = request.client.host if request.client else "unknown"
        if not limiter.allow(client):
            return _error(429, "rate_limited", "Too many requests. Wait a moment.")

        if body.messages[-1].role != "user":
            return _error(400, "bad_request", "The last message must be from the user.")

        extras = ChatExtras(
            persona=body.persona,
            context=list(body.context or []),
            image=body.image,
        )
        web_used = False
        if body.web_search and search is not None:
            try:
                results = await search.search(body.messages[-1].content)
                extras.web_snippets = [r.as_line() for r in results]
                web_used = bool(results)
            except SearchError:
                pass  # answer without web results rather than failing the whole question

        try:
            if extras.is_empty():
                reply = await provider.reply(body.messages, body.memory)
            else:
                reply = await provider.reply(body.messages, body.memory, extras=extras)
        except ProviderError as e:
            return _error(e.status, e.code, e.message)
        except Exception:
            return _error(500, "server_error", "Something went wrong on the server.")
        return ChatResponse(reply=reply, web_search_used=web_used)

    return app

app = create_app()