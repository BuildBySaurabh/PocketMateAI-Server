from typing import List, Optional

import logging

import httpx

from app.ai.base import AIProvider, ChatExtras, ProviderError, build_system_prompt, drop_leading_assistant_messages
from app.schemas.chat import ChatMessage


log = logging.getLogger("pocketmate.ai")


class OpenAICompatProvider(AIProvider):
    """Talks to any service that offers the OpenAI-style `/chat/completions` API: OpenAI, Groq, OpenRouter,
    Google Gemini (its OpenAI-compatible endpoint), Together, Mistral, DeepSeek, local servers, and more."""

    def __init__(self, api_key: str, base_url: str, model: str, client: Optional[httpx.AsyncClient] = None):
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def reply(
        self,
        messages: List[ChatMessage],
        memory: Optional[List[str]],
        extras: Optional[ChatExtras] = None,
    ) -> str:
        api_messages = [{"role": "system", "content": build_system_prompt(memory, extras)}]
        api_messages += [{"role": m.role, "content": m.content} for m in drop_leading_assistant_messages(messages)]
        if extras and extras.image and api_messages[-1]["role"] == "user":
            api_messages[-1]["content"] = [
                {"type": "text", "text": api_messages[-1]["content"]},
                {"type": "image_url", "image_url": {"url": f"data:{extras.image.media_type};base64,{extras.image.data}"}},
            ]
        body = {"model": self._model, "messages": api_messages, "max_tokens": 400}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            resp = await self._client.post(self._url, json=body, headers=headers)
        except httpx.TimeoutException:
            raise ProviderError("server_error", "The AI service took too long to answer.", 504)
        except httpx.HTTPError:
            raise ProviderError("server_error", "Could not reach the AI service.", 502)

        if resp.status_code != 200:
            # Shown in the server log only (never sent to the app) so you can see why the AI service refused.
            log.warning("AI service answered %s: %s", resp.status_code, resp.text[:300].replace("\n", " "))
        if resp.status_code == 429:
            raise ProviderError("rate_limited", "The AI service is busy. Try again shortly.", 429)
        if resp.status_code in (404, 529) or resp.status_code >= 500:
            raise ProviderError("model_unavailable", "The AI model is unavailable right now.", 503)
        if resp.status_code in (401, 403):
            raise ProviderError("server_error", "The server is not set up correctly.", 500)
        if resp.status_code != 200:
            raise ProviderError("server_error", "The AI service returned an error.", 502)
        try:
            text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ProviderError("server_error", "The AI service sent an unreadable answer.", 502)
        if not text:
            raise ProviderError("server_error", "The AI service sent an empty answer.", 502)
        return text