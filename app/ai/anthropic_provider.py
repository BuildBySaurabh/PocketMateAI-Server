from typing import List, Optional

import httpx

from app.ai.base import AIProvider, ChatExtras, ProviderError, build_system_prompt, drop_leading_assistant_messages
from app.schemas.chat import ChatMessage

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str, client: Optional[httpx.AsyncClient] = None):
        self._api_key = api_key
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=30.0)

    async def reply(
        self,
        messages: List[ChatMessage],
        memory: Optional[List[str]],
        extras: Optional[ChatExtras] = None,
    ) -> str:
        api_messages = [
            {"role": m.role, "content": m.content} for m in drop_leading_assistant_messages(messages)
        ]
        if extras and extras.image and api_messages and api_messages[-1]["role"] == "user":
            api_messages[-1]["content"] = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": extras.image.media_type,
                        "data": extras.image.data,
                    },
                },
                {"type": "text", "text": api_messages[-1]["content"]},
            ]
        body = {
            "model": self._model,
            "max_tokens": 400,
            "system": build_system_prompt(memory, extras),
            "messages": api_messages,
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        try:
            resp = await self._client.post(API_URL, json=body, headers=headers)
        except httpx.TimeoutException:
            raise ProviderError("server_error", "The AI service took too long to answer.", 504)
        except httpx.HTTPError:
            raise ProviderError("server_error", "Could not reach the AI service.", 502)

        if resp.status_code == 429:
            raise ProviderError("rate_limited", "The AI service is busy. Try again shortly.", 429)
        if resp.status_code in (404, 529) or resp.status_code >= 500:
            raise ProviderError("model_unavailable", "The AI model is unavailable right now.", 503)
        if resp.status_code in (401, 403):
            # Our key is wrong. Do not leak details to the phone; the operator must fix the server.
            raise ProviderError("server_error", "The server is not set up correctly.", 500)
        if resp.status_code != 200:
            raise ProviderError("server_error", "The AI service returned an error.", 502)

        try:
            blocks = resp.json()["content"]
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ProviderError("server_error", "The AI service sent an unreadable answer.", 502)
        if not text:
            raise ProviderError("server_error", "The AI service sent an empty answer.", 502)
        return text
