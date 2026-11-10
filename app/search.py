import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import httpx

from app.config import Settings

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
MAX_RESULTS = 4
_TAGS = re.compile(r"<[^>]+>")


class SearchError(Exception):
    """The search service failed. The chat carries on without web results."""


@dataclass
class SearchResult:
    title: str
    snippet: str
    url: str

    def as_line(self) -> str:
        return f"- {self.title}: {self.snippet} ({self.url})"


def _clean(text: str, limit: int) -> str:
    return _TAGS.sub("", text or "").strip()[:limit]


class SearchClient(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """Return a few results or raise SearchError."""


class BraveSearch(SearchClient):
    def __init__(self, api_key: str, client: Optional[httpx.AsyncClient] = None):
        self._api_key = api_key
        self._client = client or httpx.AsyncClient(timeout=8.0)

    async def search(self, query: str) -> List[SearchResult]:
        try:
            resp = await self._client.get(
                BRAVE_URL,
                params={"q": query[:300], "count": MAX_RESULTS},
                headers={"X-Subscription-Token": self._api_key, "Accept": "application/json"},
            )
        except httpx.HTTPError:
            raise SearchError("search unreachable")
        if resp.status_code != 200:
            raise SearchError(f"search returned {resp.status_code}")
        try:
            items = resp.json()["web"]["results"]
            results = [
                SearchResult(_clean(i.get("title", ""), 120), _clean(i.get("description", ""), 300), i.get("url", ""))
                for i in items[:MAX_RESULTS]
                if i.get("url")
            ]
        except (ValueError, KeyError, TypeError, AttributeError):
            raise SearchError("search answer unreadable")
        return results


def build_search(settings: Settings) -> Optional[SearchClient]:
    if settings.search_provider == "brave" and settings.search_api_key:
        return BraveSearch(settings.search_api_key)
    return None
