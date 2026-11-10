from typing import List, Optional

from app.ai.base import AIProvider, ChatExtras
from app.schemas.chat import ChatMessage


class MockProvider(AIProvider):
    """Needs no API key. Lets you test the app and the server end to end."""

    async def reply(
        self,
        messages: List[ChatMessage],
        memory: Optional[List[str]],
        extras: Optional[ChatExtras] = None,
    ) -> str:
        last = messages[-1].content.strip()
        extra = f" I also remember {len(memory)} thing(s) about you." if memory else ""
        if extras:
            if extras.persona:
                extra += " (persona set)"
            if extras.context:
                extra += f" I was given {len(extras.context)} document passage(s)."
            if extras.web_snippets:
                extra += f" I saw {len(extras.web_snippets)} web result(s)."
            if extras.image:
                extra += " I received a picture."
        return f"(server mock) You said: {last[:80]}.{extra}"
