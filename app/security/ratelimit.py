import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict


class RateLimiter:
    """Sliding one-minute window per key, in memory. Fine for one server process;
    use Redis or similar if you run several."""

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic):
        self._limit = per_minute
        self._clock = clock
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = self._clock()
        hits = self._hits[key]
        while hits and now - hits[0] >= 60.0:
            hits.popleft()
        if len(hits) >= self._limit:
            return False
        hits.append(now)
        return True
