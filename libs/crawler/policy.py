from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


@dataclass(frozen=True)
class CrawlPolicy:
    user_agent: str = "same-girl-search/0.1 (+lawful-research; contact-required)"
    min_delay_seconds: float = 2.0
    timeout_seconds: float = 10.0


class RobotsChecker:
    def __init__(self, policy: CrawlPolicy | None = None):
        self.policy = policy or CrawlPolicy()
        self._cache: dict[str, RobotFileParser] = {}

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        parser = self._cache.get(base)
        if parser is None:
            parser = RobotFileParser()
            robots_url = f"{base}/robots.txt"
            async with httpx.AsyncClient(timeout=self.policy.timeout_seconds) as client:
                response = await client.get(robots_url, headers={"User-Agent": self.policy.user_agent})
            parser.parse(response.text.splitlines() if response.status_code < 500 else [])
            self._cache[base] = parser
        return parser.can_fetch(self.policy.user_agent, url)

