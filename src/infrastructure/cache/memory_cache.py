"""In-memory cache implementation."""

import time
from typing import Any, Optional, Dict, Tuple

from .base import BaseCache


class MemoryCache(BaseCache):
    """Simple in-memory cache implementation."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = config.get("max_size", 1000)

    async def _get_impl(self, key: str) -> Optional[Any]:
        """Get from memory."""
        if key in self.cache:
            value, expires_at = self.cache[key]
            if expires_at > time.time():
                return value
            else:
                del self.cache[key]
        return None

    async def _set_impl(self, key: str, value: Any, ttl: int) -> None:
        """Set in memory."""
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

        expires_at = time.time() + ttl
        self.cache[key] = (value, expires_at)

    async def _delete_impl(self, key: str) -> None:
        """Delete from memory."""
        self.cache.pop(key, None)

    async def _clear_impl(self) -> None:
        """Clear memory cache."""
        self.cache.clear()
