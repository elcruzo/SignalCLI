"""Redis cache implementation."""

import json
import redis.asyncio as redis
from typing import Any, Optional

from .base import BaseCache
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisCache(BaseCache):
    """Cache implementation using Redis."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6379)
        self.db = config.get("db", 0)
        self.client = None

    async def _ensure_connected(self):
        """Ensure Redis connection."""
        if self.client is None:
            self.client = redis.Redis(
                host=self.host, port=self.port, db=self.db, decode_responses=True
            )

    async def _get_impl(self, key: str) -> Optional[Any]:
        """Get from Redis."""
        await self._ensure_connected()
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def _set_impl(self, key: str, value: Any, ttl: int) -> None:
        """Set in Redis."""
        await self._ensure_connected()
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self.client.setex(key, ttl, value)

    async def _delete_impl(self, key: str) -> None:
        """Delete from Redis."""
        await self._ensure_connected()
        await self.client.delete(key)

    async def _clear_impl(self) -> None:
        """Clear Redis cache."""
        await self._ensure_connected()
        await self.client.flushdb()
