"""Cache implementations."""

from typing import Dict, Any

from src.core.interfaces import ICache
from src.core.exceptions import CacheError
from .redis_cache import RedisCache
from .memory_cache import MemoryCache


def create_cache(config: Dict[str, Any]) -> ICache:
    """Factory function to create cache instances."""
    provider = config.get("provider", "memory")

    if provider == "redis":
        return RedisCache(config)
    elif provider == "memory":
        return MemoryCache(config)
    else:
        raise CacheError(f"Unknown cache provider: {provider}")


__all__ = ["create_cache", "RedisCache", "MemoryCache"]
