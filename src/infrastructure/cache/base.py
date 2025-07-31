"""Base cache implementation."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.core.interfaces import ICache
from src.core.exceptions import CacheError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseCache(ABC, ICache):
    """Base class for cache implementations."""
    
    def __init__(self, config: dict):
        self.config = config
        self.ttl = config.get('ttl', 3600)
        self.enabled = config.get('enabled', True)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled:
            return None
            
        try:
            return await self._get_impl(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
            
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        if not self.enabled:
            return
            
        try:
            await self._set_impl(key, value, ttl or self.ttl)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        if not self.enabled:
            return
            
        try:
            await self._delete_impl(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            
    async def clear(self) -> None:
        """Clear all cache entries."""
        if not self.enabled:
            return
            
        try:
            await self._clear_impl()
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            
    async def health_check(self) -> bool:
        """Check if cache is healthy."""
        try:
            await self.set("health_check", "ok", 10)
            value = await self.get("health_check")
            return value == "ok"
        except Exception:
            return False
    
    @abstractmethod
    async def _get_impl(self, key: str) -> Optional[Any]:
        """Get implementation."""
        pass
    
    @abstractmethod
    async def _set_impl(self, key: str, value: Any, ttl: int) -> None:
        """Set implementation."""
        pass
    
    @abstractmethod
    async def _delete_impl(self, key: str) -> None:
        """Delete implementation."""
        pass
    
    @abstractmethod
    async def _clear_impl(self) -> None:
        """Clear implementation."""
        pass