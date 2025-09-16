"""Caching system for MCP server."""

import asyncio
import hashlib
import json
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import heapq

from ..utils.logger import get_logger

logger = get_logger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive strategy based on usage patterns


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    size: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if not self.ttl:
            return False
        return datetime.utcnow() > self.created_at + self.ttl

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class MCPCache:
    """Intelligent caching system for MCP server."""

    def __init__(
        self,
        max_size_mb: int = 1024,
        default_ttl: timedelta = timedelta(minutes=15),
        strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
        persistence_path: Optional[str] = None,
    ):
        """Initialize cache."""
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.persistence_path = persistence_path

        self._cache: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._lock = asyncio.Lock()

        # LFU tracking
        self._frequency_heap: List[Tuple[int, str]] = []

        # Load persisted cache if available
        if persistence_path:
            self._load_cache()

    def generate_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key from request."""
        # Create deterministic key from request
        key_data = {
            "method": request.get("method"),
            "params": request.get("params"),
            # Don't include context that changes per request
            "context_keys": sorted(request.get("context", {}).keys()),
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if not entry:
                self._misses += 1
                return None

            # Check expiration
            if entry.is_expired():
                await self._remove_entry(key)
                self._misses += 1
                return None

            # Update access metadata
            entry.touch()
            self._hits += 1

            # Update frequency heap for LFU
            if self.strategy == CacheStrategy.LFU:
                heapq.heappush(self._frequency_heap, (entry.access_count, key))

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Set value in cache."""
        try:
            # Serialize to get size
            serialized = pickle.dumps(value)
            size = len(serialized)

            # Check if value is too large
            if size > self.max_size_bytes:
                logger.warning(f"Value too large for cache: {size} bytes")
                return False

            async with self._lock:
                # Make room if needed
                await self._ensure_space(size)

                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size=size,
                    ttl=ttl or self.default_ttl,
                    metadata=metadata or {},
                )

                # Remove old entry if exists
                if key in self._cache:
                    await self._remove_entry(key)

                # Add new entry
                self._cache[key] = entry
                self._current_size += size

                # Update frequency heap for LFU
                if self.strategy == CacheStrategy.LFU:
                    heapq.heappush(self._frequency_heap, (0, key))

                return True

        except Exception as e:
            logger.error(f"Error setting cache entry: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        async with self._lock:
            return await self._remove_entry(key)

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._current_size = 0
            self._frequency_heap.clear()
            logger.info("Cache cleared")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            # Calculate average entry age
            ages = []
            now = datetime.utcnow()
            for entry in self._cache.values():
                age = (now - entry.created_at).total_seconds()
                ages.append(age)

            avg_age = sum(ages) / len(ages) if ages else 0

            return {
                "entries": len(self._cache),
                "size_mb": self._current_size / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "avg_entry_age_seconds": avg_age,
                "strategy": self.strategy.value,
            }

    async def _ensure_space(self, required_size: int) -> None:
        """Ensure there's enough space for new entry."""
        while self._current_size + required_size > self.max_size_bytes:
            # Evict based on strategy
            evicted = await self._evict_entry()
            if not evicted:
                break

    async def _evict_entry(self) -> bool:
        """Evict an entry based on current strategy."""
        if not self._cache:
            return False

        key_to_evict = None

        if self.strategy == CacheStrategy.LRU:
            # Find least recently used
            oldest_access = datetime.utcnow()
            for key, entry in self._cache.items():
                if entry.last_accessed < oldest_access:
                    oldest_access = entry.last_accessed
                    key_to_evict = key

        elif self.strategy == CacheStrategy.LFU:
            # Use frequency heap
            while self._frequency_heap:
                freq, key = heapq.heappop(self._frequency_heap)
                if key in self._cache:
                    key_to_evict = key
                    break

        elif self.strategy == CacheStrategy.TTL:
            # Evict expired entries first
            for key, entry in self._cache.items():
                if entry.is_expired():
                    key_to_evict = key
                    break

            # If no expired, use LRU
            if not key_to_evict:
                return await self._evict_lru()

        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on usage patterns
            key_to_evict = await self._adaptive_evict()

        if key_to_evict:
            await self._remove_entry(key_to_evict)
            self._evictions += 1
            return True

        return False

    async def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self._cache:
            return False

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        await self._remove_entry(oldest_key)
        self._evictions += 1
        return True

    async def _adaptive_evict(self) -> Optional[str]:
        """Adaptive eviction based on multiple factors."""
        if not self._cache:
            return None

        # Score each entry (lower is better for eviction)
        scores = {}
        now = datetime.utcnow()

        for key, entry in self._cache.items():
            # Factors:
            # 1. Recency (time since last access)
            recency_score = (now - entry.last_accessed).total_seconds() / 3600  # hours

            # 2. Frequency (inverse of access count)
            frequency_score = 1.0 / (entry.access_count + 1)

            # 3. Size (larger entries get higher eviction score)
            size_score = entry.size / self.max_size_bytes

            # 4. Age (older entries get higher score)
            age_score = (now - entry.created_at).total_seconds() / 3600  # hours

            # 5. Expiration (expired or nearly expired get highest score)
            if entry.ttl:
                time_to_expire = (entry.created_at + entry.ttl - now).total_seconds()
                expiry_score = (
                    10.0 if time_to_expire <= 0 else 1.0 / (time_to_expire + 1)
                )
            else:
                expiry_score = 0.0

            # Weighted combination
            total_score = (
                recency_score * 0.3
                + frequency_score * 0.2
                + size_score * 0.2
                + age_score * 0.2
                + expiry_score * 0.1
            )

            scores[key] = total_score

        # Select entry with highest score for eviction
        return max(scores.keys(), key=lambda k: scores[k])

    async def _remove_entry(self, key: str) -> bool:
        """Remove entry from cache."""
        if key not in self._cache:
            return False

        entry = self._cache[key]
        self._current_size -= entry.size
        del self._cache[key]
        return True

    def _save_cache(self) -> None:
        """Persist cache to disk."""
        if not self.persistence_path:
            return

        try:
            # Save cache metadata and entries
            cache_data = {
                "entries": self._cache,
                "stats": {
                    "hits": self._hits,
                    "misses": self._misses,
                    "evictions": self._evictions,
                },
            }

            with open(self.persistence_path, "wb") as f:
                pickle.dump(cache_data, f)

            logger.info(f"Cache persisted to {self.persistence_path}")

        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def _load_cache(self) -> None:
        """Load cache from disk."""
        if not self.persistence_path:
            return

        try:
            with open(self.persistence_path, "rb") as f:
                cache_data = pickle.load(f)

            # Filter out expired entries
            now = datetime.utcnow()
            valid_entries = {}
            total_size = 0

            for key, entry in cache_data["entries"].items():
                if not entry.is_expired():
                    valid_entries[key] = entry
                    total_size += entry.size

            self._cache = valid_entries
            self._current_size = total_size

            # Restore stats
            stats = cache_data.get("stats", {})
            self._hits = stats.get("hits", 0)
            self._misses = stats.get("misses", 0)
            self._evictions = stats.get("evictions", 0)

            logger.info(
                f"Cache loaded from {self.persistence_path}: {len(self._cache)} entries"
            )

        except FileNotFoundError:
            logger.info("No cache file found, starting with empty cache")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")


class CacheWarmer:
    """Proactively warm cache with frequently used entries."""

    def __init__(self, cache: MCPCache):
        """Initialize cache warmer."""
        self.cache = cache
        self._patterns: Dict[str, int] = {}  # Request pattern -> frequency

    async def analyze_patterns(self, requests: List[Dict[str, Any]]) -> None:
        """Analyze request patterns for cache warming."""
        for request in requests:
            pattern = self._extract_pattern(request)
            self._patterns[pattern] = self._patterns.get(pattern, 0) + 1

    def _extract_pattern(self, request: Dict[str, Any]) -> str:
        """Extract pattern from request for analysis."""
        # Simple pattern extraction - can be made more sophisticated
        method = request.get("method", "")
        tool = request.get("params", {}).get("tool", "")
        return f"{method}:{tool}"

    async def warm_cache(self, executor) -> None:
        """Warm cache with likely requests."""
        # Sort patterns by frequency
        sorted_patterns = sorted(
            self._patterns.items(), key=lambda x: x[1], reverse=True
        )

        # Warm top patterns
        for pattern, frequency in sorted_patterns[:10]:
            if frequency > 5:  # Only warm frequently used patterns
                # Generate synthetic request based on pattern
                method, tool = pattern.split(":")
                synthetic_request = {"method": method, "params": {"tool": tool}}

                # Execute and cache
                try:
                    result = await executor(synthetic_request)
                    key = self.cache.generate_key(synthetic_request)
                    await self.cache.set(key, result)
                    logger.info(f"Warmed cache for pattern: {pattern}")
                except Exception as e:
                    logger.error(f"Error warming cache for pattern {pattern}: {e}")
