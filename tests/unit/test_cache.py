"""Unit tests for cache implementations."""

import pytest

from src.infrastructure.cache import create_cache

@pytest.mark.asyncio
async def test_memory_cache_basic_operations():
    """Test basic cache operations."""
    cache = create_cache({"provider": "memory", "ttl": 60})
    
    # Test set and get
    await cache.set("test_key", "test_value")
    value = await cache.get("test_key")
    assert value == "test_value"
    
    # Test delete
    await cache.delete("test_key")
    value = await cache.get("test_key")
    assert value is None
    
    # Test clear
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.clear()
    assert await cache.get("key1") is None
    assert await cache.get("key2") is None

@pytest.mark.asyncio
async def test_memory_cache_ttl():
    """Test cache TTL functionality."""
    cache = create_cache({"provider": "memory", "ttl": 1})
    
    # Set with short TTL
    await cache.set("ttl_key", "ttl_value", ttl=0)  # Expires immediately
    value = await cache.get("ttl_key")
    assert value is None

@pytest.mark.asyncio
async def test_cache_disabled():
    """Test cache when disabled."""
    cache = create_cache({"provider": "memory", "enabled": False})
    
    # Should not store when disabled
    await cache.set("test_key", "test_value")
    value = await cache.get("test_key")
    assert value is None

@pytest.mark.asyncio
async def test_cache_health_check():
    """Test cache health check."""
    cache = create_cache({"provider": "memory"})
    
    is_healthy = await cache.health_check()
    assert is_healthy is True

@pytest.mark.asyncio
async def test_cache_complex_values():
    """Test caching complex data types."""
    cache = create_cache({"provider": "memory"})
    
    # Test dict
    test_dict = {"key": "value", "nested": {"data": [1, 2, 3]}}
    await cache.set("dict_key", test_dict)
    cached_dict = await cache.get("dict_key")
    assert cached_dict == test_dict
    
    # Test list
    test_list = [1, "two", {"three": 3}]
    await cache.set("list_key", test_list)
    cached_list = await cache.get("list_key")
    assert cached_list == test_list

@pytest.mark.asyncio
async def test_cache_size_limit():
    """Test memory cache size limit."""
    cache = create_cache({"provider": "memory", "max_size": 3})
    
    # Add items up to limit
    await cache.set("key1", "value1")
    await cache.set("key2", "value2")
    await cache.set("key3", "value3")
    
    # Adding fourth should evict oldest
    await cache.set("key4", "value4")
    
    # Check that we have the right items
    assert await cache.get("key1") is None  # Should be evicted
    assert await cache.get("key2") == "value2"
    assert await cache.get("key3") == "value3"
    assert await cache.get("key4") == "value4"