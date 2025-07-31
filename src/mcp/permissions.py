"""Permission system for MCP server."""

import asyncio
import secrets
import hashlib
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PermissionLevel(Enum):
    """Permission levels for MCP clients."""

    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class RateLimit:
    """Rate limit configuration."""

    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    concurrent_requests: int = 10
    burst_size: int = 20


@dataclass
class MCPClient:
    """MCP client configuration."""

    id: str
    name: str
    api_key: str
    allowed_tools: List[str] = field(default_factory=list)
    permission_level: PermissionLevel = PermissionLevel.EXECUTE
    rate_limits: RateLimit = field(default_factory=RateLimit)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_seen: Optional[datetime] = None
    enabled: bool = True


class PermissionManager:
    """Manages permissions and access control for MCP clients."""

    def __init__(self, storage_backend=None):
        """Initialize permission manager."""
        self._clients: Dict[str, MCPClient] = {}
        self._api_key_index: Dict[str, str] = {}  # api_key -> client_id
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self.storage_backend = storage_backend
        self._load_clients()

    def _load_clients(self) -> None:
        """Load clients from storage."""
        if self.storage_backend:
            # TODO: Implement storage backend loading
            pass

    async def register_client(
        self,
        name: str,
        allowed_tools: Optional[List[str]] = None,
        permission_level: PermissionLevel = PermissionLevel.EXECUTE,
        rate_limits: Optional[RateLimit] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MCPClient:
        """Register a new MCP client."""
        client_id = self._generate_client_id()
        api_key = self._generate_api_key()

        client = MCPClient(
            id=client_id,
            name=name,
            api_key=api_key,
            allowed_tools=allowed_tools or [],
            permission_level=permission_level,
            rate_limits=rate_limits or RateLimit(),
            metadata=metadata or {},
        )

        self._clients[client_id] = client
        self._api_key_index[api_key] = client_id
        self._rate_limiters[client_id] = RateLimiter(client.rate_limits)

        # Save to storage
        if self.storage_backend:
            await self._save_client(client)

        logger.info(f"Registered new client: {client_id} ({name})")
        return client

    async def get_client(self, client_id: str) -> Optional[MCPClient]:
        """Get client by ID."""
        return self._clients.get(client_id)

    async def get_client_by_api_key(self, api_key: str) -> Optional[MCPClient]:
        """Get client by API key."""
        client_id = self._api_key_index.get(api_key)
        if client_id:
            return self._clients.get(client_id)
        return None

    async def update_client(
        self, client_id: str, updates: Dict[str, Any]
    ) -> Optional[MCPClient]:
        """Update client configuration."""
        client = self._clients.get(client_id)
        if not client:
            return None

        # Update fields
        for key, value in updates.items():
            if hasattr(client, key) and key not in ["id", "api_key", "created_at"]:
                setattr(client, key, value)

        # Update rate limiter if needed
        if "rate_limits" in updates:
            self._rate_limiters[client_id] = RateLimiter(client.rate_limits)

        # Save to storage
        if self.storage_backend:
            await self._save_client(client)

        logger.info(f"Updated client {client_id}")
        return client

    async def delete_client(self, client_id: str) -> bool:
        """Delete a client."""
        client = self._clients.get(client_id)
        if not client:
            return False

        # Remove from indices
        del self._clients[client_id]
        del self._api_key_index[client.api_key]
        if client_id in self._rate_limiters:
            del self._rate_limiters[client_id]

        # Remove from storage
        if self.storage_backend:
            await self._delete_client(client_id)

        logger.info(f"Deleted client {client_id}")
        return True

    async def can_access_tool(self, client: MCPClient, tool_name: str) -> bool:
        """Check if client can access a specific tool."""
        if not client.enabled:
            return False

        # Admin can access everything
        if client.permission_level == PermissionLevel.ADMIN:
            return True

        # Check allowed tools
        if client.allowed_tools:
            # Empty list means no restrictions
            if not client.allowed_tools:
                return True
            return tool_name in client.allowed_tools

        return True

    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        rate_limiter = self._rate_limiters.get(client_id)
        if not rate_limiter:
            return True

        return await rate_limiter.check()

    async def record_request(self, client_id: str) -> None:
        """Record a request for rate limiting."""
        # Update last seen
        if client_id in self._clients:
            self._clients[client_id].last_seen = datetime.utcnow()

        # Record in rate limiter
        rate_limiter = self._rate_limiters.get(client_id)
        if rate_limiter:
            await rate_limiter.record()

    def _generate_client_id(self) -> str:
        """Generate unique client ID."""
        return f"mcp_client_{secrets.token_hex(8)}"

    def _generate_api_key(self) -> str:
        """Generate secure API key."""
        return f"mcp_{secrets.token_urlsafe(32)}"

    async def _save_client(self, client: MCPClient) -> None:
        """Save client to storage."""
        # TODO: Implement storage backend
        pass

    async def _delete_client(self, client_id: str) -> None:
        """Delete client from storage."""
        # TODO: Implement storage backend
        pass


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, config: RateLimit):
        """Initialize rate limiter."""
        self.config = config
        self._minute_requests: List[datetime] = []
        self._hour_requests: List[datetime] = []
        self._concurrent_requests = 0
        self._lock = asyncio.Lock()

    async def check(self) -> bool:
        """Check if request is allowed."""
        async with self._lock:
            now = datetime.utcnow()
            self._cleanup_old_requests(now)

            # Check concurrent requests
            if self._concurrent_requests >= self.config.concurrent_requests:
                return False

            # Check minute limit
            if len(self._minute_requests) >= self.config.requests_per_minute:
                return False

            # Check hour limit
            if len(self._hour_requests) >= self.config.requests_per_hour:
                return False

            return True

    async def record(self) -> None:
        """Record a request."""
        async with self._lock:
            now = datetime.utcnow()
            self._minute_requests.append(now)
            self._hour_requests.append(now)
            self._concurrent_requests += 1

    async def release(self) -> None:
        """Release a concurrent request slot."""
        async with self._lock:
            self._concurrent_requests = max(0, self._concurrent_requests - 1)

    def _cleanup_old_requests(self, now: datetime) -> None:
        """Remove old requests from tracking."""
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        self._minute_requests = [
            req for req in self._minute_requests if req > minute_ago
        ]
        self._hour_requests = [req for req in self._hour_requests if req > hour_ago]


class PermissionDeniedError(Exception):
    """Raised when permission is denied."""

    pass


class RateLimitExceededError(Exception):
    """Raised when rate limit is exceeded."""

    pass


# Permission decorators for tool methods
def require_permission(level: PermissionLevel):
    """Decorator to require specific permission level."""

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Extract client from context
            context = kwargs.get("context", {})
            client = context.get("client")

            if not client:
                raise PermissionDeniedError("No client context provided")

            if client.permission_level.value < level.value:
                raise PermissionDeniedError(
                    f"Client requires {level.value} permission level"
                )

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def rate_limited(func):
    """Decorator to apply rate limiting."""

    async def wrapper(self, *args, **kwargs):
        # Extract client from context
        context = kwargs.get("context", {})
        client_id = context.get("client_id")

        if client_id:
            # Check rate limit
            permission_manager = context.get("permission_manager")
            if permission_manager:
                allowed = await permission_manager.check_rate_limit(client_id)
                if not allowed:
                    raise RateLimitExceededError("Rate limit exceeded")

                # Record request
                await permission_manager.record_request(client_id)

                # Execute function
                try:
                    result = await func(self, *args, **kwargs)
                    return result
                finally:
                    # Release concurrent slot
                    rate_limiter = permission_manager._rate_limiters.get(client_id)
                    if rate_limiter:
                        await rate_limiter.release()

        return await func(self, *args, **kwargs)

    return wrapper
