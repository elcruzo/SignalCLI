"""API client for SignalCLI."""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from src.core.exceptions import SignalCLIError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class APIClient:
    """Async HTTP client for SignalCLI API."""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    async def _ensure_session(self):
        """Ensure session exists."""
        if self.session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            
    async def query(
        self,
        query: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send query to API.
        
        Args:
            query: User query
            schema: Optional JSON schema
            max_tokens: Maximum tokens
            temperature: LLM temperature
            
        Returns:
            API response
        """
        await self._ensure_session()
        
        # Prepare request
        payload = {
            "query": query,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }
        
        if schema:
            payload["schema"] = schema
            
        # Make request
        try:
            url = urljoin(self.base_url, "/query")
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise SignalCLIError(
                        f"API error ({response.status}): {error_text}"
                    )
                    
        except aiohttp.ClientError as e:
            logger.error(f"HTTP client error: {e}")
            raise SignalCLIError(f"Connection error: {e}")
        except asyncio.TimeoutError:
            raise SignalCLIError("Request timeout")
            
    async def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        await self._ensure_session()
        
        try:
            url = urljoin(self.base_url, "/health")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise SignalCLIError(f"Health check failed: {response.status}")
                    
        except Exception as e:
            logger.error(f"Health check error: {e}")
            raise SignalCLIError(f"Health check failed: {e}")
            
    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None