"""MCP integration for main API."""

import aiohttp
import asyncio
import json
from typing import Dict, Any, Optional, List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MCPIntegration:
    """Handles integration with MCP server."""

    def __init__(self, mcp_server_url: str, api_key: Optional[str] = None):
        """Initialize MCP integration."""
        self.base_url = mcp_server_url.rstrip("/")
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self._tools_cache: Optional[Dict[str, Any]] = None
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = 0

    async def initialize(self):
        """Initialize the MCP integration."""
        self.session = aiohttp.ClientSession(
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(total=30),
        )
        # Pre-fetch tools
        await self.list_tools()
        logger.info(f"MCP integration initialized with {self.base_url}")

    async def close(self):
        """Close the MCP integration."""
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def health_check(self) -> str:
        """Check MCP server health."""
        try:
            async with self.session.get(f"{self.base_url}/mcp/v1/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status", "healthy")
                return "unhealthy"
        except Exception as e:
            logger.error(f"MCP health check failed: {e}")
            return "error"

    async def list_tools(self, force_refresh: bool = False) -> Dict[str, Any]:
        """List available MCP tools."""
        # Check cache
        if (
            not force_refresh
            and self._tools_cache
            and (asyncio.get_event_loop().time() - self._last_cache_update) < self._cache_ttl
        ):
            return self._tools_cache

        try:
            async with self.session.get(f"{self.base_url}/mcp/v1/tools") as response:
                if response.status == 200:
                    self._tools_cache = await response.json()
                    self._last_cache_update = asyncio.get_event_loop().time()
                    return self._tools_cache
                else:
                    error_data = await response.text()
                    raise Exception(f"Failed to list tools: {error_data}")
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            raise

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Execute an MCP tool."""
        request_data = {
            "method": "execute",
            "params": {"tool": tool_name, **arguments},
            "context": context or {},
            "stream": stream,
        }

        try:
            async with self.session.post(
                f"{self.base_url}/mcp/v1/execute", json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.text()
                    raise Exception(f"Tool execution failed: {error_data}")
        except Exception as e:
            logger.error(f"Failed to execute MCP tool {tool_name}: {e}")
            raise

    async def execute_chain(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a chain of MCP tools."""
        try:
            async with self.session.post(
                f"{self.base_url}/mcp/v1/execute/chain", json=tool_calls
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.text()
                    raise Exception(f"Chain execution failed: {error_data}")
        except Exception as e:
            logger.error(f"Failed to execute MCP chain: {e}")
            raise

    async def get_tool_info(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        try:
            async with self.session.get(
                f"{self.base_url}/mcp/v1/tools/{tool_name}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_data = await response.text()
                    raise Exception(f"Failed to get tool info: {error_data}")
        except Exception as e:
            logger.error(f"Failed to get MCP tool info for {tool_name}: {e}")
            raise

    async def stream_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ):
        """Stream results from an MCP tool."""
        # First initiate the stream
        result = await self.execute_tool(tool_name, arguments, context, stream=True)
        stream_id = result["result"]["stream_id"]

        # Connect to SSE endpoint
        async with self.session.get(
            f"{self.base_url}/mcp/v1/stream/sse?stream_id={stream_id}"
        ) as response:
            async for line in response.content:
                if line.startswith(b"data: "):
                    try:
                        data = json.loads(line[6:].decode())
                        yield data
                    except json.JSONDecodeError:
                        continue


class MCPToolProxy:
    """Proxy for executing MCP tools with additional features."""

    def __init__(self, mcp_integration: MCPIntegration):
        """Initialize tool proxy."""
        self.mcp = mcp_integration
        self._execution_history: List[Dict[str, Any]] = []

    async def execute_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        max_retries: int = 3,
        backoff_factor: float = 2.0,
    ) -> Dict[str, Any]:
        """Execute tool with retry logic."""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await self.mcp.execute_tool(tool_name, arguments)
                self._record_execution(tool_name, arguments, result, success=True)
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = backoff_factor ** attempt
                    logger.warning(
                        f"Tool execution failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self._record_execution(tool_name, arguments, None, success=False, error=str(e))
                    raise

    async def execute_with_fallback(
        self,
        primary_tool: str,
        fallback_tool: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute tool with fallback option."""
        try:
            return await self.mcp.execute_tool(primary_tool, arguments)
        except Exception as e:
            logger.warning(f"Primary tool {primary_tool} failed, using fallback: {e}")
            return await self.mcp.execute_tool(fallback_tool, arguments)

    def _record_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: Optional[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
    ):
        """Record tool execution for history."""
        self._execution_history.append({
            "timestamp": asyncio.get_event_loop().time(),
            "tool": tool_name,
            "arguments": arguments,
            "result": result,
            "success": success,
            "error": error,
        })
        
        # Keep only last 100 executions
        if len(self._execution_history) > 100:
            self._execution_history = self._execution_history[-100:]

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self._execution_history:
            return {"total": 0, "success": 0, "failure": 0, "success_rate": 0.0}
        
        total = len(self._execution_history)
        success = sum(1 for e in self._execution_history if e["success"])
        failure = total - success
        
        return {
            "total": total,
            "success": success,
            "failure": failure,
            "success_rate": success / total if total > 0 else 0.0,
            "tools_used": list(set(e["tool"] for e in self._execution_history)),
        }
