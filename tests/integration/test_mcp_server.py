"""Integration tests for MCP server."""

import pytest
import asyncio
import json
from typing import Dict, Any
from unittest.mock import AsyncMock, Mock, patch

from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.mcp.server import SignalCLIMCPServer
from src.mcp.tools import ToolRegistry, Tool, ToolCapability
from src.mcp.router import ContextAwareRouter
from src.mcp.permissions import PermissionManager
from src.mcp.cache import MCPCache
from src.mcp.protocol import MCPRequest, MCPResponse
from src.core.interfaces import Document


@pytest.fixture
async def mock_tool_registry():
    """Mock tool registry with test tools."""
    registry = ToolRegistry()

    # Test RAG tool
    rag_tool = Tool(
        name="rag_query",
        description="Perform RAG query",
        capabilities=[ToolCapability.QUERY],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}, "sources": {"type": "array"}},
        },
        supports_streaming=True,
    )

    async def mock_rag_execute(
        params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "answer": f"Answer to: {params.get('query')}",
            "sources": [
                {
                    "content": "Mock source content",
                    "source": "test_doc.txt",
                    "similarity_score": 0.95,
                }
            ],
        }

    rag_tool.execute = mock_rag_execute
    registry.register_tool(rag_tool)

    # Test JSON tool
    json_tool = Tool(
        name="json_query",
        description="Query with structured JSON output",
        capabilities=[ToolCapability.QUERY, ToolCapability.JSON_OUTPUT],
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "schema": {"type": "object"}},
            "required": ["query", "schema"],
        },
        output_schema={"type": "object"},
        supports_streaming=False,
    )

    async def mock_json_execute(
        params: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        schema = params.get("schema", {})
        return {
            "result": f"Structured answer to: {params.get('query')}",
            "metadata": {"schema_provided": True},
        }

    json_tool.execute = mock_json_execute
    registry.register_tool(json_tool)

    return registry


@pytest.fixture
async def mock_router():
    """Mock context-aware router."""
    router = ContextAwareRouter()

    async def mock_route(request: MCPRequest) -> str:
        # Simple routing based on request content
        params = request.params or {}
        if "schema" in params:
            return "json_query"
        return "rag_query"

    router.route = mock_route
    return router


@pytest.fixture
async def mock_permission_manager():
    """Mock permission manager."""
    manager = PermissionManager()

    # Mock client
    mock_client = Mock()
    mock_client.id = "test_client"
    mock_client.allowed_tools = ["rag_query", "json_query"]

    async def mock_get_client(client_id: str):
        return mock_client if client_id == "test_client" else None

    async def mock_can_access_tool(client, tool_name: str):
        return tool_name in client.allowed_tools if client else True

    manager.get_client = mock_get_client
    manager.can_access_tool = mock_can_access_tool

    return manager


@pytest.fixture
async def mock_cache():
    """Mock MCP cache."""
    cache = MCPCache()

    cache_storage = {}

    async def mock_get(key: str):
        return cache_storage.get(key)

    async def mock_set(key: str, value: Any):
        cache_storage[key] = value

    def mock_generate_key(request: MCPRequest):
        return f"test_key_{request.id}"

    async def mock_get_stats():
        return {"hits": 0, "misses": 0, "size": len(cache_storage)}

    cache.get = mock_get
    cache.set = mock_set
    cache.generate_key = mock_generate_key
    cache.get_stats = mock_get_stats

    return cache


@pytest.fixture
async def mcp_server(
    mock_tool_registry, mock_router, mock_permission_manager, mock_cache
):
    """Create MCP server instance."""
    app = FastAPI()

    server = SignalCLIMCPServer(
        app=app,
        tool_registry=mock_tool_registry,
        router=mock_router,
        permission_manager=mock_permission_manager,
        cache=mock_cache,
    )

    return server, app


@pytest.fixture
async def test_client(mcp_server):
    """Create test client."""
    server, app = mcp_server
    return TestClient(app)


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    @pytest.mark.asyncio
    async def test_list_tools(self, mcp_server):
        """Test tool listing endpoint."""
        server, _ = mcp_server

        result = await server.list_tools()

        assert "tools" in result
        assert len(result["tools"]) == 2

        tool_names = [tool["name"] for tool in result["tools"]]
        assert "rag_query" in tool_names
        assert "json_query" in tool_names

    @pytest.mark.asyncio
    async def test_get_tool_details(self, mcp_server):
        """Test getting tool details."""
        server, _ = mcp_server

        result = await server.get_tool("rag_query")

        assert result["name"] == "rag_query"
        assert result["description"] == "Perform RAG query"
        assert result["streaming"] is True
        assert "input_schema" in result
        assert "output_schema" in result

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mcp_server):
        """Test successful tool execution."""
        server, _ = mcp_server

        request = MCPRequest(
            id="test_request_1",
            method="tools/call",
            params={
                "tool": "rag_query",
                "query": "What is machine learning?",
                "max_results": 3,
            },
        )

        response = await server.execute_tool(request)

        assert response.id == "test_request_1"
        assert response.error is None
        assert "result" in response.result
        assert "answer" in response.result["result"]
        assert "sources" in response.result["result"]

    @pytest.mark.asyncio
    async def test_execute_tool_with_client_permissions(self, mcp_server):
        """Test tool execution with client permissions."""
        server, _ = mcp_server

        request = MCPRequest(
            id="test_request_2",
            method="tools/call",
            params={
                "tool": "json_query",
                "query": "Generate JSON output",
                "schema": {
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                },
            },
        )

        response = await server.execute_tool(request, client_id="test_client")

        assert response.error is None
        assert "result" in response.result

    @pytest.mark.asyncio
    async def test_execute_tool_permission_denied(self, mcp_server):
        """Test tool execution with insufficient permissions."""
        server, _ = mcp_server

        # Mock permission manager to deny access
        async def mock_can_access_tool(client, tool_name: str):
            return False

        server.permission_manager.can_access_tool = mock_can_access_tool

        request = MCPRequest(
            id="test_request_3",
            method="tools/call",
            params={"tool": "rag_query", "query": "Test query"},
        )

        with pytest.raises(Exception):  # Should raise HTTPException
            await server.execute_tool(request, client_id="test_client")

    @pytest.mark.asyncio
    async def test_tool_caching(self, mcp_server):
        """Test tool response caching."""
        server, _ = mcp_server

        request = MCPRequest(
            id="test_request_4",
            method="tools/call",
            params={"tool": "rag_query", "query": "Cached query test"},
        )

        # First request - should execute and cache
        response1 = await server.execute_tool(request)
        assert response1.error is None

        # Second request with same params - should hit cache
        request.id = "test_request_5"
        response2 = await server.execute_tool(request)
        assert response2.error is None
        assert response2.result == response1.result

    @pytest.mark.asyncio
    async def test_streaming_tool_execution(self, mcp_server):
        """Test streaming tool execution setup."""
        server, _ = mcp_server

        request = MCPRequest(
            id="test_request_6",
            method="tools/call",
            params={"tool": "rag_query", "query": "Streaming test"},
            stream=True,
        )

        response = await server.execute_tool(request)

        assert response.error is None
        assert "stream_id" in response.result
        assert "stream_url" in response.result
        assert "stream_ws" in response.result

    @pytest.mark.asyncio
    async def test_health_check(self, mcp_server):
        """Test health check endpoint."""
        server, _ = mcp_server

        result = await server.health_check()

        assert result["status"] == "healthy"
        assert result["server"] == "SignalCLI-MCP"
        assert "tools_loaded" in result
        assert result["tools_loaded"] == 2

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, mcp_server):
        """Test metrics endpoint."""
        server, _ = mcp_server

        result = await server.get_metrics()

        assert "requests" in result
        assert "latency" in result
        assert "cache" in result
        assert "streams" in result

    @pytest.mark.asyncio
    async def test_tool_chain_execution(self, mcp_server):
        """Test tool chain execution."""
        server, _ = mcp_server

        from src.mcp.protocol import MCPToolCall

        tool_calls = [
            MCPToolCall(tool="rag_query", parameters={"query": "First query"}),
            MCPToolCall(
                tool="json_query",
                parameters={
                    "query": "Second query",
                    "schema": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}},
                    },
                },
            ),
        ]

        response = await server.execute_chain(tool_calls)

        assert response.error is None
        assert "chain_results" in response.result
        assert len(response.result["chain_results"]) == 2

    @pytest.mark.asyncio
    async def test_error_handling(self, mcp_server):
        """Test error handling in tool execution."""
        server, _ = mcp_server

        # Mock tool to raise exception
        async def mock_failing_execute(
            params: Dict[str, Any], context: Dict[str, Any]
        ) -> Dict[str, Any]:
            raise ValueError("Test error")

        server.tool_registry.get_tool("rag_query").execute = mock_failing_execute

        request = MCPRequest(
            id="test_request_error",
            method="tools/call",
            params={"tool": "rag_query", "query": "This will fail"},
        )

        response = await server.execute_tool(request)

        assert response.id == "test_request_error"
        assert response.error is not None
        assert "Test error" in response.error["message"]


class TestMCPServerHTTP:
    """HTTP-specific tests for MCP server."""

    def test_health_check_http(self, test_client):
        """Test health check via HTTP."""
        response = test_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_metrics_http(self, test_client):
        """Test metrics via HTTP."""
        response = test_client.get("/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "requests" in data
        assert "cache" in data


@pytest.mark.asyncio
class TestMCPServerConcurrency:
    """Test concurrent operations on MCP server."""

    async def test_concurrent_tool_execution(self, mcp_server):
        """Test concurrent tool execution."""
        server, _ = mcp_server

        # Create multiple concurrent requests
        requests = [
            MCPRequest(
                id=f"concurrent_request_{i}",
                method="tools/call",
                params={"tool": "rag_query", "query": f"Concurrent query {i}"},
            )
            for i in range(10)
        ]

        # Execute concurrently
        responses = await asyncio.gather(
            *[server.execute_tool(request) for request in requests]
        )

        # Verify all responses
        assert len(responses) == 10
        for i, response in enumerate(responses):
            assert response.id == f"concurrent_request_{i}"
            assert response.error is None

    async def test_concurrent_cache_access(self, mcp_server):
        """Test concurrent cache access."""
        server, _ = mcp_server

        # Same request executed concurrently - should use cache
        request = MCPRequest(
            id="cache_test_base",
            method="tools/call",
            params={"tool": "rag_query", "query": "Cache test query"},
        )

        # Execute first request to populate cache
        await server.execute_tool(request)

        # Now execute multiple identical requests concurrently
        concurrent_requests = [
            MCPRequest(id=f"cache_test_{i}", method="tools/call", params=request.params)
            for i in range(5)
        ]

        responses = await asyncio.gather(
            *[server.execute_tool(req) for req in concurrent_requests]
        )

        # All should return same cached result
        assert len(responses) == 5
        first_result = responses[0].result
        for response in responses[1:]:
            assert response.result == first_result
