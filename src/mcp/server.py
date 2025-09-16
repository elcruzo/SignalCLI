"""MCP Server implementation for SignalCLI."""

import asyncio
import json
from typing import Dict, Any, Optional, List, AsyncIterator, Union
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .protocol import (
    MCPRequest,
    MCPResponse,
    MCPNotification,
    MCPMethod,
    ErrorCode,
    MCPInitializeParams,
    MCPInitializeResult,
    MCPServerCapabilities,
    MCPToolCall,
    MCPToolResult,
    MCPToolInfo,
    MCPContent,
    create_error_response,
    create_success_response,
    validate_mcp_request,
    format_tool_for_mcp,
    format_tool_result,
    MCP_VERSION,
)
from .tools import ToolRegistry, Tool
from .router import ContextAwareRouter
from .handlers import StreamingHandler, ToolChainHandler
from .permissions import PermissionManager, MCPClient
from .cache import MCPCache
from ..utils.logger import get_logger
from ..infrastructure.observability.metrics import get_metrics_registry

logger = get_logger(__name__)
metrics = get_metrics_registry()


# Session management
class MCPSession:
    """MCP session state."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.initialized = False
        self.client_info: Dict[str, Any] = {}
        self.capabilities: Dict[str, Any] = {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()


class SignalCLIMCPServer:
    """MCP Server for SignalCLI."""

    def __init__(
        self,
        app: FastAPI,
        tool_registry: ToolRegistry,
        router: ContextAwareRouter,
        permission_manager: PermissionManager,
        cache: MCPCache,
    ):
        """Initialize MCP server."""
        self.app = app
        self.tool_registry = tool_registry
        self.router = router
        self.permission_manager = permission_manager
        self.cache = cache
        self.streaming_handler = StreamingHandler()
        self.chain_handler = ToolChainHandler(tool_registry, router)

        # Register MCP endpoints
        self._register_endpoints()

        # Metrics
        self.request_counter = metrics.counter(
            "mcp_requests_total", "Total MCP requests", ["method", "client", "status"]
        )
        self.latency_histogram = metrics.histogram(
            "mcp_request_duration_seconds", "MCP request latency", ["method", "tool"]
        )

    def _register_endpoints(self):
        """Register MCP endpoints."""
        # Main MCP endpoint - handles all JSON-RPC requests
        self.app.post("/mcp")(self.handle_mcp_request)
        self.app.websocket("/mcp")(self.handle_mcp_websocket)

        # Legacy/convenience endpoints (not part of MCP spec)
        self.app.get("/health")(self.health_check)
        self.app.get("/metrics")(self.get_metrics)

    async def list_tools(self, client_id: Optional[str] = None) -> Dict[str, Any]:
        """List available tools for a client."""
        # Get client permissions
        if client_id:
            client = await self.permission_manager.get_client(client_id)
            allowed_tools = client.allowed_tools if client else []
        else:
            allowed_tools = None

        # Get tools
        tools = self.tool_registry.list_tools(allowed_tools)

        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "capabilities": [cap.value for cap in tool.capabilities],
                    "input_schema": tool.input_schema,
                    "output_schema": tool.output_schema,
                    "streaming": tool.supports_streaming,
                }
                for tool in tools
            ],
            "version": "1.0.0",
            "server": "SignalCLI-MCP",
        }

    async def get_tool(
        self, tool_name: str, client_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed information about a specific tool."""
        # Check permissions
        if client_id:
            client = await self.permission_manager.get_client(client_id)
            if not await self.permission_manager.can_access_tool(client, tool_name):
                raise HTTPException(status_code=403, detail="Access denied")

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        return {
            "name": tool.name,
            "description": tool.description,
            "capabilities": [cap.value for cap in tool.capabilities],
            "input_schema": tool.input_schema,
            "output_schema": tool.output_schema,
            "streaming": tool.supports_streaming,
            "examples": tool.examples,
            "metadata": tool.metadata,
        }

    async def execute_tool(
        self, request: MCPRequest, client_id: Optional[str] = None
    ) -> MCPResponse:
        """Execute a single tool."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Validate client permissions
            if client_id:
                client = await self.permission_manager.get_client(client_id)
                tool_name = request.params.get("tool")
                if not await self.permission_manager.can_access_tool(client, tool_name):
                    raise HTTPException(status_code=403, detail="Access denied")

            # Check cache
            cache_key = self.cache.generate_key(request)
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit for request {request.id}")
                self.request_counter.labels(
                    method="execute",
                    client=client_id or "anonymous",
                    status="cache_hit",
                ).inc()
                return MCPResponse(id=request.id, result=cached_result)

            # Route to appropriate tool
            tool_name = await self.router.route(request)
            tool = self.tool_registry.get_tool(tool_name)

            if not tool:
                raise HTTPException(
                    status_code=404, detail=f"Tool '{tool_name}' not found"
                )

            # Execute tool
            if request.stream and tool.supports_streaming:
                # Return streaming response info
                stream_id = str(uuid.uuid4())
                self.streaming_handler.create_stream(stream_id, tool, request)

                result = {
                    "stream_id": stream_id,
                    "stream_url": f"/mcp/v1/stream/sse?stream_id={stream_id}",
                    "stream_ws": f"/mcp/v1/stream?stream_id={stream_id}",
                }
            else:
                # Execute synchronously
                result = await tool.execute(request.params, request.context)

                # Cache result
                await self.cache.set(cache_key, result)

            # Record metrics
            latency = asyncio.get_event_loop().time() - start_time
            self.latency_histogram.labels(method="execute", tool=tool_name).observe(
                latency
            )
            self.request_counter.labels(
                method="execute", client=client_id or "anonymous", status="success"
            ).inc()

            return MCPResponse(
                id=request.id,
                result=result,
                metadata={
                    "tool": tool_name,
                    "latency_ms": int(latency * 1000),
                    "cached": False,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            self.request_counter.labels(
                method="execute", client=client_id or "anonymous", status="error"
            ).inc()

            return MCPResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": str(e),
                    "data": {"type": type(e).__name__},
                },
            )

    async def execute_chain(
        self, tools: List[MCPToolCall], client_id: Optional[str] = None
    ) -> MCPResponse:
        """Execute a chain of tools."""
        request_id = str(uuid.uuid4())

        try:
            # Validate permissions for all tools
            if client_id:
                client = await self.permission_manager.get_client(client_id)
                for tool_call in tools:
                    if not await self.permission_manager.can_access_tool(
                        client, tool_call.tool
                    ):
                        raise HTTPException(
                            status_code=403,
                            detail=f"Access denied for tool '{tool_call.tool}'",
                        )

            # Execute chain
            results = await self.chain_handler.execute_chain(tools)

            return MCPResponse(
                id=request_id,
                result={"chain_results": results, "success": True},
                metadata={
                    "tools_executed": [t.tool for t in tools],
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

        except Exception as e:
            logger.error(f"Error executing chain: {e}")
            return MCPResponse(
                id=request_id,
                error={
                    "code": -32603,
                    "message": str(e),
                    "data": {"type": type(e).__name__},
                },
            )

    async def stream_websocket(self, websocket: WebSocket, stream_id: str):
        """WebSocket endpoint for streaming responses."""
        await websocket.accept()

        try:
            stream = self.streaming_handler.get_stream(stream_id)
            if not stream:
                await websocket.send_json({"error": "Stream not found"})
                await websocket.close()
                return

            # Stream results
            async for chunk in stream:
                await websocket.send_json({"chunk": chunk, "stream_id": stream_id})

            # Send completion
            await websocket.send_json({"status": "completed", "stream_id": stream_id})

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            await websocket.send_json({"error": str(e)})
        finally:
            await websocket.close()

    async def stream_sse(self, stream_id: str):
        """Server-Sent Events endpoint for streaming."""

        async def event_generator():
            stream = self.streaming_handler.get_stream(stream_id)
            if not stream:
                yield f"data: {json.dumps({'error': 'Stream not found'})}\n\n"
                return

            try:
                async for chunk in stream:
                    yield f"data: {json.dumps({'chunk': chunk, 'stream_id': stream_id})}\n\n"

                yield f"data: {json.dumps({'status': 'completed', 'stream_id': stream_id})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    async def register_client(self, client_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new MCP client."""
        client = await self.permission_manager.register_client(
            name=client_data["name"],
            allowed_tools=client_data.get("allowed_tools", []),
            metadata=client_data.get("metadata", {}),
        )

        return {
            "client_id": client.id,
            "api_key": client.api_key,
            "name": client.name,
            "allowed_tools": client.allowed_tools,
            "created_at": client.created_at.isoformat(),
        }

    async def get_permissions(self, client_id: str) -> Dict[str, Any]:
        """Get client permissions."""
        client = await self.permission_manager.get_client(client_id)
        if not client:
            raise HTTPException(status_code=404, detail="Client not found")

        return {
            "client_id": client.id,
            "name": client.name,
            "allowed_tools": client.allowed_tools,
            "rate_limits": client.rate_limits,
            "metadata": client.metadata,
        }

    async def handle_mcp_request(self, request: Request) -> JSONResponse:
        """Handle MCP JSON-RPC requests."""
        try:
            # Parse request body
            body = await request.json()
            
            # Validate MCP request format
            if not validate_mcp_request(body):
                return JSONResponse(
                    status_code=400,
                    content=create_error_response(
                        body.get("id", None),
                        ErrorCode.INVALID_REQUEST,
                        "Invalid MCP request format"
                    )
                )
            
            # Create MCPRequest object
            mcp_request = MCPRequest(**body)
            
            # Handle based on method
            if mcp_request.method == MCPMethod.INITIALIZE:
                result = await self._handle_initialize(mcp_request)
            elif mcp_request.method == MCPMethod.LIST_TOOLS:
                result = await self.list_tools()
            elif mcp_request.method == MCPMethod.CALL_TOOL:
                response = await self.execute_tool(mcp_request)
                result = response.result
            else:
                return JSONResponse(
                    status_code=400,
                    content=create_error_response(
                        mcp_request.id,
                        ErrorCode.METHOD_NOT_FOUND,
                        f"Method {mcp_request.method} not found"
                    )
                )
                
            return JSONResponse(
                content=create_success_response(mcp_request.id, result)
            )
            
        except Exception as e:
            logger.error(f"Error handling MCP request: {e}")
            return JSONResponse(
                status_code=500,
                content=create_error_response(
                    body.get("id", None) if "body" in locals() else None,
                    ErrorCode.INTERNAL_ERROR,
                    str(e)
                )
            )
    
    async def handle_mcp_websocket(self, websocket: WebSocket):
        """Handle MCP WebSocket connections."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        session = MCPSession(session_id)
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_json()
                
                # Validate request
                if not validate_mcp_request(data):
                    await websocket.send_json(
                        create_error_response(
                            data.get("id", None),
                            ErrorCode.INVALID_REQUEST,
                            "Invalid MCP request"
                        )
                    )
                    continue
                    
                # Create request object
                mcp_request = MCPRequest(**data)
                
                # Handle request
                if mcp_request.method == MCPMethod.INITIALIZE:
                    result = await self._handle_initialize(mcp_request)
                    session.initialized = True
                elif mcp_request.method == MCPMethod.LIST_TOOLS:
                    result = await self.list_tools()
                elif mcp_request.method == MCPMethod.CALL_TOOL:
                    response = await self.execute_tool(mcp_request)
                    result = response.result
                else:
                    await websocket.send_json(
                        create_error_response(
                            mcp_request.id,
                            ErrorCode.METHOD_NOT_FOUND,
                            f"Method {mcp_request.method} not found"
                        )
                    )
                    continue
                    
                # Send response
                await websocket.send_json(
                    create_success_response(mcp_request.id, result)
                )
                
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            await websocket.close()
            
    async def _handle_initialize(self, request: MCPRequest) -> Dict[str, Any]:
        """Handle initialization request."""
        return {
            "protocolVersion": MCP_VERSION,
            "serverInfo": {
                "name": "SignalCLI-MCP",
                "version": "1.0.0"
            },
            "capabilities": {
                "tools": True,
                "streaming": True,
                "context": True
            }
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "server": "SignalCLI-MCP",
            "version": "1.0.0",
            "tools_loaded": len(self.tool_registry.list_tools()),
            "active_streams": self.streaming_handler.active_stream_count(),
            "cache_stats": await self.cache.get_stats(),
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics."""
        # Get metrics from the metrics collector
        metrics_data = metrics.get_metrics() if metrics else {}
        
        return {
            "metrics": metrics_data,
            "cache": await self.cache.get_stats(),
            "streams": {
                "active": self.streaming_handler.active_stream_count(),
                "total_created": getattr(self.streaming_handler, 'total_streams_created', 0),
            },
        }
