"""Proper MCP Server implementation following the official protocol."""

import asyncio
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse

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
from .handlers import StreamingHandler
from .permissions import PermissionManager
from .cache import MCPCache
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MCPServer:
    """MCP Server following the official protocol specification."""

    def __init__(
        self,
        app: FastAPI,
        tool_registry: ToolRegistry,
        router: Optional[ContextAwareRouter] = None,
        permission_manager: Optional[PermissionManager] = None,
        cache: Optional[MCPCache] = None,
    ):
        """Initialize MCP server."""
        self.app = app
        self.tool_registry = tool_registry
        self.router = router
        self.permission_manager = permission_manager
        self.cache = cache

        # Session management
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Server info
        self.server_info = {
            "name": "SignalCLI-MCP",
            "version": "1.0.0",
            "vendor": "SignalCLI",
        }

        # Register endpoints
        self._register_endpoints()

    def _register_endpoints(self):
        """Register MCP endpoints."""
        # Main MCP endpoint
        self.app.post("/")(self.handle_mcp_request)
        self.app.websocket("/")(self.handle_mcp_websocket)

        # Health check (not part of MCP spec)
        self.app.get("/health")(self.health_check)

    async def handle_mcp_request(self, request: Request) -> JSONResponse:
        """Handle MCP JSON-RPC requests."""
        try:
            # Parse request body
            data = await request.json()

            # Validate request format
            validation_error = validate_mcp_request(data)
            if validation_error:
                return JSONResponse(
                    create_error_response(
                        data.get("id"), ErrorCode.INVALID_REQUEST, validation_error
                    ).dict()
                )

            # Parse into MCPRequest
            try:
                mcp_request = MCPRequest(**data)
            except Exception as e:
                return JSONResponse(
                    create_error_response(
                        data.get("id"), ErrorCode.INVALID_REQUEST, str(e)
                    ).dict()
                )

            # Route to appropriate handler
            response = await self._route_request(mcp_request)
            return JSONResponse(response.dict(exclude_none=True))

        except json.JSONDecodeError:
            return JSONResponse(
                create_error_response(
                    None, ErrorCode.PARSE_ERROR, "Invalid JSON"
                ).dict()
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return JSONResponse(
                create_error_response(
                    None, ErrorCode.INTERNAL_ERROR, "Internal server error"
                ).dict()
            )

    async def handle_mcp_websocket(self, websocket: WebSocket):
        """Handle MCP WebSocket connections."""
        await websocket.accept()
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = {
            "websocket": websocket,
            "initialized": False,
            "created_at": datetime.utcnow(),
        }

        try:
            while True:
                # Receive message
                data = await websocket.receive_json()

                # Validate and parse request
                validation_error = validate_mcp_request(data)
                if validation_error:
                    await websocket.send_json(
                        create_error_response(
                            data.get("id"), ErrorCode.INVALID_REQUEST, validation_error
                        ).dict()
                    )
                    continue

                try:
                    mcp_request = MCPRequest(**data)
                except Exception as e:
                    await websocket.send_json(
                        create_error_response(
                            data.get("id"), ErrorCode.INVALID_REQUEST, str(e)
                        ).dict()
                    )
                    continue

                # Handle request
                response = await self._route_request(mcp_request, session_id)
                await websocket.send_json(response.dict(exclude_none=True))

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            # Clean up session
            del self._sessions[session_id]
            await websocket.close()

    async def _route_request(
        self, request: MCPRequest, session_id: Optional[str] = None
    ) -> MCPResponse:
        """Route MCP request to appropriate handler."""
        try:
            # Check if session needs to be initialized
            if session_id and request.method != MCPMethod.INITIALIZE:
                session = self._sessions.get(session_id, {})
                if not session.get("initialized", False):
                    return create_error_response(
                        request.id,
                        ErrorCode.INVALID_REQUEST,
                        "Session not initialized. Call initialize first.",
                    )

            # Route based on method
            if request.method == MCPMethod.INITIALIZE:
                return await self._handle_initialize(request, session_id)
            elif request.method == MCPMethod.TOOLS_LIST:
                return await self._handle_tools_list(request)
            elif request.method == MCPMethod.TOOLS_CALL:
                return await self._handle_tools_call(request)
            else:
                return create_error_response(
                    request.id,
                    ErrorCode.METHOD_NOT_FOUND,
                    f"Method '{request.method}' not found",
                )

        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return create_error_response(request.id, ErrorCode.INTERNAL_ERROR, str(e))

    async def _handle_initialize(
        self, request: MCPRequest, session_id: Optional[str] = None
    ) -> MCPResponse:
        """Handle initialize request."""
        try:
            # Parse params
            params = (
                MCPInitializeParams(**request.params)
                if request.params
                else MCPInitializeParams()
            )

            # Check protocol version
            if params.protocolVersion != MCP_VERSION:
                logger.warning(
                    f"Client requested version {params.protocolVersion}, server supports {MCP_VERSION}"
                )

            # Update session
            if session_id:
                self._sessions[session_id]["initialized"] = True
                self._sessions[session_id]["client_info"] = params.clientInfo
                self._sessions[session_id]["capabilities"] = params.capabilities

            # Build server capabilities
            capabilities = MCPServerCapabilities(
                tools={} if self.tool_registry else None,
                prompts=None,  # Not implemented yet
                resources=None,  # Not implemented yet
                logging={},
            )

            result = MCPInitializeResult(
                protocolVersion=MCP_VERSION,
                capabilities=capabilities,
                serverInfo=self.server_info,
            )

            return create_success_response(request.id, result.dict())

        except Exception as e:
            return create_error_response(request.id, ErrorCode.INVALID_PARAMS, str(e))

    async def _handle_tools_list(self, request: MCPRequest) -> MCPResponse:
        """Handle tools/list request."""
        try:
            # Get all tools
            tools = self.tool_registry.list_tools()

            # Convert to MCP format
            tool_infos = [format_tool_for_mcp(tool).dict() for tool in tools]

            return create_success_response(request.id, {"tools": tool_infos})

        except Exception as e:
            return create_error_response(request.id, ErrorCode.INTERNAL_ERROR, str(e))

    async def _handle_tools_call(self, request: MCPRequest) -> MCPResponse:
        """Handle tools/call request."""
        try:
            # Parse tool call
            if not request.params:
                return create_error_response(
                    request.id, ErrorCode.INVALID_PARAMS, "Missing params"
                )

            tool_call = MCPToolCall(**request.params)

            # Get tool
            tool = self.tool_registry.get_tool(tool_call.name)
            if not tool:
                return create_error_response(
                    request.id,
                    ErrorCode.TOOL_NOT_FOUND,
                    f"Tool '{tool_call.name}' not found",
                )

            # Check cache if available
            cache_key = None
            if self.cache:
                cache_key = self.cache.generate_key(
                    {"tool": tool_call.name, "arguments": tool_call.arguments}
                )
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for tool {tool_call.name}")
                    return create_success_response(request.id, cached_result)

            # Execute tool
            try:
                result = await tool.execute(
                    tool_call.arguments or {}, context={"mcp_request_id": request.id}
                )

                # Format result
                tool_result = format_tool_result(result)
                response_data = tool_result.dict()

                # Cache result
                if self.cache and cache_key:
                    await self.cache.set(cache_key, response_data)

                return create_success_response(request.id, response_data)

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                error_result = format_tool_result(
                    f"Tool execution failed: {str(e)}", is_error=True
                )
                return create_success_response(request.id, error_result.dict())

        except Exception as e:
            return create_error_response(request.id, ErrorCode.INTERNAL_ERROR, str(e))

    async def send_notification(
        self, session_id: str, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """Send notification to a specific session."""
        session = self._sessions.get(session_id)
        if not session or "websocket" not in session:
            return

        notification = MCPNotification(method=method, params=params)

        try:
            await session["websocket"].send_json(notification.dict())
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    async def broadcast_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """Broadcast notification to all sessions."""
        notification = MCPNotification(method=method, params=params)

        for session_id, session in self._sessions.items():
            if "websocket" in session:
                try:
                    await session["websocket"].send_json(notification.dict())
                except Exception as e:
                    logger.error(f"Failed to send notification to {session_id}: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "server": self.server_info["name"],
            "version": self.server_info["version"],
            "protocol_version": MCP_VERSION,
            "tools_count": len(self.tool_registry.list_tools()),
            "active_sessions": len(self._sessions),
        }


def create_mcp_server(app: FastAPI, **kwargs) -> MCPServer:
    """Factory function to create MCP server."""
    return MCPServer(app, **kwargs)
