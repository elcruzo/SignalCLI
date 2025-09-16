"""MCP Protocol implementation following the official spec."""

import json
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# MCP Protocol Version
MCP_VERSION = "2024-11-05"


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error object."""

    code: int
    message: str
    data: Optional[Any] = None


# Standard JSON-RPC error codes
class ErrorCode(Enum):
    """Standard JSON-RPC and MCP error codes."""

    # JSON-RPC standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603

    # MCP-specific errors
    TOOL_NOT_FOUND = -32001
    TOOL_EXECUTION_ERROR = -32002
    PERMISSION_DENIED = -32003
    RATE_LIMIT_EXCEEDED = -32004
    CONTEXT_LENGTH_EXCEEDED = -32005
    INVALID_TOOL_SCHEMA = -32006


class MCPRequest(BaseModel):
    """MCP request following JSON-RPC 2.0 format."""

    jsonrpc: str = "2.0"
    id: Union[str, int, None] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None


class MCPResponse(BaseModel):
    """MCP response following JSON-RPC 2.0 format."""

    jsonrpc: str = "2.0"
    id: Union[str, int, None]
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None

    class Config:
        json_encoders = {JSONRPCError: lambda v: v.dict(exclude_none=True)}


class MCPNotification(BaseModel):
    """MCP notification (no id field)."""

    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None


# MCP Protocol Methods
class MCPMethod(str, Enum):
    """Standard MCP methods."""

    # Initialization
    INITIALIZE = "initialize"
    INITIALIZED = "notifications/initialized"

    # Tool discovery
    TOOLS_LIST = "tools/list"

    # Tool execution
    TOOLS_CALL = "tools/call"

    # Prompts
    PROMPTS_LIST = "prompts/list"
    PROMPTS_GET = "prompts/get"

    # Resources
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCES_SUBSCRIBE = "resources/subscribe"
    RESOURCES_UNSUBSCRIBE = "resources/unsubscribe"

    # Completions
    COMPLETION_COMPLETE = "completion/complete"

    # Logging
    LOGGING_SET_LEVEL = "logging/setLevel"

    # Notifications
    NOTIFICATION_PROGRESS = "notifications/progress"
    NOTIFICATION_LOG = "notifications/message"
    NOTIFICATION_RESOURCE_UPDATED = "notifications/resources/updated"
    NOTIFICATION_TOOL_LIST_CHANGED = "notifications/tools/list_changed"
    NOTIFICATION_PROMPT_LIST_CHANGED = "notifications/prompts/list_changed"


# MCP Protocol Types
class MCPRole(str, Enum):
    """MCP roles."""

    USER = "user"
    ASSISTANT = "assistant"


class MCPToolInfo(BaseModel):
    """Tool information in MCP format."""

    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any]


class MCPPromptInfo(BaseModel):
    """Prompt information in MCP format."""

    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None


class MCPResourceInfo(BaseModel):
    """Resource information in MCP format."""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None


class MCPToolCall(BaseModel):
    """Tool call parameters."""

    name: str
    arguments: Optional[Dict[str, Any]] = None


class MCPToolResult(BaseModel):
    """Tool execution result."""

    content: List[Dict[str, Any]]  # Array of content items
    isError: Optional[bool] = False


class MCPContent(BaseModel):
    """Content item in tool results."""

    type: str  # "text", "image", "resource"
    text: Optional[str] = None
    data: Optional[str] = None  # Base64 for images
    mimeType: Optional[str] = None
    uri: Optional[str] = None  # For resource references


class MCPInitializeParams(BaseModel):
    """Initialize request parameters."""

    protocolVersion: str = MCP_VERSION
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    clientInfo: Dict[str, Any] = Field(default_factory=dict)


class MCPServerCapabilities(BaseModel):
    """Server capabilities."""

    tools: Optional[Dict[str, Any]] = None
    prompts: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    logging: Optional[Dict[str, Any]] = None


class MCPInitializeResult(BaseModel):
    """Initialize response result."""

    protocolVersion: str = MCP_VERSION
    capabilities: MCPServerCapabilities
    serverInfo: Dict[str, Any]


# Helper functions
def create_error_response(
    request_id: Union[str, int, None],
    code: ErrorCode,
    message: str,
    data: Optional[Any] = None,
) -> MCPResponse:
    """Create an error response."""
    return MCPResponse(
        id=request_id, error=JSONRPCError(code=code.value, message=message, data=data)
    )


def create_success_response(
    request_id: Union[str, int, None], result: Any
) -> MCPResponse:
    """Create a success response."""
    return MCPResponse(id=request_id, result=result)


def validate_mcp_request(data: Dict[str, Any]) -> Optional[str]:
    """Validate MCP request format."""
    if "jsonrpc" not in data or data["jsonrpc"] != "2.0":
        return "Missing or invalid jsonrpc version"

    if "method" not in data:
        return "Missing method field"

    # Notifications don't have id
    if "id" not in data and data["method"].startswith("notifications/"):
        return None

    return None


def format_tool_for_mcp(tool: "Tool") -> MCPToolInfo:
    """Convert internal tool format to MCP format."""
    return MCPToolInfo(
        name=tool.name, description=tool.description, inputSchema=tool.input_schema
    )


def format_tool_result(result: Any, is_error: bool = False) -> MCPToolResult:
    """Format tool result for MCP response."""
    # Handle different result types
    content_items = []

    if isinstance(result, str):
        content_items.append({"type": "text", "text": result})
    elif isinstance(result, dict):
        # Check if it's already formatted as content
        if "type" in result and result["type"] in ["text", "image", "resource"]:
            content_items.append(result)
        else:
            # Convert dict to text
            content_items.append({"type": "text", "text": json.dumps(result, indent=2)})
    elif isinstance(result, list):
        # Assume it's already a list of content items
        content_items = result
    else:
        # Convert to string
        content_items.append({"type": "text", "text": str(result)})

    return MCPToolResult(content=content_items, isError=is_error)
