"""MCP (Model Context Protocol) server implementation for SignalCLI."""

from .server import SignalCLIMCPServer
from .tools import ToolRegistry, Tool, ToolCapability
from .router import ContextAwareRouter
from .handlers import StreamingHandler, ToolChainHandler

__all__ = [
    "SignalCLIMCPServer",
    "ToolRegistry",
    "Tool",
    "ToolCapability",
    "ContextAwareRouter",
    "StreamingHandler",
    "ToolChainHandler",
]