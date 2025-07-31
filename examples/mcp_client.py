"""Example MCP client implementation."""

import asyncio
import json
import aiohttp
import websockets
from typing import Dict, Any, Optional, AsyncIterator


class MCPClient:
    """Simple MCP client for interacting with SignalCLI MCP server."""

    def __init__(self, base_url: str = "http://localhost:8001", api_key: Optional[str] = None):
        """Initialize MCP client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/mcp/v1/tools", headers=self.headers
            ) as response:
                return await response.json()

    async def get_tool(self, tool_name: str) -> Dict[str, Any]:
        """Get detailed information about a tool."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/mcp/v1/tools/{tool_name}", headers=self.headers
            ) as response:
                return await response.json()

    async def execute_tool(
        self, tool: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a tool."""
        request_data = {
            "method": "execute",
            "params": {"tool": tool, **arguments},
            "context": context or {},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp/v1/execute",
                headers=self.headers,
                json=request_data,
            ) as response:
                return await response.json()

    async def execute_chain(
        self, tool_calls: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a chain of tools."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp/v1/execute/chain",
                headers=self.headers,
                json=tool_calls,
            ) as response:
                return await response.json()

    async def stream_tool(
        self, tool: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream tool execution results via SSE."""
        request_data = {
            "method": "execute",
            "params": {"tool": tool, **arguments},
            "context": context or {},
            "stream": True,
        }

        # First, initiate the stream
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp/v1/execute",
                headers=self.headers,
                json=request_data,
            ) as response:
                result = await response.json()
                stream_id = result["result"]["stream_id"]

            # Then connect to SSE endpoint
            async with session.get(
                f"{self.base_url}/mcp/v1/stream/sse?stream_id={stream_id}",
                headers=self.headers,
            ) as response:
                async for line in response.content:
                    if line.startswith(b"data: "):
                        data = json.loads(line[6:].decode())
                        yield data

    async def stream_tool_websocket(
        self, tool: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream tool execution results via WebSocket."""
        request_data = {
            "method": "execute",
            "params": {"tool": tool, **arguments},
            "context": context or {},
            "stream": True,
        }

        # First, initiate the stream
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/mcp/v1/execute",
                headers=self.headers,
                json=request_data,
            ) as response:
                result = await response.json()
                stream_id = result["result"]["stream_id"]

        # Connect to WebSocket
        ws_url = self.base_url.replace("http", "ws")
        async with websockets.connect(
            f"{ws_url}/mcp/v1/stream?stream_id={stream_id}"
        ) as websocket:
            async for message in websocket:
                data = json.loads(message)
                yield data
                if data.get("status") == "completed":
                    break


# Example usage
async def main():
    """Example usage of MCP client."""
    # Initialize client
    client = MCPClient()

    # List available tools
    print("\n=== Available Tools ===")
    tools = await client.list_tools()
    for tool in tools["tools"]:
        print(f"- {tool['name']}: {tool['description']}")
        print(f"  Capabilities: {', '.join(tool['capabilities'])}")

    # Execute RAG query
    print("\n=== RAG Query ===")
    result = await client.execute_tool(
        "rag_query",
        {"query": "What is machine learning?", "top_k": 3},
    )
    print(f"Answer: {result['result']['answer']}")
    print(f"Confidence: {result['result']['confidence']}")

    # Stream text generation
    print("\n=== Streaming Text Generation ===")
    async for chunk in client.stream_tool(
        "text_generation",
        {"prompt": "Explain quantum computing in simple terms:", "max_tokens": 200},
    ):
        if chunk.get("chunk"):
            print(chunk["chunk"]["chunk"], end="", flush=True)
        elif chunk.get("status") == "completed":
            print("\n[Stream completed]")

    # Execute tool chain
    print("\n=== Tool Chain ===")
    chain_result = await client.execute_chain(
        [
            {
                "tool": "text_generation",
                "arguments": {"prompt": "Generate a short story about AI", "max_tokens": 100},
            },
            {
                "tool": "text_analysis",
                "arguments": {"text": "{{result_0.result.generated_text}}", "analyses": ["sentiment", "topics"]},
            },
        ]
    )
    print(f"Chain results: {json.dumps(chain_result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())
