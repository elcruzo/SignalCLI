"""Handler classes for MCP server operations."""

import asyncio
import json
from typing import Dict, Any, List, AsyncIterator, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum

from .tools import Tool, ToolRegistry
from .router import ContextAwareRouter
from ..utils.logger import get_logger

logger = get_logger(__name__)


class StreamStatus(Enum):
    """Stream status enum."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamInfo:
    """Information about an active stream."""

    id: str
    tool: Tool
    request: Dict[str, Any]
    status: StreamStatus = StreamStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    chunks_sent: int = 0
    error: Optional[str] = None


class StreamingHandler:
    """Handles streaming responses from tools."""

    def __init__(self, max_streams: int = 1000):
        """Initialize streaming handler."""
        self._streams: Dict[str, StreamInfo] = {}
        self._stream_queues: Dict[str, asyncio.Queue] = {}
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self.max_streams = max_streams
        self.total_streams_created = 0

    def create_stream(
        self, stream_id: str, tool: Tool, request: Dict[str, Any]
    ) -> StreamInfo:
        """Create a new stream."""
        if len(self._streams) >= self.max_streams:
            # Clean up old completed streams
            self._cleanup_old_streams()

        if len(self._streams) >= self.max_streams:
            raise RuntimeError(
                f"Maximum number of streams ({self.max_streams}) reached"
            )

        stream_info = StreamInfo(id=stream_id, tool=tool, request=request)
        self._streams[stream_id] = stream_info
        self._stream_queues[stream_id] = asyncio.Queue()
        self.total_streams_created += 1

        # Start streaming task
        task = asyncio.create_task(self._run_stream(stream_id))
        self._active_tasks[stream_id] = task

        logger.info(f"Created stream {stream_id} for tool {tool.name}")
        return stream_info

    async def _run_stream(self, stream_id: str) -> None:
        """Run the streaming process for a tool."""
        stream_info = self._streams[stream_id]
        queue = self._stream_queues[stream_id]

        try:
            stream_info.status = StreamStatus.ACTIVE
            tool = stream_info.tool
            request = stream_info.request

            # Execute tool with streaming
            async for chunk in tool.stream_execute(
                request["params"], request.get("context")
            ):
                await queue.put(chunk)
                stream_info.chunks_sent += 1

            stream_info.status = StreamStatus.COMPLETED
            await queue.put(None)  # Signal completion

        except asyncio.CancelledError:
            stream_info.status = StreamStatus.CANCELLED
            logger.info(f"Stream {stream_id} cancelled")
        except Exception as e:
            stream_info.status = StreamStatus.ERROR
            stream_info.error = str(e)
            logger.error(f"Stream {stream_id} error: {e}")
            await queue.put({"error": str(e)})
        finally:
            # Ensure queue is closed
            await queue.put(None)

    async def get_stream(self, stream_id: str) -> AsyncIterator[Any]:
        """Get stream iterator for a stream ID."""
        if stream_id not in self._streams:
            raise ValueError(f"Stream {stream_id} not found")

        queue = self._stream_queues[stream_id]

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream."""
        if stream_id not in self._active_tasks:
            return False

        task = self._active_tasks[stream_id]
        task.cancel()
        return True

    def get_stream_info(self, stream_id: str) -> Optional[StreamInfo]:
        """Get information about a stream."""
        return self._streams.get(stream_id)

    def active_stream_count(self) -> int:
        """Get count of active streams."""
        return sum(
            1
            for stream in self._streams.values()
            if stream.status == StreamStatus.ACTIVE
        )

    def _cleanup_old_streams(self, age_minutes: int = 15) -> None:
        """Clean up old completed streams."""
        cutoff_time = datetime.utcnow()
        to_remove = []

        for stream_id, stream_info in self._streams.items():
            if stream_info.status in [
                StreamStatus.COMPLETED,
                StreamStatus.ERROR,
                StreamStatus.CANCELLED,
            ]:
                age = (cutoff_time - stream_info.created_at).total_seconds() / 60
                if age > age_minutes:
                    to_remove.append(stream_id)

        for stream_id in to_remove:
            del self._streams[stream_id]
            if stream_id in self._stream_queues:
                del self._stream_queues[stream_id]
            if stream_id in self._active_tasks:
                del self._active_tasks[stream_id]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old streams")


@dataclass
class ChainExecutionContext:
    """Context for chain execution."""

    chain_id: str
    results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)


class ToolChainHandler:
    """Handles execution of tool chains."""

    def __init__(self, tool_registry: ToolRegistry, router: ContextAwareRouter):
        """Initialize tool chain handler."""
        self.tool_registry = tool_registry
        self.router = router
        self._active_chains: Dict[str, ChainExecutionContext] = {}

    async def execute_chain(
        self, tool_calls: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a chain of tools."""
        chain_id = str(uuid.uuid4())
        chain_context = ChainExecutionContext(chain_id=chain_id)
        self._active_chains[chain_id] = chain_context

        try:
            results = []
            accumulated_context = context or {}

            for i, tool_call in enumerate(tool_calls):
                try:
                    # Execute tool
                    result = await self._execute_tool_in_chain(
                        tool_call, accumulated_context, chain_context
                    )
                    results.append(result)

                    # Update context for next tool
                    accumulated_context["previous_results"] = results
                    accumulated_context[f"result_{i}"] = result

                    # Check for chain control directives
                    if self._should_stop_chain(result):
                        logger.info(f"Chain {chain_id} stopped at step {i}")
                        break

                except Exception as e:
                    error_info = {
                        "step": i,
                        "tool": tool_call.get("tool", "unknown"),
                        "error": str(e),
                        "type": type(e).__name__,
                    }
                    chain_context.errors.append(error_info)
                    logger.error(f"Chain execution error at step {i}: {e}")

                    # Check error handling policy
                    if self._should_continue_on_error(tool_call):
                        results.append({"error": error_info})
                    else:
                        raise

            chain_context.results = results
            return results

        finally:
            # Clean up
            del self._active_chains[chain_id]

    async def _execute_tool_in_chain(
        self,
        tool_call: Dict[str, Any],
        context: Dict[str, Any],
        chain_context: ChainExecutionContext,
    ) -> Dict[str, Any]:
        """Execute a single tool within a chain."""
        tool_name = tool_call.get("tool")
        if not tool_name:
            # Use router to determine tool
            routing_request = {
                "params": tool_call.get("arguments", {}),
                "context": context,
            }
            tool_name = await self.router.route(routing_request)

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Execute tool
        start_time = asyncio.get_event_loop().time()
        result = await tool.execute(tool_call.get("arguments", {}), context)
        latency = asyncio.get_event_loop().time() - start_time

        # Wrap result with metadata
        return {
            "tool": tool_name,
            "result": result,
            "latency_ms": int(latency * 1000),
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _should_stop_chain(self, result: Dict[str, Any]) -> bool:
        """Check if chain should stop based on result."""
        # Check for explicit stop directive
        if isinstance(result, dict):
            if result.get("stop_chain", False):
                return True
            if result.get("result", {}).get("stop_chain", False):
                return True
        return False

    def _should_continue_on_error(self, tool_call: Dict[str, Any]) -> bool:
        """Check if chain should continue on error."""
        return tool_call.get("continue_on_error", False)

    async def execute_parallel(
        self, tool_calls: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute tools in parallel."""
        tasks = []
        for tool_call in tool_calls:
            task = asyncio.create_task(self._execute_single_tool(tool_call, context))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "error": str(result),
                        "type": type(result).__name__,
                        "tool": tool_calls[i].get("tool", "unknown"),
                    }
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _execute_single_tool(
        self, tool_call: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a single tool."""
        tool_name = tool_call.get("tool")
        if not tool_name:
            routing_request = {
                "params": tool_call.get("arguments", {}),
                "context": context,
            }
            tool_name = await self.router.route(routing_request)

        tool = self.tool_registry.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        result = await tool.execute(tool_call.get("arguments", {}), context)
        return {"tool": tool_name, "result": result}


class BatchHandler:
    """Handles batch processing of multiple requests."""

    def __init__(self, max_batch_size: int = 100):
        """Initialize batch handler."""
        self.max_batch_size = max_batch_size
        self._batches: Dict[str, List[Dict[str, Any]]] = {}

    async def add_to_batch(self, batch_id: str, request: Dict[str, Any]) -> None:
        """Add request to batch."""
        if batch_id not in self._batches:
            self._batches[batch_id] = []

        batch = self._batches[batch_id]
        if len(batch) >= self.max_batch_size:
            raise ValueError(
                f"Batch {batch_id} is full (max size: {self.max_batch_size})"
            )

        batch.append(request)

    async def process_batch(
        self, batch_id: str, handler: ToolChainHandler
    ) -> List[Dict[str, Any]]:
        """Process all requests in a batch."""
        if batch_id not in self._batches:
            raise ValueError(f"Batch {batch_id} not found")

        batch = self._batches[batch_id]
        results = []

        # Process in parallel chunks
        chunk_size = 10
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i : i + chunk_size]
            chunk_results = await handler.execute_parallel(chunk)
            results.extend(chunk_results)

        # Clean up
        del self._batches[batch_id]
        return results

    def get_batch_size(self, batch_id: str) -> int:
        """Get current size of a batch."""
        return len(self._batches.get(batch_id, []))
