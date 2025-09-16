"""Tool registry and definitions for MCP server."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Callable, AsyncIterator
from enum import Enum
from dataclasses import dataclass, field
import inspect
import asyncio

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ToolCapability(Enum):
    """Tool capabilities."""

    QUERY = "query"
    SEARCH = "search"
    SUMMARIZE = "summarize"
    GENERATE = "generate"
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    VALIDATE = "validate"
    STORE = "store"
    RETRIEVE = "retrieve"


@dataclass
class ToolMetadata:
    """Tool metadata."""

    version: str = "1.0.0"
    author: str = "SignalCLI"
    tags: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None  # requests per minute
    timeout: int = 30  # seconds
    cost_estimate: Optional[float] = None  # relative cost units


class Tool(ABC):
    """Base class for MCP tools."""

    def __init__(
        self,
        name: str,
        description: str,
        capabilities: List[ToolCapability],
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        supports_streaming: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[ToolMetadata] = None,
    ):
        """Initialize tool."""
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.supports_streaming = supports_streaming
        self.examples = examples or []
        self.metadata = metadata or ToolMetadata()

    @abstractmethod
    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute the tool."""
        pass

    async def stream_execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Any]:
        """Execute tool with streaming response."""
        # Default implementation yields single result
        result = await self.execute(params, context)
        yield result

    def validate_input(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters against schema."""
        # TODO: Implement JSON schema validation
        return True

    def validate_output(self, result: Any) -> bool:
        """Validate output against schema."""
        # TODO: Implement JSON schema validation
        return True


class ToolRegistry:
    """Registry for managing MCP tools."""

    def __init__(self):
        """Initialize tool registry."""
        self._tools: Dict[str, Tool] = {}
        self._capability_index: Dict[ToolCapability, Set[str]] = {}
        self._tag_index: Dict[str, Set[str]] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")

        self._tools[tool.name] = tool

        # Update capability index
        for capability in tool.capabilities:
            if capability not in self._capability_index:
                self._capability_index[capability] = set()
            self._capability_index[capability].add(tool.name)

        # Update tag index
        for tag in tool.metadata.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        if name not in self._tools:
            return

        tool = self._tools[name]

        # Remove from capability index
        for capability in tool.capabilities:
            self._capability_index[capability].discard(name)

        # Remove from tag index
        for tag in tool.metadata.tags:
            self._tag_index[tag].discard(name)

        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self, filter_names: Optional[List[str]] = None) -> List[Tool]:
        """List all tools, optionally filtered by names."""
        if filter_names:
            return [self._tools[name] for name in filter_names if name in self._tools]
        return list(self._tools.values())

    def find_by_capability(self, capability: ToolCapability) -> List[Tool]:
        """Find tools by capability."""
        tool_names = self._capability_index.get(capability, set())
        return [self._tools[name] for name in tool_names]

    def find_by_tag(self, tag: str) -> List[Tool]:
        """Find tools by tag."""
        tool_names = self._tag_index.get(tag, set())
        return [self._tools[name] for name in tool_names]

    def find_by_capabilities(self, capabilities: List[ToolCapability]) -> List[Tool]:
        """Find tools that have all specified capabilities."""
        if not capabilities:
            return []

        # Get tools with first capability
        tool_names = self._capability_index.get(capabilities[0], set()).copy()

        # Intersect with tools having other capabilities
        for capability in capabilities[1:]:
            tool_names &= self._capability_index.get(capability, set())

        return [self._tools[name] for name in tool_names]


# Built-in SignalCLI tools


class RAGQueryTool(Tool):
    """RAG query tool for SignalCLI."""

    def __init__(self, rag_pipeline):
        """Initialize RAG query tool."""
        super().__init__(
            name="rag_query",
            description="Query the knowledge base using RAG",
            capabilities=[
                ToolCapability.QUERY,
                ToolCapability.RETRIEVE,
                ToolCapability.ANALYZE,
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of documents to retrieve",
                        "default": 5,
                    },
                    "filters": {
                        "type": "object",
                        "description": "Metadata filters for document retrieval",
                    },
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
            },
            supports_streaming=True,
            metadata=ToolMetadata(tags=["rag", "search", "knowledge-base"]),
        )
        self.rag_pipeline = rag_pipeline

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute RAG query - returns MCP-compliant content array."""
        query = params["query"]
        top_k = params.get("top_k", 5)
        filters = params.get("filters", {})

        result = await self.rag_pipeline.query(query, top_k=top_k, filters=filters)

        # Return MCP-compliant content array
        content = [{"type": "text", "text": result.answer}]

        # Add sources as additional content if available
        if result.sources:
            sources_text = "\n\nSources:\n" + "\n".join(
                f"- {src}" for src in result.sources
            )
            content.append({"type": "text", "text": sources_text})

        # Add confidence as metadata
        if result.confidence:
            content.append(
                {"type": "text", "text": f"\nConfidence: {result.confidence:.2f}"}
            )

        return content

    async def stream_execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Any]:
        """Stream RAG query results."""
        query = params["query"]
        top_k = params.get("top_k", 5)
        filters = params.get("filters", {})

        async for chunk in self.rag_pipeline.stream_query(
            query, top_k=top_k, filters=filters
        ):
            yield {"chunk": chunk, "type": "partial"}

        # Yield final result
        result = await self.rag_pipeline.get_final_result()
        yield {
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "type": "final",
        }


class DocumentIndexTool(Tool):
    """Document indexing tool."""

    def __init__(self, vector_store, document_processor):
        """Initialize document index tool."""
        super().__init__(
            name="document_index",
            description="Index documents into the knowledge base",
            capabilities=[ToolCapability.STORE, ToolCapability.TRANSFORM],
            input_schema={
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "metadata": {"type": "object"},
                                "source": {"type": "string"},
                            },
                            "required": ["content"],
                        },
                    },
                    "chunk_size": {"type": "integer", "default": 512},
                    "chunk_overlap": {"type": "integer", "default": 50},
                },
                "required": ["documents"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "indexed_count": {"type": "integer"},
                    "chunk_count": {"type": "integer"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                },
            },
            metadata=ToolMetadata(tags=["indexing", "storage", "documents"]),
        )
        self.vector_store = vector_store
        self.document_processor = document_processor

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Index documents - returns MCP-compliant content array."""
        documents = params["documents"]
        chunk_size = params.get("chunk_size", 512)
        chunk_overlap = params.get("chunk_overlap", 50)

        indexed_count = 0
        total_chunks = 0
        errors = []

        for doc in documents:
            try:
                # Process document into chunks
                chunks = await self.document_processor.process(
                    doc["content"],
                    metadata=doc.get("metadata", {}),
                    source=doc.get("source", "unknown"),
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )

                # Index chunks
                await self.vector_store.add_documents(chunks)

                indexed_count += 1
                total_chunks += len(chunks)

            except Exception as e:
                errors.append(f"Error indexing document: {str(e)}")
                logger.error(f"Indexing error: {e}")

        # Return MCP-compliant content array
        content = [
            {
                "type": "text",
                "text": f"Successfully indexed {indexed_count} documents into {total_chunks} chunks.",
            }
        ]

        if errors:
            content.append(
                {"type": "text", "text": "\nErrors encountered:\n" + "\n".join(errors)}
            )

        return content


class SummarizationTool(Tool):
    """Document summarization tool."""

    def __init__(self, llm_engine):
        """Initialize summarization tool."""
        super().__init__(
            name="summarize",
            description="Summarize text or documents",
            capabilities=[ToolCapability.SUMMARIZE, ToolCapability.TRANSFORM],
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to summarize"},
                    "max_length": {
                        "type": "integer",
                        "description": "Maximum summary length",
                        "default": 200,
                    },
                    "style": {
                        "type": "string",
                        "enum": ["brief", "detailed", "bullet_points"],
                        "default": "brief",
                    },
                },
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "word_count": {"type": "integer"},
                    "key_points": {"type": "array", "items": {"type": "string"}},
                },
            },
            supports_streaming=True,
            metadata=ToolMetadata(tags=["nlp", "summarization", "text-processing"]),
        )
        self.llm_engine = llm_engine

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Summarize text - returns MCP-compliant content array."""
        text = params["text"]
        max_length = params.get("max_length", 200)
        style = params.get("style", "brief")

        # Create prompt based on style
        prompts = {
            "brief": f"Summarize the following text in {max_length} words or less:\n\n{text}",
            "detailed": f"Provide a detailed summary of the following text:\n\n{text}",
            "bullet_points": f"Summarize the following text as bullet points:\n\n{text}",
        }

        prompt = prompts[style]
        summary = await self.llm_engine.generate(prompt, max_tokens=max_length * 2)

        # Extract key points
        key_points_prompt = f"List the key points from this text:\n\n{text}"
        key_points_response = await self.llm_engine.generate(
            key_points_prompt, max_tokens=200
        )
        key_points = [p.strip() for p in key_points_response.split("\n") if p.strip()]

        # Return MCP-compliant content array
        content = [{"type": "text", "text": summary}]

        if key_points:
            content.append(
                {
                    "type": "text",
                    "text": "\n\nKey Points:\n"
                    + "\n".join(f"• {point}" for point in key_points[:5]),
                }
            )

        content.append(
            {"type": "text", "text": f"\n\nWord count: {len(summary.split())}"}
        )

        return content


def create_function_tool(
    name: str,
    func: Callable,
    description: str,
    capabilities: List[ToolCapability],
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    **kwargs,
) -> Tool:
    """Create a tool from a function."""

    class FunctionTool(Tool):
        def __init__(self):
            super().__init__(
                name=name,
                description=description,
                capabilities=capabilities,
                input_schema=input_schema,
                output_schema=output_schema,
                **kwargs,
            )
            self.func = func

        async def execute(
            self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
        ) -> Any:
            # Handle both sync and async functions
            if inspect.iscoroutinefunction(self.func):
                return await self.func(**params, context=context)
            else:
                return self.func(**params, context=context)

    return FunctionTool()
