"""Main entry point for MCP server."""

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
import uvicorn

from .server_proper import MCPServer, create_mcp_server
from .tools import ToolRegistry, RAGQueryTool, DocumentIndexTool, SummarizationTool
from .router import ContextAwareRouter, RoutingStrategy
from .permissions import PermissionManager
from .cache import MCPCache
from ..application.rag.pipeline import RAGPipeline
from ..infrastructure.llm import get_llm_engine
from ..infrastructure.vector_store import get_vector_store
from ..infrastructure.embeddings.sentence_transformer import SentenceTransformerEmbedding
from ..infrastructure.cache import get_cache
from ..utils.config import get_config
from ..utils.logger import get_logger
from ..utils.document_chunker import DocumentChunker

logger = get_logger(__name__)


def create_mcp_app() -> FastAPI:
    """Create and configure MCP FastAPI app."""
    # Load configuration
    config = get_config()

    # Create FastAPI app
    app = FastAPI(
        title="SignalCLI MCP Server",
        description="Model Context Protocol server for SignalCLI",
        version="1.0.0",
    )

    # Initialize components
    llm_engine = get_llm_engine(config["llm"]["provider"])
    vector_store = get_vector_store(config["vector_store"]["provider"])
    embedding_model = SentenceTransformerEmbedding()
    cache = get_cache(config["cache"]["provider"])
    document_processor = DocumentChunker()

    # Create RAG pipeline
    rag_pipeline = RAGPipeline(
        llm_engine=llm_engine,
        vector_store=vector_store,
        embedding_model=embedding_model,
        cache=cache,
    )

    # Initialize MCP components
    tool_registry = ToolRegistry()
    permission_manager = PermissionManager()
    mcp_cache = MCPCache(
        max_size_mb=config.get("mcp", {}).get("cache_size_mb", 1024),
        persistence_path=config.get("mcp", {}).get("cache_path"),
    )

    # Register built-in tools
    tool_registry.register_tool(RAGQueryTool(rag_pipeline))
    tool_registry.register_tool(DocumentIndexTool(vector_store, document_processor))
    tool_registry.register_tool(SummarizationTool(llm_engine))

    # Create router
    router = ContextAwareRouter(
        tool_registry=tool_registry,
        embedding_model=embedding_model,
        strategy=RoutingStrategy.CONTEXT_AWARE,
    )

    # Create MCP server
    mcp_server = create_mcp_server(
        app=app,
        tool_registry=tool_registry,
        router=router,
        permission_manager=permission_manager,
        cache=mcp_cache,
    )

    # Add middleware
    @app.middleware("http")
    async def add_mcp_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-MCP-Version"] = "1.0.0"
        response.headers["X-MCP-Server"] = "SignalCLI"
        return response

    # Add startup/shutdown events
    @app.on_event("startup")
    async def startup():
        logger.info("MCP Server starting...")
        # Initialize components that need async setup
        await vector_store.initialize()
        logger.info("MCP Server started successfully")

    @app.on_event("shutdown")
    async def shutdown():
        logger.info("MCP Server shutting down...")
        # Cleanup
        await vector_store.close()
        logger.info("MCP Server shut down")

    return app


def main():
    """Run MCP server."""
    app = create_mcp_app()
    config = get_config()

    host = config.get("mcp", {}).get("host", "0.0.0.0")
    port = config.get("mcp", {}).get("port", 8001)
    workers = config.get("mcp", {}).get("workers", 1)

    logger.info(f"Starting MCP server on {host}:{port}")

    uvicorn.run(
        "src.mcp.main:create_mcp_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv("ENV") == "development",
    )


if __name__ == "__main__":
    main()
