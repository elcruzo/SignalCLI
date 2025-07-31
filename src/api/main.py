"""
SignalCLI FastAPI Server
Main API application
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import uvicorn
from typing import Dict, Any, Optional

from .models import QueryRequest, QueryResponse, HealthResponse
from .dependencies import get_rag_engine, get_llm_engine, get_observability
from ..utils.config import load_config
from ..utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

# Global state
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting SignalCLI API server")

    # Initialize services
    try:
        # This would initialize your RAG engine, LLM, etc.
        logger.info("✅ Services initialized successfully")
        yield
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    finally:
        logger.info("🔄 Shutting down SignalCLI API server")


# Create FastAPI app
app = FastAPI(
    title="SignalCLI API",
    description="LLM-Powered Knowledge CLI with RAG and structured JSON output",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        services={
            "api": "healthy",
            "vector_store": "healthy",  # Would check actual status
            "llm_engine": "healthy",  # Would check actual status
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    rag_engine=Depends(get_rag_engine),
    llm_engine=Depends(get_llm_engine),
    observability=Depends(get_observability),
):
    """
    Main query endpoint for LLM with RAG
    """
    start_time = time.time()
    query_id = f"query_{int(start_time * 1000)}"

    try:
        # Log the incoming request
        logger.info(
            f"Processing query: {query_id}",
            extra={
                "query_id": query_id,
                "query_length": len(request.query),
                "has_schema": request.schema is not None,
            },
        )

        # Step 1: Retrieve relevant context
        context_docs = await rag_engine.retrieve(
            request.query, top_k=request.top_k_retrieval
        )

        # Step 2: Generate response with LLM
        response = await llm_engine.generate(
            query=request.query,
            context=context_docs,
            schema=request.schema,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        # Step 3: Calculate metrics
        latency_ms = int((time.time() - start_time) * 1000)

        # Log completion
        logger.info(
            f"Query completed: {query_id}",
            extra={
                "query_id": query_id,
                "latency_ms": latency_ms,
                "tokens_used": response.get("tokens_used", 0),
                "success": True,
            },
        )

        return QueryResponse(
            query_id=query_id,
            result=response["result"],
            metadata={
                "tokens_used": response.get("tokens_used", 0),
                "latency_ms": latency_ms,
                "model_name": response.get("model_name", "unknown"),
                "confidence_score": response.get("confidence", 0.0),
            },
            sources=response.get("sources", []),
        )

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)

        # Log the error
        logger.error(
            f"Query failed: {query_id}",
            extra={
                "query_id": query_id,
                "error": str(e),
                "latency_ms": latency_ms,
                "success": False,
            },
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Query processing failed",
                "query_id": query_id,
                "message": str(e),
            },
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics endpoint"""
    # This would return actual metrics
    return {
        "queries_total": 0,
        "queries_success": 0,
        "queries_failed": 0,
        "average_latency_ms": 0,
        "tokens_processed": 0,
    }


@app.get("/schemas")
async def list_schemas():
    """List available JSON schemas"""
    # This would return actual schema registry
    return {"schemas": ["qa_response", "list_response", "comparison_response"]}


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=True,
        log_level="info",
    )
