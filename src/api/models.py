"""
Pydantic models for SignalCLI API
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional
from datetime import datetime


class QueryRequest(BaseModel):
    """Request model for query endpoint"""

    query: str = Field(..., min_length=1, max_length=2048, description="User query")
    schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON schema for structured output"
    )
    max_tokens: int = Field(
        2048, ge=1, le=4096, description="Maximum tokens in response"
    )
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="LLM temperature")
    top_k_retrieval: int = Field(
        5, ge=1, le=20, description="Number of documents to retrieve"
    )
    include_sources: bool = Field(
        True, description="Include source documents in response"
    )

    @validator("query")
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class TokenUsage(BaseModel):
    """Token usage information"""

    input_tokens: int = Field(..., description="Input tokens count")
    output_tokens: int = Field(..., description="Output tokens count")
    total_tokens: int = Field(..., description="Total tokens used")
    estimated_cost: float = Field(0.0, description="Estimated cost in USD")


class ResponseMetadata(BaseModel):
    """Metadata for query response"""

    tokens_used: int = Field(..., description="Total tokens used")
    latency_ms: int = Field(..., description="Response latency in milliseconds")
    model_name: str = Field(..., description="Name of the model used")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Response confidence"
    )
    retrieval_time_ms: Optional[int] = Field(
        None, description="Time spent on retrieval"
    )
    inference_time_ms: Optional[int] = Field(
        None, description="Time spent on inference"
    )


class SourceDocument(BaseModel):
    """Source document information"""

    content: str = Field(..., description="Document content")
    source: str = Field(..., description="Document source/filename")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint"""

    query_id: str = Field(..., description="Unique query identifier")
    result: Dict[str, Any] = Field(..., description="Structured query result")
    metadata: ResponseMetadata = Field(..., description="Response metadata")
    sources: List[SourceDocument] = Field(
        default_factory=list, description="Source documents"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    services: Dict[str, str] = Field(..., description="Individual service statuses")


class ErrorResponse(BaseModel):
    """Error response model"""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    query_id: Optional[str] = Field(None, description="Query ID if available")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class MetricsResponse(BaseModel):
    """Metrics response model"""

    queries_total: int = Field(..., description="Total queries processed")
    queries_success: int = Field(..., description="Successful queries")
    queries_failed: int = Field(..., description="Failed queries")
    average_latency_ms: float = Field(..., description="Average response latency")
    tokens_processed: int = Field(..., description="Total tokens processed")
    uptime_seconds: float = Field(..., description="Service uptime")


class SchemaInfo(BaseModel):
    """JSON schema information"""

    name: str = Field(..., description="Schema name")
    description: str = Field(..., description="Schema description")
    schema: Dict[str, Any] = Field(..., description="JSON schema definition")
    examples: List[Dict[str, Any]] = Field(
        default_factory=list, description="Example outputs"
    )


class SchemasResponse(BaseModel):
    """Available schemas response"""

    schemas: List[SchemaInfo] = Field(..., description="Available JSON schemas")
