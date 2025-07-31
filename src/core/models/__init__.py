"""Core domain models for SignalCLI."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ModelProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LLAMACPP = "llamacpp"
    MOCK = "mock"


class VectorStoreProvider(Enum):
    """Supported vector store providers."""

    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MEMORY = "memory"


@dataclass
class QueryContext:
    """Context for a query."""

    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class QueryResult:
    """Result of a query."""

    answer: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None


@dataclass
class DocumentChunk:
    """A chunk of a document."""

    content: str
    chunk_id: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


__all__ = [
    "ModelProvider",
    "VectorStoreProvider",
    "QueryContext",
    "QueryResult",
    "DocumentChunk",
]
