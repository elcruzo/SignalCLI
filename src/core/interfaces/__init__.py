"""Core interfaces for SignalCLI."""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Document:
    """Represents a document in the vector store."""
    content: str
    source: str
    chunk_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

@dataclass
class LLMResponse:
    """Response from LLM engine."""
    text: str
    tokens_used: int
    model_name: str
    latency_ms: int
    metadata: Dict[str, Any]

class ILLMEngine(Protocol):
    """Interface for LLM engines."""
    
    async def initialize(self) -> None:
        """Initialize the LLM engine."""
        ...
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from prompt."""
        ...
    
    async def health_check(self) -> bool:
        """Check if engine is healthy."""
        ...
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        ...

class IVectorStore(Protocol):
    """Interface for vector stores."""
    
    async def initialize(self) -> None:
        """Initialize vector store connection."""
        ...
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Document]:
        """Search for similar documents."""
        ...
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        ...
    
    async def delete_documents(self, chunk_ids: List[str]) -> None:
        """Delete documents by chunk IDs."""
        ...
    
    async def health_check(self) -> bool:
        """Check if store is healthy."""
        ...
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        ...

class ICache(Protocol):
    """Interface for caching."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...
    
    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        ...
    
    async def clear(self) -> None:
        """Clear all cache entries."""
        ...
    
    async def health_check(self) -> bool:
        """Check if cache is healthy."""
        ...

class IEmbeddingModel(Protocol):
    """Interface for embedding models."""
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        ...
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        ...