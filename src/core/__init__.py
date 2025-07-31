"""Core domain logic for SignalCLI."""

from .interfaces import (
    Document,
    LLMResponse,
    ILLMEngine,
    IVectorStore,
    ICache,
    IEmbeddingModel,
)
from .exceptions import (
    SignalCLIError,
    ConfigurationError,
    LLMError,
    VectorStoreError,
    ValidationError,
    CacheError,
    RAGPipelineError,
    JSONFormerError,
)

__all__ = [
    "Document",
    "LLMResponse",
    "ILLMEngine",
    "IVectorStore",
    "ICache",
    "IEmbeddingModel",
    "SignalCLIError",
    "ConfigurationError",
    "LLMError",
    "VectorStoreError",
    "ValidationError",
    "CacheError",
    "RAGPipelineError",
    "JSONFormerError",
]
