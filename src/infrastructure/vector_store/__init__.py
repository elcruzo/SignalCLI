"""Vector store implementations."""

from typing import Dict, Any

from src.core.interfaces import IVectorStore, IEmbeddingModel
from src.core.exceptions import VectorStoreError
from .weaviate_store import WeaviateVectorStore
from .memory_store import MemoryVectorStore


def create_vector_store(
    config: Dict[str, Any], embedding_model: IEmbeddingModel
) -> IVectorStore:
    """
    Factory function to create vector stores.

    Args:
        config: Vector store configuration
        embedding_model: Embedding model to use

    Returns:
        Vector store instance

    Raises:
        VectorStoreError: If store type is unknown
    """
    provider = config.get("provider", "weaviate")

    if provider == "weaviate":
        return WeaviateVectorStore(config, embedding_model)
    elif provider == "memory":
        return MemoryVectorStore(config, embedding_model)
    else:
        raise VectorStoreError(f"Unknown vector store provider: {provider}")


__all__ = ["create_vector_store", "WeaviateVectorStore", "MemoryVectorStore"]
