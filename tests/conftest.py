"""Pytest configuration and fixtures."""

import pytest
import asyncio
from typing import Dict, Any

from src.infrastructure.llm.mock_engine import MockLLMEngine
from src.infrastructure.vector_store.memory_store import MemoryVectorStore
from src.infrastructure.embeddings.sentence_transformer import MockEmbedding
from src.application.rag.pipeline import RAGPipeline
from src.application.jsonformer.validator import JSONValidator

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Mock configuration for testing."""
    return {
        "api": {
            "host": "0.0.0.0",
            "port": 8000
        },
        "llm": {
            "model_type": "mock",
            "max_tokens": 2048,
            "temperature": 0.7
        },
        "vector_store": {
            "provider": "memory",
            "collection_name": "test_documents",
            "embedding_model": "mock"
        },
        "rag": {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_context_length": 2000,
            "reranking": True
        },
        "jsonformer": {
            "max_string_length": 500,
            "max_array_length": 20,
            "max_object_properties": 50
        },
        "observability": {
            "metrics_enabled": True
        }
    }

@pytest.fixture
async def mock_llm_engine(mock_config):
    """Mock LLM engine fixture."""
    engine = MockLLMEngine(mock_config["llm"])
    await engine.initialize()
    yield engine
    await engine.shutdown()

@pytest.fixture
async def mock_embedding_model():
    """Mock embedding model fixture."""
    return MockEmbedding(dimension=384)

@pytest.fixture
async def mock_vector_store(mock_config, mock_embedding_model):
    """Mock vector store fixture."""
    store = MemoryVectorStore(mock_config["vector_store"], mock_embedding_model)
    await store.initialize()
    yield store
    await store.shutdown()

@pytest.fixture
async def rag_pipeline(mock_llm_engine, mock_vector_store, mock_config):
    """RAG pipeline fixture."""
    return RAGPipeline(
        llm_engine=mock_llm_engine,
        vector_store=mock_vector_store,
        config=mock_config["rag"]
    )

@pytest.fixture
def json_validator(mock_config):
    """JSON validator fixture."""
    return JSONValidator(mock_config["jsonformer"])

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    from src.core.interfaces import Document
    
    return [
        Document(
            content="Machine learning is a subset of artificial intelligence that focuses on training algorithms to make predictions.",
            source="ml_basics.txt",
            chunk_id="chunk_001",
            metadata={"topic": "machine_learning"}
        ),
        Document(
            content="Python is a popular programming language for data science and machine learning applications.",
            source="python_intro.txt",
            chunk_id="chunk_002",
            metadata={"topic": "programming"}
        ),
        Document(
            content="Neural networks are inspired by the structure and function of the human brain.",
            source="neural_nets.txt",
            chunk_id="chunk_003",
            metadata={"topic": "deep_learning"}
        )
    ]

@pytest.fixture
def sample_schema():
    """Sample JSON schema for testing."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "topics": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["answer", "confidence"]
    }