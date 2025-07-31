"""Unit tests for LLM engine."""

import pytest

from src.core.interfaces import LLMResponse

@pytest.mark.asyncio
async def test_mock_llm_initialization(mock_llm_engine):
    """Test mock LLM engine initialization."""
    assert mock_llm_engine._initialized
    assert mock_llm_engine.model_name == "mock-llm-v1"

@pytest.mark.asyncio
async def test_mock_llm_generation(mock_llm_engine):
    """Test basic text generation."""
    response = await mock_llm_engine.generate(
        prompt="Hello, how are you?",
        max_tokens=50,
        temperature=0.7
    )
    
    assert isinstance(response, LLMResponse)
    assert len(response.text) > 0
    assert response.tokens_used > 0
    assert response.model_name == "mock-llm-v1"
    assert response.latency_ms > 0
    assert response.metadata["mock"] is True

@pytest.mark.asyncio
async def test_mock_llm_structured_generation(mock_llm_engine, sample_schema):
    """Test structured output generation."""
    response = await mock_llm_engine.generate(
        prompt="What is machine learning?",
        max_tokens=100,
        temperature=0.3,
        schema=sample_schema
    )
    
    assert isinstance(response, LLMResponse)
    # Response should be valid JSON
    import json
    parsed = json.loads(response.text)
    assert "answer" in parsed
    assert "confidence" in parsed

@pytest.mark.asyncio
async def test_mock_llm_health_check(mock_llm_engine):
    """Test health check functionality."""
    is_healthy = await mock_llm_engine.health_check()
    assert is_healthy is True

@pytest.mark.asyncio
async def test_mock_llm_shutdown(mock_llm_engine):
    """Test shutdown functionality."""
    await mock_llm_engine.shutdown()
    assert not mock_llm_engine._initialized