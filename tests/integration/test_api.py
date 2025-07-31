"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.api.main import app
from src.api.models import QueryResponse, HealthResponse

@pytest.fixture
def test_client():
    """Create test client."""
    return TestClient(app)

def test_health_endpoint(test_client):
    """Test health check endpoint."""
    response = test_client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "services" in data

@patch('src.api.dependencies.get_rag_engine')
@patch('src.api.dependencies.get_observability')
def test_query_endpoint(mock_observability, mock_rag_engine, test_client):
    """Test query endpoint."""
    # Mock RAG engine response
    mock_rag = AsyncMock()
    mock_rag.process.return_value = {
        "result": {"answer": "Test response"},
        "sources": [],
        "tokens_used": 50,
        "model_name": "mock-llm",
        "confidence": 0.9,
        "metadata": {
            "retrieval_time_ms": 10,
            "generation_time_ms": 100,
            "total_time_ms": 110
        }
    }
    mock_rag_engine.return_value = mock_rag
    
    # Mock observability
    mock_obs = AsyncMock()
    mock_observability.return_value = mock_obs
    
    # Make request
    response = test_client.post("/query", json={
        "query": "What is machine learning?",
        "max_tokens": 100,
        "temperature": 0.7
    })
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "query_id" in data
    assert "result" in data
    assert data["result"]["answer"] == "Test response"
    assert "metadata" in data
    assert data["metadata"]["tokens_used"] == 50

def test_query_validation_error(test_client):
    """Test query validation."""
    # Empty query
    response = test_client.post("/query", json={
        "query": "",
        "max_tokens": 100
    })
    
    assert response.status_code == 422  # Validation error
    
    # Invalid temperature
    response = test_client.post("/query", json={
        "query": "Test query",
        "temperature": 2.0  # Out of range
    })
    
    assert response.status_code == 422

def test_metrics_endpoint(test_client):
    """Test metrics endpoint."""
    response = test_client.get("/metrics")
    
    assert response.status_code == 200
    data = response.json()
    assert "queries_total" in data
    assert "average_latency_ms" in data

def test_schemas_endpoint(test_client):
    """Test schemas listing endpoint."""
    response = test_client.get("/schemas")
    
    assert response.status_code == 200
    data = response.json()
    assert "schemas" in data
    assert isinstance(data["schemas"], list)