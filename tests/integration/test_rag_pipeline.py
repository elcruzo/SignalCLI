"""Integration tests for RAG pipeline."""

import pytest

@pytest.mark.asyncio
async def test_rag_pipeline_without_context(rag_pipeline):
    """Test RAG pipeline without any documents."""
    response = await rag_pipeline.process(
        query="What is artificial intelligence?",
        max_tokens=100,
        temperature=0.7
    )
    
    assert "result" in response
    assert "answer" in response["result"]
    assert len(response["sources"]) == 0
    assert response["tokens_used"] > 0
    assert response["confidence"] == 0.5  # Default confidence without context

@pytest.mark.asyncio
async def test_rag_pipeline_with_context(rag_pipeline, sample_documents):
    """Test RAG pipeline with documents."""
    # Add documents to vector store
    await rag_pipeline.vector_store.add_documents(sample_documents)
    
    # Query about machine learning
    response = await rag_pipeline.process(
        query="What is machine learning?",
        max_tokens=200,
        temperature=0.7
    )
    
    assert "result" in response
    assert "answer" in response["result"]
    assert len(response["sources"]) > 0
    assert response["confidence"] > 0.5
    
    # Check metadata
    assert "retrieval_time_ms" in response["metadata"]
    assert "generation_time_ms" in response["metadata"]
    assert response["metadata"]["documents_retrieved"] > 0

@pytest.mark.asyncio
async def test_rag_pipeline_with_schema(rag_pipeline, sample_documents, sample_schema):
    """Test RAG pipeline with structured output."""
    await rag_pipeline.vector_store.add_documents(sample_documents)
    
    response = await rag_pipeline.process(
        query="Explain machine learning",
        schema=sample_schema,
        max_tokens=150,
        temperature=0.3
    )
    
    result = response["result"]
    assert "answer" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], (int, float))
    assert 0 <= result["confidence"] <= 1

@pytest.mark.asyncio
async def test_rag_reranking(rag_pipeline, sample_documents):
    """Test document reranking."""
    await rag_pipeline.vector_store.add_documents(sample_documents)
    
    # Query that matches multiple documents
    response = await rag_pipeline.process(
        query="machine learning programming python",
        max_tokens=100
    )
    
    # Sources should be reranked based on relevance
    assert len(response["sources"]) > 0
    sources = response["sources"]
    
    # Check that sources contain relevant content
    source_contents = " ".join(s["content"] for s in sources).lower()
    assert "machine learning" in source_contents or "python" in source_contents

@pytest.mark.asyncio
async def test_rag_context_length_limit(rag_pipeline, sample_documents):
    """Test context length limiting."""
    # Add many copies of documents to exceed context limit
    many_docs = sample_documents * 20
    for i, doc in enumerate(many_docs):
        doc.chunk_id = f"chunk_{i:03d}"
    
    await rag_pipeline.vector_store.add_documents(many_docs)
    
    response = await rag_pipeline.process(
        query="Tell me about machine learning",
        max_tokens=100
    )
    
    # Context should be limited
    assert response["metadata"]["context_length"] <= rag_pipeline.max_context_length

@pytest.mark.asyncio
async def test_rag_error_handling(rag_pipeline):
    """Test RAG pipeline error handling."""
    # Test with empty query
    response = await rag_pipeline.process(
        query="",
        max_tokens=50
    )
    
    # Should still return a response
    assert "result" in response
    assert response["tokens_used"] > 0