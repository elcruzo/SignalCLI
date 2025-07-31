"""Unit tests for vector store."""

import pytest

from src.core.interfaces import Document


@pytest.mark.asyncio
async def test_memory_store_initialization(mock_vector_store):
    """Test memory vector store initialization."""
    assert mock_vector_store._initialized
    assert len(mock_vector_store.documents) == 0


@pytest.mark.asyncio
async def test_add_documents(mock_vector_store, sample_documents):
    """Test adding documents to store."""
    await mock_vector_store.add_documents(sample_documents)

    assert len(mock_vector_store.documents) == len(sample_documents)
    assert mock_vector_store.embeddings is not None
    assert mock_vector_store.embeddings.shape[0] == len(sample_documents)


@pytest.mark.asyncio
async def test_search_documents(mock_vector_store, sample_documents):
    """Test searching for documents."""
    # Add documents first
    await mock_vector_store.add_documents(sample_documents)

    # Search for related content
    results = await mock_vector_store.search(
        query="machine learning", top_k=2, threshold=0.0
    )

    assert len(results) <= 2
    assert all(isinstance(doc, Document) for doc in results)
    assert all(doc.similarity_score is not None for doc in results)

    # Most relevant should be the ML document
    if results:
        assert "machine learning" in results[0].content.lower()


@pytest.mark.asyncio
async def test_delete_documents(mock_vector_store, sample_documents):
    """Test deleting documents."""
    # Add documents
    await mock_vector_store.add_documents(sample_documents)
    initial_count = len(mock_vector_store.documents)

    # Delete one document
    await mock_vector_store.delete_documents(["chunk_002"])

    assert len(mock_vector_store.documents) == initial_count - 1
    assert all(doc.chunk_id != "chunk_002" for doc in mock_vector_store.documents)


@pytest.mark.asyncio
async def test_empty_search(mock_vector_store):
    """Test searching empty store."""
    results = await mock_vector_store.search("test query", top_k=5)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_threshold_filtering(mock_vector_store, sample_documents):
    """Test similarity threshold filtering."""
    await mock_vector_store.add_documents(sample_documents)

    # High threshold should return fewer results
    results = await mock_vector_store.search(
        query="programming", top_k=10, threshold=0.9
    )

    # With mock embeddings, scores might be low
    assert all(doc.similarity_score >= 0.9 for doc in results if doc.similarity_score)
