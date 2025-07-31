"""Unit tests for document chunker."""

import pytest

from src.utils.document_chunker import DocumentChunker, ChunkConfig


def test_basic_chunking():
    """Test basic text chunking."""
    chunker = DocumentChunker(ChunkConfig(chunk_size=50, chunk_overlap=10))

    text = "This is a test document. " * 10  # ~250 chars
    chunks = chunker.chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 50 for chunk in chunks)
    assert all(
        chunk.chunk_id.endswith(f"_chunk_{i:04d}") for i, chunk in enumerate(chunks)
    )


def test_chunking_with_separators():
    """Test chunking with paragraph separators."""
    chunker = DocumentChunker(
        ChunkConfig(chunk_size=100, chunk_overlap=20, separator="\n\n")
    )

    text = """First paragraph here.
    
Second paragraph here.

Third paragraph here."""

    chunks = chunker.chunk_text(text)
    assert len(chunks) == 3
    assert "First paragraph" in chunks[0].content
    assert "Second paragraph" in chunks[1].content
    assert "Third paragraph" in chunks[2].content


def test_chunking_with_metadata():
    """Test chunking preserves metadata."""
    chunker = DocumentChunker()
    metadata = {"source": "test.txt", "author": "Test Author"}

    chunks = chunker.chunk_text("Test content", metadata)

    assert all(chunk.source == "test.txt" for chunk in chunks)
    assert all(chunk.metadata["author"] == "Test Author" for chunk in chunks)
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_empty_text():
    """Test chunking empty text."""
    chunker = DocumentChunker()

    chunks = chunker.chunk_text("")
    assert len(chunks) == 0

    chunks = chunker.chunk_text("   ")
    assert len(chunks) == 0


def test_max_chunks_limit():
    """Test max chunks per document limit."""
    chunker = DocumentChunker(ChunkConfig(chunk_size=10, max_chunks_per_doc=3))

    text = "x" * 100  # Should create 10 chunks
    chunks = chunker.chunk_text(text)

    assert len(chunks) == 3  # Limited by max_chunks_per_doc


def test_chunk_overlap():
    """Test chunk overlap functionality."""
    chunker = DocumentChunker(
        ChunkConfig(chunk_size=20, chunk_overlap=5, separator=" ")
    )

    text = "one two three four five six seven eight nine ten"
    chunks = chunker.chunk_text(text)

    # Check that chunks have overlap
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        # Last 5 chars of chunk i should appear in chunk i+1
        assert chunks[i].content[-5:] in chunks[i + 1].content


def test_chunk_documents_batch():
    """Test chunking multiple documents."""
    chunker = DocumentChunker()

    documents = [
        {"content": "Document one content", "metadata": {"id": 1}},
        {"content": "Document two content", "metadata": {"id": 2}},
        {"content": "Document three content", "metadata": {"id": 3}},
    ]

    all_chunks = chunker.chunk_documents(documents)

    assert len(all_chunks) == 3
    assert all_chunks[0].metadata["id"] == 1
    assert all_chunks[1].metadata["id"] == 2
    assert all_chunks[2].metadata["id"] == 3
