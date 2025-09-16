"""Document chunking for RAG processing."""

import re
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

from src.core.interfaces import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChunkingStrategy(Enum):
    """Different chunking strategies."""
    FIXED_SIZE = "fixed_size"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    SEMANTIC = "semantic"


@dataclass
class ChunkConfig:
    """Configuration for chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BOUNDARY
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    respect_word_boundaries: bool = True
    preserve_formatting: bool = False


class DocumentChunker:
    """Chunks documents for vector storage and retrieval."""

    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_text(self, text: str, source: str = "unknown") -> List[Document]:
        """
        Chunk text into smaller documents.
        
        Args:
            text: Input text to chunk
            source: Source identifier
            
        Returns:
            List of Document chunks
        """
        if not text.strip():
            return []

        chunks = self._apply_chunking_strategy(text)
        documents = []

        for i, chunk in enumerate(chunks):
            if len(chunk) < self.config.min_chunk_size:
                # Skip chunks that are too small
                continue

            doc = Document(
                content=chunk,
                source=source,
                chunk_id=f"{source}#{i}",
                metadata={
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_size": len(chunk),
                    "strategy": self.config.strategy.value,
                }
            )
            documents.append(doc)

        logger.info(f"Created {len(documents)} chunks from {source}")
        return documents

    def _apply_chunking_strategy(self, text: str) -> List[str]:
        """Apply the configured chunking strategy."""
        if self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text)
        elif self.config.strategy == ChunkingStrategy.SENTENCE_BOUNDARY:
            return self._chunk_sentence_boundary(text)
        elif self.config.strategy == ChunkingStrategy.PARAGRAPH_BOUNDARY:
            return self._chunk_paragraph_boundary(text)
        elif self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

    def _chunk_fixed_size(self, text: str) -> List[str]:
        """Chunk text into fixed-size pieces."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            
            # Respect word boundaries if enabled
            if self.config.respect_word_boundaries and end < len(text):
                # Find the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start with overlap
            start = max(start + 1, end - self.config.chunk_overlap)
        
        return chunks

    def _chunk_sentence_boundary(self, text: str) -> List[str]:
        """Chunk text at sentence boundaries."""
        # Split into sentences using regex
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If single sentence is too long, split it
                if len(sentence) > self.config.chunk_size:
                    sub_chunks = self._chunk_fixed_size(sentence)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._add_overlap(chunks)

    def _chunk_paragraph_boundary(self, text: str) -> List[str]:
        """Chunk text at paragraph boundaries."""
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.config.chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph is too long, split by sentences
                if len(paragraph) > self.config.chunk_size:
                    sub_chunks = self._chunk_sentence_boundary(paragraph)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._add_overlap(chunks)

    def _chunk_semantic(self, text: str) -> List[str]:
        """Chunk text using semantic boundaries (simplified implementation)."""
        # This is a simplified semantic chunking
        # In practice, you'd use NLP libraries like spaCy or sentence transformers
        
        # For now, use sentence boundary with semantic markers
        semantic_markers = [
            r'\n#+\s+',  # Markdown headers
            r'\n\*\s+',  # Bullet points
            r'\n\d+\.\s+',  # Numbered lists
            r'\nHowever,',  # Transition words
            r'\nFurthermore,',
            r'\nIn addition,',
            r'\nOn the other hand,',
        ]
        
        # Split by semantic markers first
        chunks = [text]
        for marker in semantic_markers:
            new_chunks = []
            for chunk in chunks:
                parts = re.split(marker, chunk)
                new_chunks.extend(parts)
            chunks = new_chunks
        
        # Filter and size chunks
        result = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            if len(chunk) <= self.config.chunk_size:
                result.append(chunk)
            else:
                # Fall back to sentence boundary for large chunks
                sub_chunks = self._chunk_sentence_boundary(chunk)
                result.extend(sub_chunks)
        
        return result

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if self.config.chunk_overlap <= 0 or len(chunks) <= 1:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # Take last N characters from previous chunk
            overlap_text = prev_chunk[-self.config.chunk_overlap:].strip()
            
            # Add to beginning of current chunk
            if overlap_text:
                overlapped_chunk = overlap_text + " " + current_chunk
                overlapped_chunks.append(overlapped_chunk)
            else:
                overlapped_chunks.append(current_chunk)
        
        return overlapped_chunks

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk multiple documents."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc.content, doc.source)
            
            # Preserve original metadata
            for chunk in chunks:
                chunk.metadata.update(doc.metadata)
            
            all_chunks.extend(chunks)
        
        return all_chunks