"""Document chunking utilities for RAG."""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

from src.core.interfaces import Document
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ChunkConfig:
    """Configuration for document chunking."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_chunks_per_doc: int = 100
    separator: str = "\n\n"
    secondary_separator: str = "\n"
    keep_separator: bool = True

class DocumentChunker:
    """Splits documents into chunks for vector storage."""
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig()
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of Document chunks
        """
        if not text.strip():
            return []
            
        # Split by primary separator first
        splits = self._split_text(text, self.config.separator)
        
        # Further split if needed
        chunks = []
        for split in splits:
            if len(split) <= self.config.chunk_size:
                chunks.append(split)
            else:
                # Split by secondary separator
                sub_splits = self._split_text(split, self.config.secondary_separator)
                chunks.extend(self._merge_splits(sub_splits))
                
        # Create Document objects
        documents = []
        source = metadata.get('source', 'unknown') if metadata else 'unknown'
        
        for i, chunk in enumerate(chunks[:self.config.max_chunks_per_doc]):
            doc = Document(
                content=chunk.strip(),
                source=source,
                chunk_id=f"{source}_chunk_{i:04d}",
                metadata={
                    **(metadata or {}),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            )
            documents.append(doc)
            
        logger.debug(f"Created {len(documents)} chunks from {len(text)} chars")
        return documents
        
    def _split_text(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        if self.config.keep_separator:
            # Keep separator at end of each split
            splits = text.split(separator)
            return [s + separator if i < len(splits) - 1 else s 
                   for i, s in enumerate(splits) if s]
        else:
            return [s for s in text.split(separator) if s]
            
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge small splits to reach target chunk size."""
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) <= self.config.chunk_size:
                current_chunk += split
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    
                # Add overlap from previous chunk
                if chunks and self.config.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.config.chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split
                    
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
        
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Document]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of document dicts with 'content' and 'metadata'
            
        Returns:
            List of all chunks
        """
        all_chunks = []
        
        for doc in documents:
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            chunks = self.chunk_text(content, metadata)
            all_chunks.extend(chunks)
            
        return all_chunks