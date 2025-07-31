"""In-memory vector store for testing."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseVectorStore
from src.core.interfaces import Document, IEmbeddingModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store for development and testing."""
    
    def __init__(self, config: Dict[str, Any], embedding_model: IEmbeddingModel):
        super().__init__(config, embedding_model)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None
        
    async def _connect(self) -> None:
        """No connection needed for in-memory store."""
        logger.info("Using in-memory vector store")
        
    async def _ensure_collection(self) -> None:
        """No collection setup needed."""
        pass
        
    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform vector search in memory."""
        if not self.documents or self.embeddings is None:
            return []
            
        # Calculate similarities
        query_vec = np.array(query_embedding).reshape(1, -1)
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter by threshold and prepare results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= threshold:
                doc = self.documents[idx]
                results.append({
                    'content': doc.content,
                    'source': doc.source,
                    'chunk_id': doc.chunk_id,
                    'metadata': doc.metadata,
                    'score': score
                })
                
        return results
        
    async def _add_vectors(self, documents: List[Document]) -> None:
        """Add vectors to memory."""
        # Add documents
        self.documents.extend(documents)
        
        # Update embeddings matrix
        new_embeddings = np.array([doc.embedding for doc in documents])
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        logger.debug(f"Added {len(documents)} documents to memory store")
        
    async def delete_documents(self, chunk_ids: List[str]) -> None:
        """Delete documents by chunk IDs."""
        if not self.documents:
            return
            
        # Find indices to delete
        indices_to_delete = []
        for i, doc in enumerate(self.documents):
            if doc.chunk_id in chunk_ids:
                indices_to_delete.append(i)
                
        # Remove documents and embeddings
        for idx in sorted(indices_to_delete, reverse=True):
            self.documents.pop(idx)
            
        if indices_to_delete and self.embeddings is not None:
            mask = np.ones(len(self.embeddings), dtype=bool)
            mask[indices_to_delete] = False
            self.embeddings = self.embeddings[mask]
            
        logger.info(f"Deleted {len(indices_to_delete)} documents")
        
    async def _health_check_impl(self) -> None:
        """Always healthy for memory store."""
        pass
        
    async def _disconnect(self) -> None:
        """Clear memory."""
        self.documents.clear()
        self.embeddings = None