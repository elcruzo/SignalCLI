"""Base vector store implementation."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from src.core.interfaces import IVectorStore, Document, IEmbeddingModel
from src.core.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class BaseVectorStore(ABC, IVectorStore):
    """Base class for vector stores."""
    
    def __init__(self, config: Dict[str, Any], embedding_model: IEmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self.collection_name = config.get('collection_name', 'documents')
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize vector store connection."""
        if self._initialized:
            return
            
        try:
            await self._connect()
            await self._ensure_collection()
            self._initialized = True
            logger.info(f"Vector store initialized: {self.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise VectorStoreError(f"Initialization failed: {e}")
    
    @abstractmethod
    async def _connect(self) -> None:
        """Connect to the vector store."""
        pass
    
    @abstractmethod
    async def _ensure_collection(self) -> None:
        """Ensure collection exists."""
        pass
    
    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Document]:
        """Search for similar documents."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate query embedding
            query_embedding = await self.embedding_model.embed_text(query)
            
            # Search in vector store
            results = await self._vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=threshold
            )
            
            # Convert to Document objects
            documents = []
            for result in results:
                doc = Document(
                    content=result['content'],
                    source=result['source'],
                    chunk_id=result['chunk_id'],
                    metadata=result.get('metadata', {}),
                    similarity_score=result.get('score', 0.0)
                )
                documents.append(doc)
                
            logger.debug(f"Found {len(documents)} documents for query")
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    @abstractmethod
    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform vector search."""
        pass
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the store."""
        if not self._initialized:
            await self.initialize()
            
        if not documents:
            return
            
        try:
            # Generate embeddings for documents without them
            for doc in documents:
                if doc.embedding is None:
                    doc.embedding = await self.embedding_model.embed_text(doc.content)
            
            # Add to vector store
            await self._add_vectors(documents)
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to add documents: {e}")
    
    @abstractmethod
    async def _add_vectors(self, documents: List[Document]) -> None:
        """Add vectors to the store."""
        pass
    
    async def health_check(self) -> bool:
        """Check if store is healthy."""
        try:
            if not self._initialized:
                await self.initialize()
                
            # Try a simple operation
            await self._health_check_impl()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    @abstractmethod
    async def _health_check_impl(self) -> None:
        """Health check implementation."""
        pass
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._initialized:
            await self._disconnect()
            self._initialized = False
            logger.info("Vector store shut down")
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Disconnect from the store."""
        pass