"""Weaviate vector store implementation."""

import asyncio
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.exceptions import WeaviateException

from .base import BaseVectorStore
from src.core.interfaces import Document, IEmbeddingModel
from src.core.exceptions import VectorStoreError, VectorStoreConnectionError
from src.utils.logger import get_logger

logger = get_logger(__name__)

class WeaviateVectorStore(BaseVectorStore):
    """Vector store implementation using Weaviate."""
    
    def __init__(self, config: Dict[str, Any], embedding_model: IEmbeddingModel):
        super().__init__(config, embedding_model)
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 8080)
        self.client: Optional[weaviate.Client] = None
        
    async def _connect(self) -> None:
        """Connect to Weaviate."""
        try:
            # Create client
            self.client = weaviate.Client(
                url=f"http://{self.host}:{self.port}",
                timeout_config=(5, 15)  # (connect, read) timeouts
            )
            
            # Test connection
            if not self.client.is_ready():
                raise VectorStoreConnectionError("Weaviate is not ready")
                
            logger.info(f"Connected to Weaviate at {self.host}:{self.port}")
            
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to connect to Weaviate: {e}")
    
    async def _ensure_collection(self) -> None:
        """Ensure collection exists in Weaviate."""
        try:
            # Check if class exists
            schema = self.client.schema.get()
            class_names = [c['class'] for c in schema.get('classes', [])]
            
            if self.collection_name not in class_names:
                # Create class
                class_obj = {
                    "class": self.collection_name,
                    "description": "Document collection for SignalCLI",
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Document content"
                        },
                        {
                            "name": "source",
                            "dataType": ["string"],
                            "description": "Document source"
                        },
                        {
                            "name": "chunk_id",
                            "dataType": ["string"],
                            "description": "Unique chunk identifier"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata"
                        }
                    ],
                    "vectorizer": "none",  # We provide our own embeddings
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "skip": True
                        }
                    }
                }
                
                self.client.schema.create_class(class_obj)
                logger.info(f"Created Weaviate class: {self.collection_name}")
                
        except WeaviateException as e:
            raise VectorStoreError(f"Failed to ensure collection: {e}")
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        top_k: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform vector search in Weaviate."""
        try:
            # Build query
            near_vector = {
                "vector": query_embedding,
                "certainty": threshold
            }
            
            # Execute search
            results = (
                self.client.query
                .get(self.collection_name, ["content", "source", "chunk_id", "metadata"])
                .with_near_vector(near_vector)
                .with_limit(top_k)
                .with_additional(["certainty"])
                .do()
            )
            
            # Extract results
            documents = []
            data = results.get('data', {}).get('Get', {}).get(self.collection_name, [])
            
            for item in data:
                documents.append({
                    'content': item.get('content', ''),
                    'source': item.get('source', ''),
                    'chunk_id': item.get('chunk_id', ''),
                    'metadata': item.get('metadata', {}),
                    'score': item.get('_additional', {}).get('certainty', 0.0)
                })
                
            return documents
            
        except WeaviateException as e:
            raise VectorStoreError(f"Vector search failed: {e}")
    
    async def _add_vectors(self, documents: List[Document]) -> None:
        """Add vectors to Weaviate."""
        try:
            # Prepare batch
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for doc in documents:
                    properties = {
                        "content": doc.content,
                        "source": doc.source,
                        "chunk_id": doc.chunk_id,
                        "metadata": doc.metadata
                    }
                    
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        vector=doc.embedding
                    )
                    
            logger.debug(f"Added {len(documents)} documents to Weaviate")
            
        except WeaviateException as e:
            raise VectorStoreError(f"Failed to add vectors: {e}")
    
    async def delete_documents(self, chunk_ids: List[str]) -> None:
        """Delete documents by chunk IDs."""
        if not self._initialized:
            await self.initialize()
            
        try:
            for chunk_id in chunk_ids:
                where = {
                    "path": ["chunk_id"],
                    "operator": "Equal",
                    "valueString": chunk_id
                }
                
                self.client.batch.delete_objects(
                    class_name=self.collection_name,
                    where=where
                )
                
            logger.info(f"Deleted {len(chunk_ids)} documents")
            
        except WeaviateException as e:
            raise VectorStoreError(f"Failed to delete documents: {e}")
    
    async def _health_check_impl(self) -> None:
        """Check Weaviate health."""
        if not self.client or not self.client.is_ready():
            raise VectorStoreError("Weaviate is not ready")
    
    async def _disconnect(self) -> None:
        """Disconnect from Weaviate."""
        # Weaviate client doesn't need explicit disconnection
        self.client = None