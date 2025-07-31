"""Sentence transformer embedding model."""

import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from src.core.interfaces import IEmbeddingModel
from src.core.exceptions import VectorStoreError
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed")

class SentenceTransformerEmbedding(IEmbeddingModel):
    """Embedding model using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise VectorStoreError("sentence-transformers not installed")
            
        self.model_name = model_name
        self.model = None
        self._dimension = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        
    def _initialize_model(self):
        """Initialize the model if needed."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        self._initialize_model()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self._executor,
            lambda: self.model.encode(text, convert_to_tensor=False)
        )
        
        return embedding.tolist()
        
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        self._initialize_model()
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self._executor,
            lambda: self.model.encode(texts, convert_to_tensor=False)
        )
        
        return embeddings.tolist()
        
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._initialize_model()
        return self._dimension

class MockEmbedding(IEmbeddingModel):
    """Mock embedding model for testing."""
    
    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        
    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embedding."""
        # Simple hash-based embedding for consistency
        import hashlib
        
        hash_object = hashlib.md5(text.encode())
        hash_hex = hash_object.hexdigest()
        
        # Convert to floats
        embedding = []
        for i in range(0, len(hash_hex), 2):
            value = int(hash_hex[i:i+2], 16) / 255.0
            embedding.append(value)
            
        # Pad or truncate to dimension
        if len(embedding) < self._dimension:
            embedding.extend([0.0] * (self._dimension - len(embedding)))
        else:
            embedding = embedding[:self._dimension]
            
        return embedding
        
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
        
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self._dimension

def create_embedding_model(config: Dict[str, Any]) -> IEmbeddingModel:
    """Create embedding model based on config."""
    model_name = config.get('embedding_model', 'all-MiniLM-L6-v2')
    
    if model_name == 'mock':
        return MockEmbedding()
    else:
        return SentenceTransformerEmbedding(model_name)