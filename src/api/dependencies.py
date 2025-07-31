"""FastAPI dependencies for dependency injection."""

from typing import Optional
from functools import lru_cache

from src.core.interfaces import ILLMEngine, IVectorStore
from src.infrastructure.llm import create_llm_engine
from src.infrastructure.vector_store import create_vector_store
from src.infrastructure.embeddings.sentence_transformer import create_embedding_model
from src.application.rag.pipeline import RAGPipeline
from src.application.jsonformer.validator import JSONValidator
from src.infrastructure.observability.metrics import MetricsCollector
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Singleton instances
_llm_engine: Optional[ILLMEngine] = None
_vector_store: Optional[IVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None
_metrics_collector: Optional[MetricsCollector] = None
_json_validator: Optional[JSONValidator] = None

@lru_cache()
def get_config():
    """Get configuration singleton."""
    return load_config()

async def get_llm_engine() -> ILLMEngine:
    """Get LLM engine singleton."""
    global _llm_engine
    
    if _llm_engine is None:
        config = get_config()
        _llm_engine = create_llm_engine(config['llm'])
        await _llm_engine.initialize()
        logger.info("LLM engine initialized")
        
    return _llm_engine

async def get_vector_store() -> IVectorStore:
    """Get vector store singleton."""
    global _vector_store
    
    if _vector_store is None:
        config = get_config()
        embedding_model = create_embedding_model(config['vector_store'])
        _vector_store = create_vector_store(config['vector_store'], embedding_model)
        await _vector_store.initialize()
        logger.info("Vector store initialized")
        
    return _vector_store

async def get_rag_engine() -> RAGPipeline:
    """Get RAG pipeline singleton."""
    global _rag_pipeline
    
    if _rag_pipeline is None:
        config = get_config()
        llm_engine = await get_llm_engine()
        vector_store = await get_vector_store()
        
        _rag_pipeline = RAGPipeline(
            llm_engine=llm_engine,
            vector_store=vector_store,
            config=config['rag']
        )
        logger.info("RAG pipeline initialized")
        
    return _rag_pipeline

def get_json_validator() -> JSONValidator:
    """Get JSON validator singleton."""
    global _json_validator
    
    if _json_validator is None:
        config = get_config()
        _json_validator = JSONValidator(config.get('jsonformer', {}))
        logger.info("JSON validator initialized")
        
    return _json_validator

def get_observability() -> MetricsCollector:
    """Get metrics collector singleton."""
    global _metrics_collector
    
    if _metrics_collector is None:
        config = get_config()
        _metrics_collector = MetricsCollector(config.get('observability', {}))
        logger.info("Metrics collector initialized")
        
    return _metrics_collector

async def cleanup_resources():
    """Cleanup all resources on shutdown."""
    global _llm_engine, _vector_store, _rag_pipeline, _metrics_collector
    
    if _llm_engine:
        await _llm_engine.shutdown()
        _llm_engine = None
        
    if _vector_store:
        await _vector_store.shutdown()
        _vector_store = None
        
    _rag_pipeline = None
    _metrics_collector = None
    
    logger.info("All resources cleaned up")