"""LLM engine implementations."""

from typing import Dict, Any

from src.core.interfaces import ILLMEngine
from src.core.exceptions import LLMError
from .llamacpp_engine import LlamaCppEngine
from .mock_engine import MockLLMEngine

def create_llm_engine(config: Dict[str, Any]) -> ILLMEngine:
    """
    Factory function to create LLM engines.
    
    Args:
        config: LLM configuration
        
    Returns:
        LLM engine instance
        
    Raises:
        LLMError: If engine type is unknown
    """
    engine_type = config.get('model_type', 'llamafile')
    
    if engine_type == 'llamafile' or engine_type == 'gguf':
        return LlamaCppEngine(config)
    elif engine_type == 'mock':
        return MockLLMEngine(config)
    else:
        raise LLMError(f"Unknown LLM engine type: {engine_type}")

__all__ = ['create_llm_engine', 'LlamaCppEngine', 'MockLLMEngine']