"""Custom exceptions for SignalCLI."""

class SignalCLIError(Exception):
    """Base exception for SignalCLI."""
    pass

class ConfigurationError(SignalCLIError):
    """Configuration related errors."""
    pass

class LLMError(SignalCLIError):
    """LLM engine related errors."""
    pass

class VectorStoreError(SignalCLIError):
    """Vector store related errors."""
    pass

class ValidationError(SignalCLIError):
    """Input/output validation errors."""
    pass

class CacheError(SignalCLIError):
    """Cache related errors."""
    pass

class RAGPipelineError(SignalCLIError):
    """RAG pipeline errors."""
    pass

class JSONFormerError(SignalCLIError):
    """JSON formatting errors."""
    pass

class TokenLimitError(LLMError):
    """Token limit exceeded."""
    pass

class ModelNotFoundError(LLMError):
    """Model file not found."""
    pass

class VectorStoreConnectionError(VectorStoreError):
    """Failed to connect to vector store."""
    pass

class DocumentNotFoundError(VectorStoreError):
    """Document not found in vector store."""
    pass

class SchemaValidationError(ValidationError):
    """Schema validation failed."""
    pass

class InputValidationError(ValidationError):
    """Input validation failed."""
    pass