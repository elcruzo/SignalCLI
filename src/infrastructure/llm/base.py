"""Base LLM engine implementation."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.core.interfaces import ILLMEngine, LLMResponse
from src.core.exceptions import LLMError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseLLMEngine(ABC, ILLMEngine):
    """Base class for LLM engines."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "unknown")
        self.max_tokens = config.get("max_tokens", 2048)
        self.temperature = config.get("temperature", 0.7)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the LLM engine."""
        if self._initialized:
            return

        try:
            await self._initialize_model()
            self._initialized = True
            logger.info(f"LLM engine initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM engine: {e}")
            raise LLMError(f"Initialization failed: {e}")

    @abstractmethod
    async def _initialize_model(self) -> None:
        """Initialize the specific model implementation."""
        pass

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response from prompt."""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Use provided params or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature

            # Log generation request
            logger.debug(
                f"Generating response",
                extra={
                    "prompt_length": len(prompt),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "has_schema": schema is not None,
                },
            )

            # Generate response
            response_text, tokens_used, metadata = await self._generate_response(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                schema=schema,
                **kwargs,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            return LLMResponse(
                text=response_text,
                tokens_used=tokens_used,
                model_name=self.model_name,
                latency_ms=latency_ms,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise LLMError(f"Generation failed: {e}")

    @abstractmethod
    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, int, Dict[str, Any]]:
        """Generate response implementation."""
        pass

    async def health_check(self) -> bool:
        """Check if engine is healthy."""
        try:
            if not self._initialized:
                await self.initialize()

            # Try a simple generation
            response = await self.generate("Hello", max_tokens=10, temperature=0.0)

            return len(response.text) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def shutdown(self) -> None:
        """Cleanup resources."""
        if self._initialized:
            await self._cleanup()
            self._initialized = False
            logger.info(f"LLM engine shut down: {self.model_name}")

    @abstractmethod
    async def _cleanup(self) -> None:
        """Cleanup implementation."""
        pass
