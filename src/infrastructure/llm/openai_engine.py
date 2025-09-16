"""OpenAI API engine for cloud-based LLM support."""

import asyncio
import json
from typing import Dict, Any, Optional
from openai import AsyncOpenAI

from .base import BaseLLMEngine
from src.core.exceptions import LLMError, APIKeyError, TokenLimitError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIEngine(BaseLLMEngine):
    """LLM engine using OpenAI API."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.organization = config.get("organization")
        self.base_url = config.get("base_url")
        self.model_name = config.get("model_name", "gpt-3.5-turbo")
        self.client: Optional[AsyncOpenAI] = None

    async def _initialize_model(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            raise APIKeyError("OpenAI API key not provided")

        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                base_url=self.base_url,
            )

            # Test the connection
            models = await self.client.models.list()
            available_models = [model.id for model in models.data]

            if self.model_name not in available_models:
                logger.warning(
                    f"Model {self.model_name} not found. Available: {available_models[:5]}"
                )

            logger.info(f"Initialized OpenAI client for model: {self.model_name}")

        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}")

    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, int, Dict[str, Any]]:
        """Generate response using OpenAI API."""
        if not self.client:
            raise LLMError("OpenAI client not initialized")

        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]

            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0),
                "stop": kwargs.get("stop"),
            }

            # Add JSON mode if schema is provided and model supports it
            if schema and self.model_name in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]:
                request_params["response_format"] = {"type": "json_object"}
                # Add schema instruction to prompt
                schema_instruction = f"\n\nPlease respond with valid JSON matching this schema: {json.dumps(schema)}"
                messages[0]["content"] += schema_instruction

            # Make API call
            response = await self.client.chat.completions.create(**request_params)

            # Extract response data
            choice = response.choices[0]
            generated_text = choice.message.content or ""

            # Get token usage
            usage = response.usage
            total_tokens = usage.total_tokens if usage else 0
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0

            # Check for content filtering
            if choice.finish_reason == "content_filter":
                raise LLMError("Content was filtered by OpenAI")

            # Check token limits
            if total_tokens > max_tokens:
                logger.warning(f"Response truncated due to token limit: {total_tokens}")

            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": response.model,
                "finish_reason": choice.finish_reason,
                "system_fingerprint": response.system_fingerprint,
            }

            return generated_text, total_tokens, metadata

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            if "quota" in str(e).lower():
                raise LLMError("OpenAI API quota exceeded")
            elif "api_key" in str(e).lower():
                raise APIKeyError("Invalid OpenAI API key")
            else:
                raise LLMError(f"OpenAI API error: {e}")

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response generation."""
        if not self.client:
            raise LLMError("OpenAI client not initialized")

        try:
            messages = [{"role": "user", "content": prompt}]

            # Note: JSON mode doesn't work with streaming
            if schema:
                schema_instruction = f"\n\nPlease respond with valid JSON matching this schema: {json.dumps(schema)}"
                messages[0]["content"] += schema_instruction

            request_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                **kwargs,
            }

            # Stream the response
            async for chunk in await self.client.chat.completions.create(
                **request_params
            ):
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise LLMError(f"Streaming failed: {e}")

    async def get_embeddings(
        self, texts: list[str], model: str = "text-embedding-ada-002"
    ) -> list[list[float]]:
        """Get embeddings for texts."""
        if not self.client:
            raise LLMError("OpenAI client not initialized")

        try:
            response = await self.client.embeddings.create(input=texts, model=model)

            return [embedding.embedding for embedding in response.data]

        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise LLMError(f"Embedding generation failed: {e}")

    async def _cleanup(self) -> None:
        """Cleanup OpenAI client."""
        if self.client:
            await self.client.close()
            self.client = None

    @property
    def supports_streaming(self) -> bool:
        """Check if engine supports streaming."""
        return True

    @property
    def supports_json_mode(self) -> bool:
        """Check if engine supports JSON mode."""
        return self.model_name in ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
