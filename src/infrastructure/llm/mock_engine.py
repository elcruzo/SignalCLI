"""Mock LLM engine for testing."""

import asyncio
import json
from typing import Dict, Any, Optional

from .base import BaseLLMEngine


class MockLLMEngine(BaseLLMEngine):
    """Mock LLM engine for testing and development."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = "mock-llm-v1"
        self.responses = {
            "default": "This is a mock response from the LLM engine.",
            "hello": "Hello! I'm a mock LLM assistant. How can I help you?",
            "error": "Mock error response for testing.",
        }

    async def _initialize_model(self) -> None:
        """Mock initialization."""
        await asyncio.sleep(0.1)  # Simulate initialization time

    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, int, Dict[str, Any]]:
        """Generate mock response."""
        await asyncio.sleep(0.2)  # Simulate processing time

        # Determine response based on prompt
        prompt_lower = prompt.lower()

        if schema:
            # Generate structured response based on schema
            response_text = self._generate_structured_response(schema)
        elif "hello" in prompt_lower or "hi" in prompt_lower:
            response_text = self.responses["hello"]
        elif "error" in prompt_lower:
            response_text = self.responses["error"]
        else:
            response_text = self.responses["default"]

        # Add some context from the prompt
        if len(prompt) > 20:
            response_text += f" Based on your query about '{prompt[:50]}...'"

        # Mock token counting
        tokens_used = len(response_text.split()) + len(prompt.split())

        metadata = {
            "mock": True,
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response_text.split()),
            "total_tokens": tokens_used,
        }

        return response_text, tokens_used, metadata

    def _generate_structured_response(self, schema: Dict[str, Any]) -> str:
        """Generate response matching schema."""
        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            response_obj = {}

            for prop, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "string")

                if prop_type == "string":
                    response_obj[prop] = f"Mock {prop} value"
                elif prop_type == "number":
                    response_obj[prop] = 42
                elif prop_type == "boolean":
                    response_obj[prop] = True
                elif prop_type == "array":
                    response_obj[prop] = ["item1", "item2", "item3"]
                else:
                    response_obj[prop] = None

            return json.dumps(response_obj, indent=2)

        return json.dumps({"message": "Mock structured response"})

    async def _cleanup(self) -> None:
        """Mock cleanup."""
        await asyncio.sleep(0.05)  # Simulate cleanup time
