"""LlamaCpp engine for GGUF model support."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from concurrent.futures import ThreadPoolExecutor

from .base import BaseLLMEngine
from src.core.exceptions import LLMError, ModelNotFoundError, TokenLimitError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Import llama-cpp-python if available
try:
    from llama_cpp import Llama, LlamaGrammar

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. GGUF support disabled.")


class LlamaCppEngine(BaseLLMEngine):
    """LLM engine using llama-cpp-python for GGUF models."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_path = config.get("model_path")
        self.n_ctx = config.get("context_length", 4096)
        self.n_gpu_layers = config.get("gpu_layers", 35)
        self.use_gpu = config.get("use_gpu", True)
        self.n_threads = config.get("threads", 4)
        self.model: Optional["Llama"] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def _initialize_model(self) -> None:
        """Initialize the GGUF model."""
        if not LLAMA_CPP_AVAILABLE:
            raise LLMError("llama-cpp-python not installed")

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise ModelNotFoundError(f"Model not found: {self.model_path}")

        try:
            # Initialize model in thread pool
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(self._executor, self._create_model)

            logger.info(f"Loaded GGUF model: {model_path.name}")

        except Exception as e:
            raise LLMError(f"Failed to load model: {e}")

    def _create_model(self) -> "Llama":
        """Create Llama model instance."""
        return Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers if self.use_gpu else 0,
            n_threads=self.n_threads,
            verbose=False,
            seed=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mmap=True,
            use_mlock=False,
        )

    async def _generate_response(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        schema: Optional[Dict[str, Any]],
        **kwargs,
    ) -> tuple[str, int, Dict[str, Any]]:
        """Generate response using llama.cpp."""
        if self.model is None:
            raise LLMError("Model not initialized")

        # Prepare generation parameters
        generation_kwargs = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 40),
            "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
            "stop": kwargs.get("stop", []),
            "echo": False,
        }

        # Add grammar for JSON schema if provided
        if schema:
            grammar_str = self._schema_to_grammar(schema)
            if grammar_str:
                generation_kwargs["grammar"] = LlamaGrammar.from_string(grammar_str)

        try:
            # Generate in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor, lambda: self.model(prompt, **generation_kwargs)
            )

            # Extract response
            generated_text = response["choices"][0]["text"]

            # Count tokens
            prompt_tokens = len(self.model.tokenize(prompt.encode()))
            completion_tokens = response["usage"]["completion_tokens"]
            total_tokens = prompt_tokens + completion_tokens

            # Check token limits
            if total_tokens > self.n_ctx:
                raise TokenLimitError(
                    f"Token limit exceeded: {total_tokens} > {self.n_ctx}"
                )

            metadata = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model_path": self.model_path,
                "stop_reason": response["choices"][0].get("finish_reason", "unknown"),
            }

            return generated_text, total_tokens, metadata

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise LLMError(f"Generation failed: {e}")

    def _schema_to_grammar(self, schema: Dict[str, Any]) -> Optional[str]:
        """Convert JSON schema to GBNF grammar."""
        # This is a simplified implementation
        # A full implementation would handle all JSON schema features

        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            grammar_parts = ['root ::= "{" ']
            prop_rules = []

            for i, (prop, prop_schema) in enumerate(properties.items()):
                rule_name = f"prop_{prop}"
                prop_rules.append(f'{rule_name} ::= ""{prop}":" value')

                if i < len(properties) - 1:
                    grammar_parts.append(f'{rule_name} "," ')
                else:
                    grammar_parts.append(f"{rule_name} ")

            grammar_parts.append('"}"')

            # Add value types
            grammar_parts.extend(
                [
                    "",
                    "value ::= string | number | boolean | null",
                    'string ::= """ ([^"\\\\] | "\\\\" .)* """',
                    'number ::= "-"? [0-9]+ ("." [0-9]+)?',
                    'boolean ::= "true" | "false"',
                    'null ::= "null"',
                ]
            )

            return "\n".join(grammar_parts)

        return None

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self.model:
            # Clear model from memory
            self.model = None

        # Shutdown executor
        self._executor.shutdown(wait=True)
