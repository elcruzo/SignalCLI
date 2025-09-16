"""Additional built-in tools for MCP server."""

import json
import asyncio
from typing import Dict, Any, List, Optional, AsyncIterator
from datetime import datetime
import numpy as np

from .tools import Tool, ToolCapability, ToolMetadata
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VectorSearchTool(Tool):
    """Direct vector similarity search tool."""

    def __init__(self, vector_store, embedding_model):
        """Initialize vector search tool."""
        super().__init__(
            name="vector_search",
            description="Perform direct vector similarity search",
            capabilities=[ToolCapability.SEARCH, ToolCapability.RETRIEVE],
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "default": 10},
                    "threshold": {"type": "number", "default": 0.7},
                    "filters": {"type": "object"},
                },
                "required": ["query"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "content": {"type": "string"},
                                "score": {"type": "number"},
                                "metadata": {"type": "object"},
                            },
                        },
                    },
                    "query_embedding": {"type": "array", "items": {"type": "number"}},
                },
            },
            metadata=ToolMetadata(tags=["search", "vector", "similarity"]),
        )
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute vector search."""
        query = params["query"]
        top_k = params.get("top_k", 10)
        threshold = params.get("threshold", 0.7)
        filters = params.get("filters", {})

        # Generate query embedding
        query_embedding = await self.embedding_model.encode(query)

        # Search
        results = await self.vector_store.search(
            query_embedding, top_k=top_k, threshold=threshold, filters=filters
        )

        return {
            "results": [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "query_embedding": query_embedding.tolist(),
        }


class TextGenerationTool(Tool):
    """Pure text generation tool."""

    def __init__(self, llm_engine):
        """Initialize text generation tool."""
        super().__init__(
            name="text_generation",
            description="Generate text based on prompts",
            capabilities=[ToolCapability.GENERATE],
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Generation prompt"},
                    "max_tokens": {"type": "integer", "default": 500},
                    "temperature": {"type": "number", "default": 0.7},
                    "top_p": {"type": "number", "default": 0.9},
                    "stop_sequences": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["prompt"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "generated_text": {"type": "string"},
                    "tokens_used": {"type": "integer"},
                    "finish_reason": {"type": "string"},
                },
            },
            supports_streaming=True,
            metadata=ToolMetadata(tags=["generation", "llm", "text"]),
        )
        self.llm_engine = llm_engine

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute text generation."""
        prompt = params["prompt"]
        max_tokens = params.get("max_tokens", 500)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        stop_sequences = params.get("stop_sequences", [])

        # Generate
        result = await self.llm_engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        )

        return {
            "generated_text": result.text,
            "tokens_used": result.tokens_used,
            "finish_reason": result.finish_reason,
        }

    async def stream_execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Any]:
        """Stream text generation."""
        prompt = params["prompt"]
        max_tokens = params.get("max_tokens", 500)
        temperature = params.get("temperature", 0.7)
        top_p = params.get("top_p", 0.9)
        stop_sequences = params.get("stop_sequences", [])

        # Stream generation
        async for chunk in self.llm_engine.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop_sequences,
        ):
            yield {"chunk": chunk.text, "type": "partial"}

        # Final result
        yield {"type": "complete", "finish_reason": chunk.finish_reason}


class StructuredGenerationTool(Tool):
    """Generate structured JSON output."""

    def __init__(self, llm_engine, jsonformer_validator):
        """Initialize structured generation tool."""
        super().__init__(
            name="structured_generation",
            description="Generate structured JSON output with schema validation",
            capabilities=[ToolCapability.GENERATE, ToolCapability.VALIDATE],
            input_schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "schema": {
                        "type": "object",
                        "description": "JSON schema for output",
                    },
                    "examples": {
                        "type": "array",
                        "items": {"type": "object"},
                        "default": [],
                    },
                },
                "required": ["prompt", "schema"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "object"},
                    "valid": {"type": "boolean"},
                    "errors": {"type": "array", "items": {"type": "string"}},
                },
            },
            metadata=ToolMetadata(tags=["json", "structured", "validation"]),
        )
        self.llm_engine = llm_engine
        self.jsonformer_validator = jsonformer_validator

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute structured generation."""
        prompt = params["prompt"]
        schema = params["schema"]
        examples = params.get("examples", [])

        # Add examples to prompt if provided
        if examples:
            example_text = "\n\nExamples:\n"
            for i, example in enumerate(examples, 1):
                example_text += f"Example {i}: {json.dumps(example, indent=2)}\n"
            prompt = prompt + example_text

        # Generate with schema
        result = await self.jsonformer_validator.generate_json(
            prompt=prompt, schema=schema, llm_engine=self.llm_engine
        )

        # Validate result
        is_valid, errors = self.jsonformer_validator.validate(result, schema)

        return {"result": result, "valid": is_valid, "errors": errors}


class TextAnalysisTool(Tool):
    """Analyze text for various properties."""

    def __init__(self, llm_engine, embedding_model):
        """Initialize text analysis tool."""
        super().__init__(
            name="text_analysis",
            description="Analyze text for sentiment, entities, topics, and more",
            capabilities=[ToolCapability.ANALYZE],
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "analyses": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "sentiment",
                                "entities",
                                "topics",
                                "summary",
                                "language",
                                "complexity",
                            ],
                        },
                        "default": ["sentiment", "entities", "topics"],
                    },
                },
                "required": ["text"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "score": {"type": "number"},
                        },
                    },
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "type": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "summary": {"type": "string"},
                    "language": {"type": "string"},
                    "complexity": {
                        "type": "object",
                        "properties": {
                            "readability_score": {"type": "number"},
                            "grade_level": {"type": "string"},
                        },
                    },
                },
            },
            metadata=ToolMetadata(tags=["analysis", "nlp", "text"]),
        )
        self.llm_engine = llm_engine
        self.embedding_model = embedding_model

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute text analysis."""
        text = params["text"]
        analyses = params.get("analyses", ["sentiment", "entities", "topics"])

        results = {}

        # Run requested analyses in parallel
        tasks = []
        if "sentiment" in analyses:
            tasks.append(self._analyze_sentiment(text))
        if "entities" in analyses:
            tasks.append(self._extract_entities(text))
        if "topics" in analyses:
            tasks.append(self._extract_topics(text))
        if "summary" in analyses:
            tasks.append(self._generate_summary(text))
        if "language" in analyses:
            tasks.append(self._detect_language(text))
        if "complexity" in analyses:
            tasks.append(self._analyze_complexity(text))

        analysis_results = await asyncio.gather(*tasks)

        # Combine results
        for analysis, result in zip(analyses, analysis_results):
            results[analysis] = result

        return results

    async def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        prompt = f"Analyze the sentiment of the following text. Respond with 'positive', 'negative', or 'neutral' and a confidence score (0-1):\n\n{text}"
        response = await self.llm_engine.generate(prompt, max_tokens=50)

        # Parse response
        # Simple parsing - in production would use structured output
        response_lower = response.text.lower()
        if "positive" in response_lower:
            label = "positive"
        elif "negative" in response_lower:
            label = "negative"
        else:
            label = "neutral"

        return {"label": label, "score": 0.85}  # Dummy score

    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities."""
        prompt = f"Extract named entities (people, organizations, locations) from the following text:\n\n{text}"
        response = await self.llm_engine.generate(prompt, max_tokens=200)

        # Parse response - simplified
        entities = []
        lines = response.text.strip().split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    entity_type, entity_text = parts
                    entities.append(
                        {
                            "text": entity_text.strip(),
                            "type": entity_type.strip(),
                            "confidence": 0.9,
                        }
                    )

        return entities

    async def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics."""
        prompt = f"List the main topics discussed in this text (max 5):\n\n{text}"
        response = await self.llm_engine.generate(prompt, max_tokens=100)

        # Parse topics
        topics = []
        lines = response.text.strip().split("\n")
        for line in lines:
            line = line.strip("- 1234567890.")
            if line:
                topics.append(line)

        return topics[:5]

    async def _generate_summary(self, text: str) -> str:
        """Generate text summary."""
        prompt = f"Summarize the following text in 2-3 sentences:\n\n{text}"
        response = await self.llm_engine.generate(prompt, max_tokens=150)
        return response.text.strip()

    async def _detect_language(self, text: str) -> str:
        """Detect text language."""
        prompt = f"What language is this text written in? Reply with just the language name:\n\n{text[:200]}"
        response = await self.llm_engine.generate(prompt, max_tokens=20)
        return response.text.strip()

    async def _analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze text complexity."""
        # Simple metrics
        words = text.split()
        sentences = text.split(".")
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0

        # Flesch Reading Ease approximation
        reading_ease = (
            206.835 - 1.015 * avg_sentence_length - 84.6 * (avg_word_length / 4.7)
        )
        reading_ease = max(0, min(100, reading_ease))

        # Grade level
        if reading_ease >= 90:
            grade_level = "Elementary"
        elif reading_ease >= 60:
            grade_level = "Middle School"
        elif reading_ease >= 30:
            grade_level = "High School"
        else:
            grade_level = "College"

        return {"readability_score": round(reading_ease, 2), "grade_level": grade_level}


class MultiModalTool(Tool):
    """Handle multi-modal inputs (text + images)."""

    def __init__(self, llm_engine, vision_model=None):
        """Initialize multi-modal tool."""
        super().__init__(
            name="multimodal_analysis",
            description="Analyze multi-modal inputs including text and images",
            capabilities=[ToolCapability.ANALYZE, ToolCapability.TRANSFORM],
            input_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text input"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs of images to analyze",
                    },
                    "task": {
                        "type": "string",
                        "enum": ["describe", "qa", "caption", "ocr"],
                        "default": "describe",
                    },
                },
                "required": ["task"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "task": {"type": "string"},
                    "metadata": {"type": "object"},
                },
            },
            metadata=ToolMetadata(tags=["multimodal", "vision", "image"]),
        )
        self.llm_engine = llm_engine
        self.vision_model = vision_model

    async def execute(
        self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute multi-modal analysis."""
        text = params.get("text", "")
        image_urls = params.get("image_urls", [])
        task = params["task"]

        if task == "describe" and image_urls:
            # Describe images
            result = await self._describe_images(image_urls, text)
        elif task == "qa" and text and image_urls:
            # Visual Q&A
            result = await self._visual_qa(image_urls[0], text)
        elif task == "caption" and image_urls:
            # Generate captions
            result = await self._generate_captions(image_urls)
        elif task == "ocr" and image_urls:
            # OCR
            result = await self._perform_ocr(image_urls)
        else:
            result = "Invalid task or missing inputs"

        return {
            "result": result,
            "task": task,
            "metadata": {"input_count": len(image_urls)},
        }

    async def _describe_images(self, image_urls: List[str], context_text: str) -> str:
        """Describe images with optional context."""
        # Placeholder - would use vision model
        prompt = f"Describe the following images"
        if context_text:
            prompt += f" in the context of: {context_text}"
        return await self.llm_engine.generate(prompt, max_tokens=200)

    async def _visual_qa(self, image_url: str, question: str) -> str:
        """Answer questions about an image."""
        # Placeholder
        prompt = f"Answer this question about the image: {question}"
        return await self.llm_engine.generate(prompt, max_tokens=150)

    async def _generate_captions(self, image_urls: List[str]) -> str:
        """Generate captions for images."""
        # Placeholder
        captions = []
        for i, url in enumerate(image_urls):
            caption = f"Caption for image {i+1}: A descriptive caption"
            captions.append(caption)
        return "\n".join(captions)

    async def _perform_ocr(self, image_urls: List[str]) -> str:
        """Extract text from images."""
        # Placeholder
        return "Extracted text from images"
