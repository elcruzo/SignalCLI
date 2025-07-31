"""Advanced MCP client examples showcasing different features."""

import asyncio
import json
from typing import List, Dict, Any
from mcp_client import MCPClient


class AdvancedMCPExamples:
    """Advanced examples for MCP server usage."""

    def __init__(self, client: MCPClient):
        """Initialize with MCP client."""
        self.client = client

    async def context_aware_routing_example(self):
        """Example of context-aware routing."""
        print("\n=== Context-Aware Routing Example ===")

        # Different contexts will route to different tools
        contexts = [
            {
                "query": "What are the latest developments in quantum computing?",
                "context": {"preferred_tools": ["rag_query"], "quality_preference": "accurate"},
            },
            {
                "query": "Generate a technical report on quantum computing",
                "context": {"urgency": "low", "quality_preference": "accurate"},
            },
            {
                "query": "Summarize this document about quantum computing",
                "context": {"max_cost": 0.5, "urgency": "high"},
            },
        ]

        for ctx in contexts:
            result = await self.client.execute_tool(
                "auto",  # Let router decide
                {"query": ctx["query"]},
                context=ctx["context"],
            )
            print(f"\nQuery: {ctx['query']}")
            print(f"Routed to: {result['metadata']['tool']}")
            print(f"Latency: {result['metadata']['latency_ms']}ms")

    async def parallel_tool_execution(self):
        """Example of parallel tool execution."""
        print("\n=== Parallel Tool Execution ===")

        # Execute multiple analyses in parallel
        texts = [
            "Artificial intelligence is transforming healthcare.",
            "Climate change poses significant challenges to our planet.",
            "Quantum computers will revolutionize cryptography.",
        ]

        tasks = [
            self.client.execute_tool(
                "text_analysis",
                {"text": text, "analyses": ["sentiment", "topics", "complexity"]},
            )
            for text in texts
        ]

        results = await asyncio.gather(*tasks)

        for i, (text, result) in enumerate(zip(texts, results)):
            print(f"\nText {i+1}: {text[:50]}...")
            analysis = result["result"]
            print(f"Sentiment: {analysis['sentiment']['label']}")
            print(f"Topics: {', '.join(analysis['topics'][:3])}")
            print(f"Complexity: {analysis['complexity']['grade_level']}")

    async def structured_generation_with_schema(self):
        """Example of structured JSON generation."""
        print("\n=== Structured Generation Example ===")

        # Define a schema for a product description
        schema = {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "description": {"type": "string", "maxLength": 200},
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                },
                "price_range": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                    },
                    "required": ["min", "max"],
                },
                "target_audience": {"type": "string"},
                "availability": {"type": "boolean"},
            },
            "required": ["product_name", "description", "features", "price_range"],
        }

        result = await self.client.execute_tool(
            "structured_generation",
            {
                "prompt": "Generate a product description for an AI-powered smart home assistant",
                "schema": schema,
                "examples": [
                    {
                        "product_name": "EcoSmart Thermostat",
                        "description": "Intelligent thermostat that learns your preferences",
                        "features": ["Auto-scheduling", "Energy savings", "Remote control"],
                        "price_range": {"min": 199, "max": 299},
                        "target_audience": "Homeowners",
                        "availability": True,
                    }
                ],
            },
        )

        print("Generated Product:")
        print(json.dumps(result["result"]["result"], indent=2))
        print(f"Valid: {result['result']['valid']}")

    async def rag_with_filters(self):
        """Example of RAG with metadata filters."""
        print("\n=== RAG with Filters Example ===")

        # Query with different filters
        queries = [
            {
                "query": "machine learning algorithms",
                "filters": {"category": "technical", "year": {"$gte": 2023}},
            },
            {
                "query": "machine learning algorithms",
                "filters": {"category": "beginner", "language": "english"},
            },
        ]

        for q in queries:
            result = await self.client.execute_tool(
                "rag_query",
                {"query": q["query"], "top_k": 3, "filters": q["filters"]},
            )

            print(f"\nQuery: {q['query']}")
            print(f"Filters: {q['filters']}")
            print(f"Answer: {result['result']['answer'][:200]}...")
            print(f"Sources: {', '.join(result['result']['sources'])}")

    async def streaming_with_callbacks(self):
        """Example of streaming with progress callbacks."""
        print("\n=== Streaming with Progress ===")

        total_chunks = 0
        start_time = asyncio.get_event_loop().time()

        async for chunk in self.client.stream_tool(
            "text_generation",
            {
                "prompt": "Write a comprehensive guide to neural networks",
                "max_tokens": 500,
                "temperature": 0.7,
            },
        ):
            if chunk.get("chunk"):
                total_chunks += 1
                # Print progress every 10 chunks
                if total_chunks % 10 == 0:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    print(f"\n[Progress: {total_chunks} chunks, {elapsed:.1f}s]\n", end="")
                print(chunk["chunk"]["chunk"], end="", flush=True)
            elif chunk.get("status") == "completed":
                elapsed = asyncio.get_event_loop().time() - start_time
                print(f"\n\n[Completed: {total_chunks} chunks in {elapsed:.1f}s]")

    async def adaptive_tool_chaining(self):
        """Example of adaptive tool chaining based on results."""
        print("\n=== Adaptive Tool Chaining ===")

        # Initial query
        initial_query = "Explain the concept of transfer learning in AI"

        # Step 1: Generate explanation
        step1 = await self.client.execute_tool(
            "text_generation",
            {"prompt": initial_query, "max_tokens": 200},
        )

        explanation = step1["result"]["generated_text"]

        # Step 2: Analyze complexity
        step2 = await self.client.execute_tool(
            "text_analysis",
            {"text": explanation, "analyses": ["complexity"]},
        )

        complexity = step2["result"]["complexity"]["grade_level"]

        # Step 3: Adapt based on complexity
        if complexity in ["College", "High School"]:
            # Simplify if too complex
            step3 = await self.client.execute_tool(
                "text_generation",
                {
                    "prompt": f"Simplify this explanation for a middle school student: {explanation}",
                    "max_tokens": 200,
                },
            )
            final_text = step3["result"]["generated_text"]
            action = "Simplified"
        else:
            # Add more detail if too simple
            step3 = await self.client.execute_tool(
                "text_generation",
                {
                    "prompt": f"Add more technical detail to this explanation: {explanation}",
                    "max_tokens": 200,
                },
            )
            final_text = step3["result"]["generated_text"]
            action = "Enhanced"

        print(f"Original complexity: {complexity}")
        print(f"Action taken: {action}")
        print(f"\nFinal explanation: {final_text}")

    async def multimodal_analysis(self):
        """Example of multimodal analysis."""
        print("\n=== Multimodal Analysis Example ===")

        # Analyze image with text context
        result = await self.client.execute_tool(
            "multimodal_analysis",
            {
                "text": "What machine learning concepts are illustrated in this diagram?",
                "image_urls": ["https://example.com/ml-diagram.png"],
                "task": "qa",
            },
        )

        print(f"Question: {result['metadata']['input_count']} image(s) analyzed")
        print(f"Answer: {result['result']['result']}")

        # Generate captions for multiple images
        result2 = await self.client.execute_tool(
            "multimodal_analysis",
            {
                "image_urls": [
                    "https://example.com/image1.jpg",
                    "https://example.com/image2.jpg",
                    "https://example.com/image3.jpg",
                ],
                "task": "caption",
            },
        )

        print(f"\nGenerated captions:\n{result2['result']['result']}")

    async def batch_processing(self):
        """Example of batch processing multiple requests."""
        print("\n=== Batch Processing Example ===")

        # Create batch of similar requests
        batch_requests = [
            {
                "tool": "text_analysis",
                "arguments": {
                    "text": f"Sample text {i}: This is a test document for batch processing.",
                    "analyses": ["sentiment"],
                },
            }
            for i in range(10)
        ]

        # Process in chunks
        chunk_size = 3
        results = []

        for i in range(0, len(batch_requests), chunk_size):
            chunk = batch_requests[i : i + chunk_size]
            chunk_results = await asyncio.gather(
                *[
                    self.client.execute_tool(req["tool"], req["arguments"])
                    for req in chunk
                ]
            )
            results.extend(chunk_results)
            print(f"Processed batch {i//chunk_size + 1}/{(len(batch_requests) + chunk_size - 1)//chunk_size}")

        # Aggregate results
        sentiments = [r["result"]["sentiment"]["label"] for r in results]
        print(f"\nSentiment distribution:")
        for sentiment in set(sentiments):
            count = sentiments.count(sentiment)
            print(f"  {sentiment}: {count}/{len(sentiments)}")


async def main():
    """Run all advanced examples."""
    client = MCPClient()
    examples = AdvancedMCPExamples(client)

    # Run examples
    await examples.context_aware_routing_example()
    await examples.parallel_tool_execution()
    await examples.structured_generation_with_schema()
    await examples.rag_with_filters()
    await examples.streaming_with_callbacks()
    await examples.adaptive_tool_chaining()
    # await examples.multimodal_analysis()  # Requires actual image URLs
    await examples.batch_processing()


if __name__ == "__main__":
    asyncio.run(main())
