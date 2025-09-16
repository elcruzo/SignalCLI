"""Context-aware routing for MCP tools."""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from .tools import Tool, ToolCapability, ToolRegistry
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RoutingStrategy(Enum):
    """Routing strategies."""

    CAPABILITY_MATCH = "capability_match"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CONTEXT_AWARE = "context_aware"
    LOAD_BALANCED = "load_balanced"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class RoutingContext:
    """Context for routing decisions."""

    query: str
    user_intent: Optional[str] = None
    required_capabilities: List[ToolCapability] = None
    preferred_tools: List[str] = None
    excluded_tools: List[str] = None
    max_cost: Optional[float] = None
    urgency: Optional[str] = None  # "low", "medium", "high"
    quality_preference: Optional[str] = None  # "fast", "balanced", "accurate"


class ContextAwareRouter:
    """Intelligent router for MCP tool selection."""

    def __init__(
        self,
        tool_registry: ToolRegistry,
        embedding_model=None,
        strategy: RoutingStrategy = RoutingStrategy.CONTEXT_AWARE,
    ):
        """Initialize router."""
        self.tool_registry = tool_registry
        self.embedding_model = embedding_model
        self.strategy = strategy
        self.tool_usage_stats: Dict[str, Dict[str, Any]] = {}
        self._initialize_routing_rules()

    def _initialize_routing_rules(self):
        """Initialize routing rules and patterns."""
        self.capability_keywords = {
            ToolCapability.QUERY: ["search", "find", "query", "lookup", "retrieve"],
            ToolCapability.SUMMARIZE: [
                "summarize",
                "summary",
                "brief",
                "overview",
                "tldr",
            ],
            ToolCapability.GENERATE: [
                "create",
                "generate",
                "write",
                "compose",
                "draft",
            ],
            ToolCapability.ANALYZE: [
                "analyze",
                "examine",
                "investigate",
                "study",
                "evaluate",
            ],
            ToolCapability.TRANSFORM: [
                "convert",
                "transform",
                "translate",
                "format",
                "modify",
            ],
            ToolCapability.VALIDATE: ["validate", "check", "verify", "confirm", "test"],
            ToolCapability.STORE: ["save", "store", "index", "add", "insert"],
            ToolCapability.RETRIEVE: ["get", "fetch", "retrieve", "load", "read"],
        }

        self.intent_patterns = {
            "question": [
                "what",
                "when",
                "where",
                "who",
                "why",
                "how",
                "is",
                "are",
                "can",
            ],
            "command": ["show", "list", "display", "get", "find", "create", "delete"],
            "analysis": ["analyze", "compare", "evaluate", "assess", "investigate"],
        }

    async def route(self, request: Dict[str, Any]) -> str:
        """Route request to appropriate tool."""
        # Extract routing context
        context = self._extract_context(request)

        # Apply routing strategy
        if self.strategy == RoutingStrategy.CAPABILITY_MATCH:
            return await self._route_by_capability(context)
        elif self.strategy == RoutingStrategy.SEMANTIC_SIMILARITY:
            return await self._route_by_similarity(context)
        elif self.strategy == RoutingStrategy.CONTEXT_AWARE:
            return await self._route_context_aware(context)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._route_load_balanced(context)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return await self._route_cost_optimized(context)
        else:
            # Default to capability matching
            return await self._route_by_capability(context)

    def _extract_context(self, request: Dict[str, Any]) -> RoutingContext:
        """Extract routing context from request."""
        query = request.get("params", {}).get("query", "")
        context_data = request.get("context", {})

        # Detect user intent
        user_intent = self._detect_intent(query)

        # Extract required capabilities
        required_capabilities = self._detect_capabilities(query)

        return RoutingContext(
            query=query,
            user_intent=user_intent,
            required_capabilities=required_capabilities,
            preferred_tools=context_data.get("preferred_tools"),
            excluded_tools=context_data.get("excluded_tools"),
            max_cost=context_data.get("max_cost"),
            urgency=context_data.get("urgency", "medium"),
            quality_preference=context_data.get("quality_preference", "balanced"),
        )

    def _detect_intent(self, query: str) -> str:
        """Detect user intent from query."""
        query_lower = query.lower()

        for intent, patterns in self.intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent

        return "general"

    def _detect_capabilities(self, query: str) -> List[ToolCapability]:
        """Detect required capabilities from query."""
        query_lower = query.lower()
        capabilities = []

        for capability, keywords in self.capability_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                capabilities.append(capability)

        # Default to QUERY if no specific capability detected
        if not capabilities:
            capabilities = [ToolCapability.QUERY]

        return capabilities

    async def _route_by_capability(self, context: RoutingContext) -> str:
        """Route based on capability matching."""
        # Find tools with required capabilities
        matching_tools = self.tool_registry.find_by_capabilities(
            context.required_capabilities
        )

        if not matching_tools:
            # Fallback to tools with any of the capabilities
            for capability in context.required_capabilities:
                matching_tools = self.tool_registry.find_by_capability(capability)
                if matching_tools:
                    break

        # Filter by preferences
        matching_tools = self._apply_filters(matching_tools, context)

        if not matching_tools:
            # Default to first available tool
            all_tools = self.tool_registry.list_tools()
            if all_tools:
                return all_tools[0].name
            raise ValueError("No tools available")

        # Return tool with highest priority
        return self._select_best_tool(matching_tools, context)

    async def _route_by_similarity(self, context: RoutingContext) -> str:
        """Route based on semantic similarity."""
        if not self.embedding_model:
            # Fallback to capability matching
            return await self._route_by_capability(context)

        # Get query embedding
        query_embedding = await self.embedding_model.encode(context.query)

        # Calculate similarity scores
        tools = self.tool_registry.list_tools()
        similarities = []

        for tool in tools:
            # Get tool description embedding
            tool_text = f"{tool.name} {tool.description} {' '.join([c.value for c in tool.capabilities])}"
            tool_embedding = await self.embedding_model.encode(tool_text)

            # Calculate cosine similarity
            similarity = np.dot(query_embedding, tool_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
            )
            similarities.append((tool, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Filter and select
        filtered_tools = [
            tool
            for tool, _ in similarities
            if tool in self._apply_filters(tools, context)
        ]

        if filtered_tools:
            return filtered_tools[0].name

        # Fallback
        return similarities[0][0].name if similarities else tools[0].name

    async def _route_context_aware(self, context: RoutingContext) -> str:
        """Route using context-aware strategy."""
        # Score each tool based on multiple factors
        tools = self.tool_registry.list_tools()
        tool_scores = []

        for tool in tools:
            score = 0.0

            # Capability match score (weight: 0.4)
            capability_score = self._calculate_capability_score(tool, context)
            score += capability_score * 0.4

            # Semantic similarity score (weight: 0.3)
            if self.embedding_model:
                similarity_score = await self._calculate_similarity_score(tool, context)
                score += similarity_score * 0.3
            else:
                # Use keyword matching as fallback
                keyword_score = self._calculate_keyword_score(tool, context)
                score += keyword_score * 0.3

            # Performance score based on usage stats (weight: 0.2)
            perf_score = self._calculate_performance_score(tool)
            score += perf_score * 0.2

            # Cost score (weight: 0.1)
            cost_score = self._calculate_cost_score(tool, context)
            score += cost_score * 0.1

            # Apply urgency and quality adjustments
            score = self._adjust_for_preferences(score, tool, context)

            tool_scores.append((tool, score))

        # Sort by score
        tool_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter and select
        filtered_tools = [
            (tool, score)
            for tool, score in tool_scores
            if tool in self._apply_filters(tools, context)
        ]

        if filtered_tools:
            selected_tool = filtered_tools[0][0]
            logger.info(
                f"Selected tool '{selected_tool.name}' with score {filtered_tools[0][1]:.3f}"
            )
            return selected_tool.name

        # Fallback
        return tool_scores[0][0].name if tool_scores else "rag_query"

    async def _route_load_balanced(self, context: RoutingContext) -> str:
        """Route with load balancing."""
        matching_tools = self.tool_registry.find_by_capabilities(
            context.required_capabilities
        )
        matching_tools = self._apply_filters(matching_tools, context)

        if not matching_tools:
            return await self._route_by_capability(context)

        # Select tool with lowest current load
        min_load = float("inf")
        selected_tool = matching_tools[0]

        for tool in matching_tools:
            stats = self.tool_usage_stats.get(tool.name, {})
            current_load = stats.get("active_requests", 0)

            if current_load < min_load:
                min_load = current_load
                selected_tool = tool

        return selected_tool.name

    async def _route_cost_optimized(self, context: RoutingContext) -> str:
        """Route based on cost optimization."""
        matching_tools = self.tool_registry.find_by_capabilities(
            context.required_capabilities
        )
        matching_tools = self._apply_filters(matching_tools, context)

        if not matching_tools:
            return await self._route_by_capability(context)

        # Select tool with lowest cost
        min_cost = float("inf")
        selected_tool = matching_tools[0]

        for tool in matching_tools:
            cost = tool.metadata.cost_estimate or 1.0

            if cost < min_cost and (not context.max_cost or cost <= context.max_cost):
                min_cost = cost
                selected_tool = tool

        return selected_tool.name

    def _apply_filters(self, tools: List[Tool], context: RoutingContext) -> List[Tool]:
        """Apply context filters to tools."""
        filtered = tools

        # Apply preferred tools filter
        if context.preferred_tools:
            preferred = [t for t in filtered if t.name in context.preferred_tools]
            if preferred:
                filtered = preferred

        # Apply excluded tools filter
        if context.excluded_tools:
            filtered = [t for t in filtered if t.name not in context.excluded_tools]

        # Apply cost filter
        if context.max_cost is not None:
            filtered = [
                t
                for t in filtered
                if not t.metadata.cost_estimate
                or t.metadata.cost_estimate <= context.max_cost
            ]

        return filtered

    def _select_best_tool(self, tools: List[Tool], context: RoutingContext) -> str:
        """Select best tool from filtered list."""
        if not tools:
            raise ValueError("No tools to select from")

        # Simple selection based on priority
        # Could be enhanced with more sophisticated logic
        return tools[0].name

    def _calculate_capability_score(self, tool: Tool, context: RoutingContext) -> float:
        """Calculate capability match score."""
        if not context.required_capabilities:
            return 0.5

        tool_capabilities = set(tool.capabilities)
        required_capabilities = set(context.required_capabilities)

        # Jaccard similarity
        intersection = len(tool_capabilities & required_capabilities)
        union = len(tool_capabilities | required_capabilities)

        return intersection / union if union > 0 else 0.0

    def _calculate_keyword_score(self, tool: Tool, context: RoutingContext) -> float:
        """Calculate keyword match score."""
        query_lower = context.query.lower()
        tool_text = f"{tool.name} {tool.description}".lower()

        # Count keyword matches
        matches = 0
        total_keywords = 0

        for keywords in self.capability_keywords.values():
            for keyword in keywords:
                total_keywords += 1
                if keyword in query_lower and keyword in tool_text:
                    matches += 1

        return matches / total_keywords if total_keywords > 0 else 0.0

    async def _calculate_similarity_score(
        self, tool: Tool, context: RoutingContext
    ) -> float:
        """Calculate semantic similarity score."""
        if not self.embedding_model:
            return 0.5

        query_embedding = await self.embedding_model.encode(context.query)
        tool_text = f"{tool.name} {tool.description}"
        tool_embedding = await self.embedding_model.encode(tool_text)

        similarity = np.dot(query_embedding, tool_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(tool_embedding)
        )

        return float(similarity)

    def _calculate_performance_score(self, tool: Tool) -> float:
        """Calculate performance score based on usage stats."""
        stats = self.tool_usage_stats.get(tool.name, {})

        if not stats:
            return 0.5  # Neutral score for new tools

        # Consider success rate and average latency
        success_rate = stats.get("success_rate", 0.5)
        avg_latency = stats.get("avg_latency", 1.0)

        # Normalize latency (assuming 0-5 seconds range)
        latency_score = max(0, 1 - (avg_latency / 5.0))

        # Combined score
        return (success_rate * 0.7) + (latency_score * 0.3)

    def _calculate_cost_score(self, tool: Tool, context: RoutingContext) -> float:
        """Calculate cost score."""
        if not tool.metadata.cost_estimate:
            return 0.5

        if context.max_cost:
            # Normalize cost to 0-1 range
            return max(0, 1 - (tool.metadata.cost_estimate / context.max_cost))

        # Default scoring if no max cost specified
        # Assuming costs range from 0-10
        return max(0, 1 - (tool.metadata.cost_estimate / 10.0))

    def _adjust_for_preferences(
        self, score: float, tool: Tool, context: RoutingContext
    ) -> float:
        """Adjust score based on urgency and quality preferences."""
        adjusted_score = score

        # Urgency adjustments
        if context.urgency == "high":
            # Prefer faster tools
            stats = self.tool_usage_stats.get(tool.name, {})
            avg_latency = stats.get("avg_latency", 1.0)
            if avg_latency < 0.5:  # Fast tool
                adjusted_score *= 1.2
            elif avg_latency > 2.0:  # Slow tool
                adjusted_score *= 0.8

        # Quality preference adjustments
        if context.quality_preference == "accurate":
            # Prefer tools with high success rates
            stats = self.tool_usage_stats.get(tool.name, {})
            success_rate = stats.get("success_rate", 0.5)
            if success_rate > 0.9:
                adjusted_score *= 1.15
        elif context.quality_preference == "fast":
            # Already handled in urgency

            pass

        return min(1.0, adjusted_score)  # Cap at 1.0

    def update_usage_stats(self, tool_name: str, success: bool, latency: float) -> None:
        """Update tool usage statistics."""
        if tool_name not in self.tool_usage_stats:
            self.tool_usage_stats[tool_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_latency": 0.0,
                "active_requests": 0,
            }

        stats = self.tool_usage_stats[tool_name]
        stats["total_requests"] += 1
        stats["total_latency"] += latency

        if success:
            stats["successful_requests"] += 1

        # Calculate derived stats
        stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
        stats["avg_latency"] = stats["total_latency"] / stats["total_requests"]
