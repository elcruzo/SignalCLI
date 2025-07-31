# SignalCLI Design Document

## Product Vision

SignalCLI is an enterprise-grade AI platform that addresses the need for reliable, observable, and production-ready AI services. It provides a command-line interface, REST API, and Model Context Protocol (MCP) server for AI-powered knowledge retrieval and tool execution. Unlike simple chatbot implementations, SignalCLI emphasizes JSON safety, system observability, enterprise-grade reliability, and seamless AI interoperability.

## Problem Statement

### Current Challenges
1. **Unreliable Output**: Most LLM tools produce unstructured text that's difficult to parse programmatically
2. **Poor Observability**: Lack of visibility into token usage, costs, and failure points
3. **Integration Complexity**: Difficult to embed AI capabilities into existing workflows
4. **Production Readiness**: Most AI tools are demos, not production-ready systems

### Target Users
- **DevOps Engineers**: Need reliable AI tools for automation and scripting
- **Data Scientists**: Require structured outputs for downstream processing
- **Backend Developers**: Want AI capabilities with proper monitoring and logging
- **Enterprise Teams**: Need compliance, security, and cost tracking
- **AI Assistant Developers**: Need standardized protocols for tool integration
- **Platform Engineers**: Require scalable, observable AI infrastructure

## Design Principles

### 1. JSON-First Architecture
Every response must be valid, schema-compliant JSON to ensure programmatic reliability.

```python
# Bad: Unstructured text output
"The answer is 42, but I'm not entirely sure about this."

# Good: Structured JSON output
{
    "answer": "42",
    "confidence": 0.73,
    "reasoning": "Based on Douglas Adams' Hitchhiker's Guide to the Galaxy",
    "sources": ["literature/hitchhikers_guide.txt"]
}
```

### 2. Observability by Default
Every operation must be logged, measured, and traceable.

```python
# Automatic logging for every query
{
    "timestamp": "2024-01-15T10:30:00Z",
    "query_id": "uuid-12345",
    "input_tokens": 45,
    "output_tokens": 128,
    "latency_ms": 1247,
    "model": "llama-3.1-8b",
    "vector_search_time_ms": 23,
    "inference_time_ms": 1224,
    "success": true
}
```

### 3. Fail-Safe Operation
Graceful degradation and comprehensive error handling.

### 4. Local-First
Prioritize local execution for privacy and reliability.

### 5. Protocol Compliance
Strict adherence to standards (JSON-RPC 2.0, MCP specification) for interoperability.

### 6. Context-Aware Intelligence
Smart routing and tool selection based on query analysis and usage patterns.

## Technical Architecture

### Component Design

#### 1. CLI Layer
**Responsibility**: User interface and experience

```python
# Design: Rich, intuitive CLI with progress indicators
class SignalCLI:
    def __init__(self):
        self.api_client = APIClient()
        self.console = Console()
    
    def query(self, text: str, schema: Optional[str] = None):
        with self.console.status("Processing query..."):
            response = self.api_client.query(text, schema)
            self.display_response(response)
```

**Features**:
- Rich terminal output with colors and formatting
- Progress indicators for long-running operations
- Interactive mode for multi-turn conversations
- Shell completion support

#### 2. RAG Engine
**Responsibility**: Knowledge retrieval and context preparation

```python
class RAGEngine:
    def __init__(self, vector_store: VectorStore, embedder: TextEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunk_size = 512
        self.overlap = 50
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Document]:
        # Hybrid search: vector + keyword
        vector_results = self.vector_search(query, top_k)
        keyword_results = self.keyword_search(query, top_k)
        return self.rerank_results(vector_results + keyword_results)
```

**Advanced Features**:
- Hybrid search (vector + BM25)
- Query expansion and rewriting
- Context window optimization
- Source attribution and citation

#### 3. LLM Inference Engine
**Responsibility**: Language model inference with constraints

```python
class LLMEngine:
    def __init__(self, model_path: str):
        self.model = self.load_model(model_path)
        self.tokenizer = self.load_tokenizer()
        self.jsonformer = None
    
    def generate_structured(self, prompt: str, schema: dict) -> dict:
        self.jsonformer = Jsonformer(
            model=self.model,
            tokenizer=self.tokenizer,
            json_schema=schema,
            prompt=prompt
        )
        return self.jsonformer()
```

**Optimization**:
- GPU acceleration when available
- Model quantization (INT8/INT4)
- KV-cache optimization
- Batch processing support

#### 4. MCP Server
**Responsibility**: Protocol-compliant tool serving for AI assistants

```python
class MCPServer:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.sessions = {}
        self.router = ContextAwareRouter()
    
    async def handle_request(self, request: MCPRequest) -> MCPResponse:
        # Validate session
        if request.method != "initialize" and not self.is_initialized(request.session_id):
            return error_response("Session not initialized")
        
        # Route to handler
        handler = self.get_handler(request.method)
        return await handler(request)
```

**Key Features**:
- JSON-RPC 2.0 compliance
- Session lifecycle management
- Tool discovery and introspection
- Streaming support (WebSocket/SSE)
- Context propagation between tools

#### 5. Context-Aware Router
**Responsibility**: Intelligent tool selection and routing

```python
class ContextAwareRouter:
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.routing_strategies = [
            CapabilityMatcher(),
            SemanticSimilarity(embedding_model),
            LoadBalancer(),
            CostOptimizer()
        ]
    
    async def route(self, request: Dict[str, Any]) -> str:
        context = self.extract_context(request)
        scores = {}
        
        for strategy in self.routing_strategies:
            tool_scores = await strategy.score_tools(context)
            # Weighted combination of scores
            scores = self.combine_scores(scores, tool_scores)
        
        return self.select_best_tool(scores)
```

#### 6. Observability System
**Responsibility**: Metrics, logging, and monitoring

```python
class ObservabilityManager:
    def __init__(self):
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger()
        self.tracer = DistributedTracer()
    
    @contextmanager
    def trace_query(self, query_id: str):
        start_time = time.time()
        try:
            yield
            self.metrics.record_success(time.time() - start_time)
        except Exception as e:
            self.metrics.record_failure(e)
            raise
```

### Data Models

#### Query Request
```python
from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    schema: Optional[Dict[str, Any]] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_k_retrieval: int = 5
    include_sources: bool = True
```

#### Query Response
```python
class QueryResponse(BaseModel):
    result: Dict[str, Any]  # Schema-validated response
    metadata: ResponseMetadata
    sources: List[SourceDocument]
    
class ResponseMetadata(BaseModel):
    query_id: str
    model_name: str
    tokens_used: TokenUsage
    latency_ms: int
    confidence_score: float
    
class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
```

## JSON Schema System

### Schema Registry
Centralized management of output schemas:

```yaml
# config/schemas/qa_response.yaml
type: object
required: [answer, confidence]
properties:
  answer:
    type: string
    description: "The main response to the user's question"
  confidence:
    type: number
    minimum: 0
    maximum: 1
    description: "Confidence score for the answer"
  sources:
    type: array
    items:
      type: string
    description: "Source documents used"
```

### Dynamic Schema Validation
```python
class SchemaValidator:
    def __init__(self, schema_registry: dict):
        self.registry = schema_registry
    
    def validate_response(self, response: dict, schema_name: str) -> bool:
        schema = self.registry.get(schema_name)
        return jsonschema.validate(response, schema)
```

## Vector Store Design

### Document Processing Pipeline
```python
class DocumentProcessor:
    def process_document(self, doc: Document) -> List[Chunk]:
        # 1. Extract text
        text = self.extract_text(doc)
        
        # 2. Clean and normalize
        clean_text = self.clean_text(text)
        
        # 3. Chunk with overlap
        chunks = self.chunk_text(clean_text, size=512, overlap=50)
        
        # 4. Generate embeddings
        for chunk in chunks:
            chunk.embedding = self.embed_text(chunk.content)
        
        return chunks
```

### Search Strategy
```python
class HybridSearch:
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        # Vector similarity search
        vector_results = self.vector_search(query, top_k * 2)
        
        # Keyword search (BM25)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine and rerank
        combined = self.combine_results(vector_results, keyword_results)
        return self.rerank(combined, query)[:top_k]
```

## MCP Protocol Design

### Tool Design Pattern
```python
class MCPTool(Tool):
    """Base class for MCP-compliant tools"""
    
    async def execute(self, params: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Returns MCP-compliant content array"""
        try:
            # Validate input
            self.validate_input(params)
            
            # Execute tool logic
            result = await self._execute_logic(params, context)
            
            # Format as MCP content
            return self._format_content(result)
        except Exception as e:
            return [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }]
```

### Session Management
```python
class MCPSession:
    def __init__(self, session_id: str):
        self.id = session_id
        self.initialized = False
        self.client_info = {}
        self.context = {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
    
    def is_expired(self, timeout_minutes: int = 30) -> bool:
        return (datetime.utcnow() - self.last_activity) > timedelta(minutes=timeout_minutes)
```

## Error Handling Strategy

### Error Categories
1. **Protocol Errors**: JSON-RPC parse errors, invalid requests
2. **MCP Errors**: Tool not found, session not initialized
3. **User Errors**: Invalid input, malformed queries
4. **System Errors**: Model failures, vector store unavailable
5. **Performance Errors**: Timeouts, memory issues
6. **External Errors**: Network failures, API limits

### Error Response Formats

#### REST API Error
```json
{
    "success": false,
    "error": {
        "code": "INFERENCE_TIMEOUT",
        "message": "LLM inference timed out after 30 seconds",
        "details": {
            "timeout_seconds": 30,
            "tokens_processed": 1024,
            "retry_suggested": true
        }
    }
}
```

#### MCP Error (JSON-RPC)
```json
{
    "jsonrpc": "2.0",
    "id": "request-123",
    "error": {
        "code": -32002,
        "message": "Tool execution failed",
        "data": {
            "tool": "rag_query",
            "reason": "Vector store unavailable"
        }
    },
    "metadata": {
        "query_id": "uuid-12345",
        "timestamp": "2024-01-15T10:30:00Z"
    }
}
```

## Performance Requirements

### Latency Targets
- **Interactive queries**: < 2 seconds end-to-end
- **Batch processing**: < 10 seconds per query
- **Vector search**: < 100ms
- **Model inference**: < 1.5 seconds

### Throughput Targets
- **Concurrent queries**: 10+ simultaneous users
- **Daily volume**: 10,000+ queries
- **Token throughput**: 50+ tokens/second

### Resource Constraints
- **Memory usage**: < 8GB total
- **CPU usage**: < 80% sustained
- **Disk space**: < 20GB for models and indices

## Security Design

### Input Sanitization
```python
class InputValidator:
    MAX_QUERY_LENGTH = 2048
    BLOCKED_PATTERNS = [
        r"<script.*?>.*?</script>",  # XSS prevention
        r"DROP\s+TABLE",             # SQL injection
        r"rm\s+-rf",                 # Command injection
    ]
    
    def validate_query(self, query: str) -> str:
        if len(query) > self.MAX_QUERY_LENGTH:
            raise ValidationError("Query too long")
        
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                raise SecurityError("Blocked pattern detected")
        
        return query
```

### Data Privacy
- No persistent storage of user queries by default
- Optional query logging with explicit consent
- Local processing to avoid data transmission
- Anonymization of metrics and logs

## Quality Assurance

### Testing Strategy
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: End-to-end pipeline testing
3. **Performance Tests**: Load and stress testing
4. **Security Tests**: Penetration testing and vulnerability scanning

### Evaluation Metrics
- **Relevance**: Vector search precision and recall
- **Accuracy**: LLM response quality assessment
- **Reliability**: Uptime and error rate monitoring
- **Performance**: Latency and throughput benchmarks

## Deployment Strategy

### Local Development
```bash
# Development setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Start all services
./scripts/start_all.sh local
```

### Production Deployment
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  # Infrastructure
  weaviate:
    image: semitechnologies/weaviate:1.22.4
    restart: always
    volumes:
      - weaviate-data:/var/lib/weaviate
    
  redis:
    image: redis:7-alpine
    restart: always
    volumes:
      - redis-data:/data
    
  # Application servers
  signalcli-api:
    image: signalcli:latest
    restart: always
    depends_on:
      - weaviate
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  signalcli-mcp:
    image: signalcli-mcp:latest
    restart: always
    depends_on:
      - weaviate
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    
  # Load balancer
  nginx:
    image: nginx:alpine
    restart: always
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - signalcli-api
      - signalcli-mcp

volumes:
  weaviate-data:
  redis-data:
```

## Future Roadmap

### Phase 1 (MVP) ✅ COMPLETED
- Basic CLI with JSON output
- Local LLM inference
- Simple vector search  
- Basic observability

### Phase 2 (Enhanced) ✅ COMPLETED
- MCP server for AI interoperability
- Advanced RAG with hybrid search
- Multi-vector store support
- Context-aware routing
- Streaming support
- Enhanced security (auth, rate limiting)

### Phase 3 (Enterprise) - IN PROGRESS
- Distributed deployment (Kubernetes)
- Advanced analytics dashboard
- Plugin system for custom tools
- Enterprise integrations (SSO, LDAP)
- Multi-tenant support
- Federated MCP servers

### Phase 4 (Platform)
- GraphQL API
- Tool marketplace
- Visual workflow builder
- Auto-scaling based on load
- Global CDN for models
- SLA guarantees

## Success Metrics

### Technical Metrics
- **Uptime**: > 99.9%
- **Response Time**: < 2s P95
- **Error Rate**: < 0.1%
- **Test Coverage**: > 90%

### User Metrics
- **Daily Active Users**: 100+ developers
- **Query Success Rate**: > 95%
- **User Satisfaction**: > 4.5/5 stars
- **Documentation Completeness**: 100% API coverage