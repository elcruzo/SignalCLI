# SignalCLI MCP Server Documentation

## Overview

The SignalCLI MCP (Model Context Protocol) Server transforms SignalCLI into an interoperable AI service that other assistants and applications can leverage. It provides a standardized interface for tool discovery, execution, and streaming responses.

## Features

### 1. Tool Discovery
- Dynamic tool registration and capability advertisement
- Detailed tool metadata including schemas, examples, and costs
- Permission-based tool filtering per client

### 2. Context-Aware Routing
- Intelligent routing based on query context and available tools
- Multiple routing strategies: capability matching, semantic similarity, load balancing
- Adaptive routing based on usage patterns and performance metrics

### 3. Streaming Responses
- Real-time streaming via WebSocket and Server-Sent Events (SSE)
- Support for partial results and progress updates
- Automatic stream management and cleanup

### 4. Tool Chaining
- Sequential and parallel tool execution
- Context propagation between tools
- Conditional execution based on results

### 5. Caching & Performance
- Intelligent caching with multiple eviction strategies
- Request deduplication
- Performance metrics and monitoring

### 6. Security & Permissions
- API key authentication
- Fine-grained tool access control
- Rate limiting per client
- Audit logging

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server    │────▶│     Tools       │
│  (AI Assistant) │     │   (FastAPI)     │     │  (RAG, LLM...)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                        │
         │                       ▼                        │
         │              ┌─────────────────┐               │
         │              │ Context Router  │               │
         │              └─────────────────┘               │
         │                       │                        │
         │                       ▼                        │
         │              ┌─────────────────┐               │
         └─────────────▶│ Streaming Handler│◀─────────────┘
                        └─────────────────┘
```

## API Reference

### Tool Discovery

#### List Tools
```http
GET /mcp/v1/tools
Authorization: Bearer <api_key>
```

Response:
```json
{
  "tools": [
    {
      "name": "rag_query",
      "description": "Query the knowledge base using RAG",
      "capabilities": ["query", "retrieve", "analyze"],
      "input_schema": {...},
      "output_schema": {...},
      "streaming": true
    }
  ],
  "version": "1.0.0",
  "server": "SignalCLI-MCP"
}
```

#### Get Tool Details
```http
GET /mcp/v1/tools/{tool_name}
Authorization: Bearer <api_key>
```

### Tool Execution

#### Execute Tool
```http
POST /mcp/v1/execute
Content-Type: application/json
Authorization: Bearer <api_key>

{
  "method": "execute",
  "params": {
    "tool": "rag_query",
    "query": "What is machine learning?",
    "top_k": 5
  },
  "context": {
    "preferred_tools": ["rag_query"],
    "quality_preference": "accurate"
  },
  "stream": false
}
```

#### Execute Tool Chain
```http
POST /mcp/v1/execute/chain
Content-Type: application/json
Authorization: Bearer <api_key>

[
  {
    "tool": "text_generation",
    "arguments": {
      "prompt": "Generate a summary",
      "max_tokens": 200
    }
  },
  {
    "tool": "text_analysis",
    "arguments": {
      "text": "{{result_0.result.generated_text}}",
      "analyses": ["sentiment"]
    }
  }
]
```

### Streaming

#### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:8001/mcp/v1/stream?stream_id=<stream_id>');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.chunk) {
    console.log('Received chunk:', data.chunk);
  } else if (data.status === 'completed') {
    console.log('Stream completed');
  }
};
```

#### Server-Sent Events (SSE)
```javascript
const eventSource = new EventSource('/mcp/v1/stream/sse?stream_id=<stream_id>');

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## Built-in Tools

### 1. RAG Query Tool
- Query knowledge base with semantic search
- Retrieval-augmented generation
- Metadata filtering support

### 2. Document Index Tool
- Index documents into vector store
- Configurable chunking strategies
- Batch processing support

### 3. Text Generation Tool
- Generate text with LLM
- Streaming support
- Temperature and token control

### 4. Structured Generation Tool
- Generate JSON with schema validation
- JSONformer integration
- Example-based generation

### 5. Text Analysis Tool
- Sentiment analysis
- Entity extraction
- Topic modeling
- Complexity analysis

### 6. Vector Search Tool
- Direct vector similarity search
- Threshold-based filtering
- Metadata filtering

## Configuration

### Server Configuration
```yaml
mcp:
  host: "0.0.0.0"
  port: 8001
  workers: 4
  cache_size_mb: 1024
  cache_path: "/var/cache/mcp"
```

### Client Registration
```python
client_data = {
    "name": "My AI Assistant",
    "allowed_tools": ["rag_query", "text_generation"],
    "rate_limits": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    }
}

response = await client.register_client(client_data)
api_key = response["api_key"]
```

## Deployment

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

CMD ["python", "-m", "src.mcp.main"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    ports:
      - "8001:8001"
    environment:
      - ENV=production
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./cache:/app/cache
    depends_on:
      - weaviate
      - redis

  weaviate:
    image: semitechnologies/weaviate:latest
    ports:
      - "8080:8080"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## Performance Tuning

### Caching Strategies
1. **LRU (Least Recently Used)**: Good for general use
2. **LFU (Least Frequently Used)**: Better for repeated queries
3. **TTL (Time To Live)**: Good for time-sensitive data
4. **Adaptive**: Automatically adjusts based on usage patterns

### Routing Optimization
- Pre-warm embeddings for common queries
- Cache routing decisions
- Use load balancing for high traffic

### Scaling
- Horizontal scaling with multiple workers
- Redis for distributed caching
- Load balancer for high availability

## Monitoring

### Metrics
- Request latency by tool
- Cache hit rates
- Tool usage statistics
- Error rates and types

### Health Checks
```http
GET /mcp/v1/health
```

Response:
```json
{
  "status": "healthy",
  "tools_loaded": 15,
  "active_streams": 3,
  "cache_stats": {
    "entries": 1234,
    "hit_rate": 0.85
  }
}
```

## Security Best Practices

1. **API Key Management**
   - Rotate keys regularly
   - Use environment variables
   - Implement key expiration

2. **Rate Limiting**
   - Configure per-client limits
   - Implement backoff strategies
   - Monitor for abuse

3. **Input Validation**
   - Validate against schemas
   - Sanitize user inputs
   - Limit payload sizes

4. **Network Security**
   - Use HTTPS in production
   - Implement CORS properly
   - Use firewall rules

## Troubleshooting

### Common Issues

1. **Tool Not Found**
   - Check tool registration
   - Verify permissions
   - Check routing configuration

2. **Streaming Timeout**
   - Increase timeout settings
   - Check network connectivity
   - Monitor server resources

3. **High Latency**
   - Enable caching
   - Optimize routing strategy
   - Check model loading

### Debug Mode
```bash
LOG_LEVEL=DEBUG python -m src.mcp.main
```

## Example Integration

### Python Client
```python
from examples.mcp_client import MCPClient

client = MCPClient(base_url="http://localhost:8001", api_key="your_api_key")

# List tools
tools = await client.list_tools()

# Execute RAG query
result = await client.execute_tool(
    "rag_query",
    {"query": "What is SignalCLI?", "top_k": 5}
)

# Stream generation
async for chunk in client.stream_tool(
    "text_generation",
    {"prompt": "Explain MCP servers", "max_tokens": 500}
):
    print(chunk)
```

### JavaScript Client
```javascript
const response = await fetch('http://localhost:8001/mcp/v1/execute', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer your_api_key'
  },
  body: JSON.stringify({
    method: 'execute',
    params: {
      tool: 'rag_query',
      query: 'What is SignalCLI?'
    }
  })
});

const result = await response.json();
console.log(result);
```

## Future Enhancements

1. **GraphQL Interface**: Alternative to REST API
2. **Multi-modal Support**: Enhanced image/video processing
3. **Federation**: Connect multiple MCP servers
4. **Plugin System**: Custom tool development
5. **WebRTC Streaming**: Lower latency streaming
