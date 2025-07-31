# MCP Server Quick Start Guide

## What is MCP?

The Model Context Protocol (MCP) server transforms SignalCLI into an interoperable AI service that other AI assistants and applications can leverage. It provides standardized APIs for tool discovery, execution, and streaming.

## Quick Start

### 1. Start Everything with Docker

```bash
# Start all services (API, MCP, Weaviate, Redis)
./scripts/start_all.sh docker
```

### 2. Start MCP Server Only

```bash
# Start just the MCP server
./scripts/start_mcp.sh
```

### 3. Verify Installation

```bash
# Check health
curl http://localhost:8001/mcp/v1/health

# List available tools
curl http://localhost:8001/mcp/v1/tools
```

## Using the MCP Server

### Python Client Example

```python
import asyncio
from examples.mcp_client import MCPClient

async def main():
    # Initialize client
    client = MCPClient(base_url="http://localhost:8001")
    
    # List tools
    tools = await client.list_tools()
    print(f"Available tools: {len(tools['tools'])}")
    
    # Execute RAG query
    result = await client.execute_tool(
        "rag_query",
        {"query": "What is SignalCLI?", "top_k": 5}
    )
    print(f"Answer: {result['result']['answer']}")
    
    # Stream text generation
    async for chunk in client.stream_tool(
        "text_generation",
        {"prompt": "Explain MCP servers", "max_tokens": 200}
    ):
        print(chunk['chunk']['chunk'], end='', flush=True)

asyncio.run(main())
```

### REST API Example

```bash
# Execute a tool
curl -X POST http://localhost:8001/mcp/v1/execute \
  -H "Content-Type: application/json" \
  -d '{
    "method": "execute",
    "params": {
      "tool": "rag_query",
      "query": "What is machine learning?",
      "top_k": 5
    }
  }'

# Execute a tool chain
curl -X POST http://localhost:8001/mcp/v1/execute/chain \
  -H "Content-Type: application/json" \
  -d '[
    {
      "tool": "text_generation",
      "arguments": {"prompt": "Generate a summary", "max_tokens": 200}
    },
    {
      "tool": "text_analysis",
      "arguments": {"text": "{{result_0.result.generated_text}}", "analyses": ["sentiment"]}
    }
  ]'
```

### WebSocket Streaming

```javascript
// First, initiate a streaming request
const response = await fetch('http://localhost:8001/mcp/v1/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    method: 'execute',
    params: { tool: 'text_generation', prompt: 'Tell me a story', max_tokens: 500 },
    stream: true
  })
});

const { result } = await response.json();
const streamId = result.stream_id;

// Connect to WebSocket
const ws = new WebSocket(`ws://localhost:8001/mcp/v1/stream?stream_id=${streamId}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.chunk) {
    console.log(data.chunk);
  }
};
```

## Available Tools

### Core Tools

1. **rag_query** - Query knowledge base with RAG
   - Semantic search with context
   - Metadata filtering
   - Confidence scoring

2. **text_generation** - Generate text with LLM
   - Streaming support
   - Temperature control
   - Token limits

3. **structured_generation** - Generate JSON with schema validation
   - JSONformer integration
   - Schema enforcement
   - Example-based generation

4. **text_analysis** - Analyze text properties
   - Sentiment analysis
   - Entity extraction
   - Topic modeling
   - Complexity scoring

5. **vector_search** - Direct vector similarity search
   - Threshold filtering
   - Metadata filters
   - Embedding visualization

6. **document_index** - Index documents
   - Configurable chunking
   - Batch processing
   - Progress tracking

## Configuration

### Environment Variables

```bash
# MCP Server
export MCP_HOST=0.0.0.0
export MCP_PORT=8001
export MCP_WORKERS=4

# Cache
export CACHE_PROVIDER=redis
export REDIS_URL=redis://localhost:6379/1

# Vector Store
export VECTOR_STORE_PROVIDER=weaviate
export WEAVIATE_URL=http://localhost:8080
```

### Config File

```yaml
# config/mcp.yaml
mcp:
  host: 0.0.0.0
  port: 8001
  workers: 4
  cache_size_mb: 1024
  cache_strategy: adaptive
  
routing:
  strategy: context_aware
  embedding_cache: true
  
rate_limits:
  default:
    requests_per_minute: 60
    requests_per_hour: 1000
    concurrent_requests: 10
```

## Authentication

### Register a Client

```python
client_data = {
    "name": "My AI Assistant",
    "allowed_tools": ["rag_query", "text_generation"],
    "rate_limits": {
        "requests_per_minute": 100,
        "requests_per_hour": 2000
    }
}

response = requests.post(
    "http://localhost:8001/mcp/v1/clients/register",
    json=client_data
)

api_key = response.json()["api_key"]
```

### Use API Key

```python
client = MCPClient(
    base_url="http://localhost:8001",
    api_key=api_key
)
```

## Monitoring

### Health Check

```bash
curl http://localhost:8001/mcp/v1/health
```

### Metrics

```bash
curl http://localhost:8001/mcp/v1/metrics
```

### Grafana Dashboard

If using Docker Compose:
1. Open http://localhost:3000
2. Login with admin/admin
3. View MCP Server dashboard

## Troubleshooting

### Port Already in Use

```bash
# Check what's using the port
lsof -i :8001

# Kill the process or change port
export MCP_PORT=8002
```

### Connection Refused

```bash
# Check if server is running
curl http://localhost:8001/mcp/v1/health

# Check logs
docker logs signalcli-mcp
# or
tail -f logs/mcp.log
```

### Slow Performance

1. Enable caching:
   ```bash
   export CACHE_PROVIDER=redis
   ```

2. Increase workers:
   ```bash
   export MCP_WORKERS=8
   ```

3. Use load balancing:
   ```python
   # In config
   routing:
     strategy: load_balanced
   ```

## Next Steps

1. Read the [full documentation](mcp_server.md)
2. Try the [advanced examples](../examples/mcp_advanced_examples.py)
3. Build your own tools
4. Integrate with your AI assistant
