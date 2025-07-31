# MCP Server Best Practices & Common Mistakes Avoided

## Overview

This document outlines the best practices we've followed in SignalCLI's MCP server implementation and common mistakes we've avoided based on lessons learned from other MCP servers.

## ✅ Best Practices We Follow

### 1. Protocol Compliance

**What we do right:**
- Strict adherence to MCP 2024-11-05 specification
- Proper JSON-RPC 2.0 implementation with correct error codes
- Session initialization requirement before tool calls
- Correct response format with content arrays

**Common mistake avoided:**
- Many servers skip initialization or allow tool calls without proper session setup
- Incorrect error response formats that don't follow JSON-RPC spec

### 2. Content Format

**What we do right:**
```json
// Correct MCP response format
{
  "content": [
    {
      "type": "text",
      "text": "Response text here"
    }
  ],
  "isError": false
}
```

**Common mistake avoided:**
- Returning raw strings or custom formats instead of content arrays
- Missing the `isError` field in tool results
- Not supporting multiple content items

### 3. Error Handling

**What we do right:**
- Proper error codes: parse errors (-32700), invalid requests (-32600), etc.
- MCP-specific error codes for tool-related issues
- Graceful error recovery without crashing sessions
- Error results still follow content array format

**Common mistake avoided:**
- Using HTTP status codes instead of JSON-RPC error codes
- Throwing exceptions that terminate WebSocket connections
- Inconsistent error formats

### 4. Tool Discovery

**What we do right:**
```json
{
  "tools": [
    {
      "name": "tool_name",
      "description": "Clear description",
      "inputSchema": {
        "type": "object",
        "properties": {...},
        "required": [...]
      }
    }
  ]
}
```

**Common mistake avoided:**
- Missing or incomplete input schemas
- No description for tools
- Custom schema formats instead of JSON Schema

### 5. Streaming Support

**What we do right:**
- Proper WebSocket session management
- Server-Sent Events as fallback
- Clean stream lifecycle (creation, progress, completion)
- Graceful handling of disconnections

**Common mistake avoided:**
- No streaming support at all
- Memory leaks from unclosed streams
- No progress notifications during long operations

### 6. State Management

**What we do right:**
- Proper session tracking with unique IDs
- Session cleanup on disconnect
- No global state pollution
- Thread-safe operations

**Common mistake avoided:**
- Mixing state between different clients
- Memory leaks from abandoned sessions
- Race conditions in concurrent requests

### 7. Performance

**What we do right:**
- Intelligent caching with proper invalidation
- Request deduplication
- Efficient tool routing
- Connection pooling for external services

**Common mistake avoided:**
- No caching leading to redundant computations
- Blocking operations in async handlers
- Creating new connections for every request

### 8. Security

**What we do right:**
- Proper authentication (when enabled)
- Rate limiting per client
- Input validation before tool execution
- No arbitrary code execution

**Common mistake avoided:**
- No authentication mechanism
- Allowing unlimited requests
- Trusting client input without validation
- Exposing internal errors to clients

## 📝 Implementation Checklist

### Protocol Layer
- [x] JSON-RPC 2.0 compliance
- [x] Proper request/response correlation with IDs
- [x] Support for notifications (no ID)
- [x] Batch request support (future)

### Tool Management
- [x] Dynamic tool registration
- [x] Tool capability metadata
- [x] Input/output schema validation
- [x] Tool versioning support

### Error Handling
- [x] Standard error codes
- [x] Custom error codes for domain-specific issues
- [x] Error recovery without session termination
- [x] Detailed error logging (server-side only)

### Performance
- [x] Response caching
- [x] Connection pooling
- [x] Async/await throughout
- [x] Resource cleanup

### Monitoring
- [x] Request/response logging
- [x] Performance metrics
- [x] Error tracking
- [x] Session analytics

## 🚫 Anti-Patterns to Avoid

### 1. Stateless Tool Execution
**Wrong:**
```python
# No session management
async def execute_tool(tool_name, args):
    return tool.execute(args)
```

**Right:**
```python
# Proper session context
async def execute_tool(session_id, tool_name, args):
    session = get_session(session_id)
    if not session.initialized:
        return error("Not initialized")
    return tool.execute(args, session.context)
```

### 2. Synchronous Blocking
**Wrong:**
```python
# Blocks event loop
def expensive_operation():
    time.sleep(5)
    return result
```

**Right:**
```python
# Non-blocking async
async def expensive_operation():
    await asyncio.sleep(5)
    return result
```

### 3. Global State
**Wrong:**
```python
# Shared mutable state
current_user = None
tool_cache = {}
```

**Right:**
```python
# Isolated session state
class Session:
    def __init__(self):
        self.user = None
        self.cache = {}
```

## 🚀 Advanced Features

### Context Propagation
```python
# Context flows through tool chain
context = {
    "session_id": "...",
    "user_preferences": {...},
    "previous_results": [...]
}
```

### Tool Composition
```python
# Tools can be composed
result1 = await tool1.execute(args1)
result2 = await tool2.execute({
    **args2,
    "input": result1["output"]
})
```

### Adaptive Routing
```python
# Router learns from usage patterns
router.update_stats(tool_name, latency, success)
best_tool = router.select_optimal_tool(request)
```

## 📊 Metrics to Track

1. **Protocol Metrics**
   - Request rate by method
   - Error rate by error code
   - Session duration
   - Active sessions

2. **Tool Metrics**
   - Execution time by tool
   - Success/failure rate
   - Cache hit rate
   - Resource usage

3. **Performance Metrics**
   - Response time percentiles
   - Throughput
   - Queue depth
   - Connection pool usage

## 🔍 Testing Strategy

### Protocol Compliance Tests
```python
# Test proper initialization flow
async def test_initialization_required():
    # Should fail without init
    response = await client.call("tools/list")
    assert response.error.code == -32600
    
    # Should work after init
    await client.call("initialize")
    response = await client.call("tools/list")
    assert response.result is not None
```

### Error Handling Tests
```python
# Test various error conditions
async def test_error_handling():
    # Invalid JSON
    response = await client.send_raw("invalid json")
    assert response.error.code == -32700
    
    # Unknown method
    response = await client.call("unknown/method")
    assert response.error.code == -32601
```

### Load Tests
```python
# Test concurrent connections
async def test_concurrent_sessions():
    sessions = []
    for i in range(100):
        session = await create_session()
        sessions.append(session)
    
    # All should work independently
    results = await asyncio.gather(*[
        session.call("tools/list") 
        for session in sessions
    ])
    assert all(r.result for r in results)
```

## 🎓 Lessons Learned

1. **Start with the spec**: Read and understand the MCP specification thoroughly
2. **Test protocol compliance**: Use official test suites if available
3. **Design for scale**: Consider thousands of concurrent sessions from day one
4. **Monitor everything**: You can't improve what you don't measure
5. **Fail gracefully**: Errors should be informative and recoverable
6. **Document thoroughly**: Both API docs and implementation guides
7. **Version carefully**: MCP protocol versions matter for compatibility

## 🔗 Resources

- [MCP Specification](https://modelcontextprotocol.io/docs)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
- [JSON Schema](https://json-schema.org/)
- [WebSocket Protocol](https://datatracker.ietf.org/doc/html/rfc6455)
- [Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html)
