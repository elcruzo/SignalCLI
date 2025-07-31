# SignalCLI Setup Guide

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended for LLM)
- GPU with CUDA support (optional, for faster inference)

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd SignalCLI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a local configuration file:

```bash
cp config/settings.yaml config/settings.local.yaml
```

Edit `config/settings.local.yaml` with your settings:

```yaml
llm:
  model_type: "mock"  # Use "llamafile" for real model
  model_path: "models/llama-3.1-8b-instruct.gguf"
  
vector_store:
  provider: "memory"  # Use "weaviate" for production
```

### 3. Running with Docker

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# View logs
docker-compose logs -f signalcli-api
```

### 4. Running Locally

```bash
# Start Weaviate (if using)
docker-compose up -d weaviate

# Run API server
python -m uvicorn src.api.main:app --reload

# In another terminal, use CLI
python -m src.cli.main "What is machine learning?"
```

## Development Setup

### 1. Install Development Dependencies

```bash
pip install -r requirements-dev.txt
pre-commit install
```

### 2. Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_llm_engine.py -v
```

### 3. Code Quality

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type checking
mypy src
```

## Production Deployment

### 1. Download LLM Model

```bash
# Create models directory
mkdir -p models

# Download GGUF model (example)
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf \
  -O models/llama-2-7b-chat.gguf
```

### 2. Environment Variables

Create `.env` file:

```bash
SIGNALCLI_API_HOST=0.0.0.0
SIGNALCLI_API_PORT=8000
SIGNALCLI_LLM_MODEL_PATH=/app/models/llama-2-7b-chat.gguf
SIGNALCLI_VECTOR_HOST=weaviate
SIGNALCLI_VECTOR_PORT=8080
SIGNALCLI_LOG_LEVEL=INFO
```

### 3. Deploy with Docker

```bash
# Build production image
docker build -t signalcli:latest .

# Run with docker-compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 4. Initialize Vector Store

```bash
# Add sample documents
docker-compose exec signalcli-api python -m src.utils.init_db
```

## Monitoring

### 1. Health Check

```bash
curl http://localhost:8000/health
```

### 2. Metrics

```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# View in Prometheus
open http://localhost:9090

# View in Grafana
open http://localhost:3000  # admin/admin
```

### 3. Logs

```bash
# API logs
docker-compose logs -f signalcli-api

# All logs
docker-compose logs -f
```

## Troubleshooting

### Issue: Out of Memory

```bash
# Reduce model size or use quantization
llm:
  gpu_layers: 20  # Reduce from 35
  max_tokens: 1024  # Reduce from 2048
```

### Issue: Slow Inference

```bash
# Enable GPU acceleration
llm:
  use_gpu: true
  gpu_layers: 35
```

### Issue: Vector Store Connection

```bash
# Check Weaviate status
curl http://localhost:8080/v1/.well-known/ready

# Restart Weaviate
docker-compose restart weaviate
```

### Issue: Port Conflicts

```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # API
  - "8081:8080"  # Weaviate
```

## Advanced Configuration

### Custom Schemas

Create JSON schemas in `data/schemas/`:

```json
{
  "name": "qa_response",
  "description": "Q&A response format",
  "schema": {
    "type": "object",
    "properties": {
      "answer": {"type": "string"},
      "confidence": {"type": "number"},
      "sources": {
        "type": "array",
        "items": {"type": "string"}
      }
    },
    "required": ["answer", "confidence"]
  }
}
```

### Performance Tuning

1. **Caching**: Enable Redis caching
2. **Batch Processing**: Use batch endpoints for multiple queries
3. **Model Optimization**: Use quantized models (Q4_K_M, Q5_K_M)
4. **Connection Pooling**: Configure in `settings.yaml`

## Security

1. **API Keys**: Enable authentication in production
2. **HTTPS**: Use reverse proxy (nginx/traefik)
3. **Rate Limiting**: Configure in API settings
4. **Input Validation**: Automatic with Pydantic models

## Support

- Documentation: `/docs` (API docs)
- Issues: GitHub Issues
- Logs: Check `/app/logs/` in container