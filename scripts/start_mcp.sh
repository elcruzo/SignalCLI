#!/bin/bash
# Start SignalCLI MCP Server

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting SignalCLI MCP Server...${NC}"

# Check if we're in the right directory
if [ ! -f "src/mcp/main.py" ]; then
    echo -e "${RED}Error: Must run from SignalCLI root directory${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.9"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies if needed
if ! pip show fastapi >/dev/null 2>&1; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${YELLOW}⚠ Redis is not running (caching will use in-memory fallback)${NC}"
    fi
fi

# Check if Weaviate is running (optional)
if curl -s http://localhost:8080/v1/.well-known/ready >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Weaviate is running${NC}"
else
    echo -e "${YELLOW}⚠ Weaviate is not running (will use in-memory vector store)${NC}"
fi

# Set environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export ENV=${ENV:-development}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MCP_HOST=${MCP_HOST:-0.0.0.0}
export MCP_PORT=${MCP_PORT:-8001}
export MCP_WORKERS=${MCP_WORKERS:-1}

echo -e "${GREEN}Configuration:${NC}"
echo "  Environment: $ENV"
echo "  Log Level: $LOG_LEVEL"
echo "  Host: $MCP_HOST"
echo "  Port: $MCP_PORT"
echo "  Workers: $MCP_WORKERS"

# Start the server
echo -e "${GREEN}Starting MCP server on http://$MCP_HOST:$MCP_PORT${NC}"
echo -e "${GREEN}API docs available at http://$MCP_HOST:$MCP_PORT/docs${NC}"
echo -e "${GREEN}MCP endpoints at http://$MCP_HOST:$MCP_PORT/mcp/v1/*${NC}"

if [ "$ENV" = "production" ]; then
    # Production mode with gunicorn
    exec gunicorn src.mcp.main:create_mcp_app \
        --workers $MCP_WORKERS \
        --worker-class uvicorn.workers.UvicornWorker \
        --bind $MCP_HOST:$MCP_PORT \
        --log-level $LOG_LEVEL \
        --access-logfile - \
        --error-logfile -
else
    # Development mode with auto-reload
    exec python -m src.mcp.main
fi
