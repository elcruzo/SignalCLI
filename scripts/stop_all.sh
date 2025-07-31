#!/bin/bash
# Stop all SignalCLI services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Stopping SignalCLI services...${NC}"

# Stop Python services if running locally
if [ -f .api.pid ]; then
    API_PID=$(cat .api.pid)
    if kill -0 $API_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping API server (PID: $API_PID)...${NC}"
        kill $API_PID
        rm .api.pid
    fi
fi

if [ -f .mcp.pid ]; then
    MCP_PID=$(cat .mcp.pid)
    if kill -0 $MCP_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping MCP server (PID: $MCP_PID)...${NC}"
        kill $MCP_PID
        rm .mcp.pid
    fi
fi

# Stop Docker containers
if command -v docker &> /dev/null; then
    # Stop docker-compose services
    if [ -f docker-compose.mcp.yml ]; then
        echo -e "${YELLOW}Stopping Docker Compose services...${NC}"
        docker-compose -f docker-compose.mcp.yml down
    fi
    
    # Stop individual containers
    echo -e "${YELLOW}Stopping Docker containers...${NC}"
    docker stop signalcli-redis signalcli-weaviate 2>/dev/null || true
    docker rm signalcli-redis signalcli-weaviate 2>/dev/null || true
fi

echo -e "${GREEN}All services stopped.${NC}"
