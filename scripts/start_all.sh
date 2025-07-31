#!/bin/bash
# Start all SignalCLI services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   SignalCLI Full Stack Startup${NC}"
echo -e "${BLUE}======================================${NC}"

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local name=$1
    local url=$2
    local max_attempts=30
    local attempt=0
    
    echo -e "${YELLOW}Waiting for $name to be ready...${NC}"
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e "${GREEN}✓ $name is ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 1
    done
    
    echo -e "${RED}✗ $name failed to start${NC}"
    return 1
}

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker is running${NC}"

# Check if services are already running
if check_port 8080; then
    echo -e "${YELLOW}⚠ Port 8080 is already in use (Weaviate?)${NC}"
fi

if check_port 6379; then
    echo -e "${YELLOW}⚠ Port 6379 is already in use (Redis?)${NC}"
fi

if check_port 8000; then
    echo -e "${YELLOW}⚠ Port 8000 is already in use (API Server?)${NC}"
fi

if check_port 8001; then
    echo -e "${YELLOW}⚠ Port 8001 is already in use (MCP Server?)${NC}"
fi

# Start services based on argument
MODE=${1:-docker}

if [ "$MODE" = "docker" ]; then
    echo -e "${BLUE}Starting services with Docker Compose...${NC}"
    
    # Use the MCP docker-compose file
    docker-compose -f docker-compose.mcp.yml up -d
    
    # Wait for services
    wait_for_service "Weaviate" "http://localhost:8080/v1/.well-known/ready"
    wait_for_service "Redis" "http://localhost:6379" || true
    wait_for_service "API Server" "http://localhost:8000/health"
    wait_for_service "MCP Server" "http://localhost:8001/mcp/v1/health"
    
    echo -e "${GREEN}\nAll services started successfully!${NC}"
    echo -e "${BLUE}\nService URLs:${NC}"
    echo "  API Server: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo "  MCP Server: http://localhost:8001"
    echo "  MCP Docs: http://localhost:8001/docs"
    echo "  Weaviate: http://localhost:8080"
    echo "  Redis: localhost:6379"
    
    if [ -d "docker-compose.mcp.yml" ]; then
        echo "  Grafana: http://localhost:3000 (admin/admin)"
        echo "  Prometheus: http://localhost:9090"
    fi
    
elif [ "$MODE" = "local" ]; then
    echo -e "${BLUE}Starting services locally...${NC}"
    
    # Start infrastructure services with Docker
    echo -e "${YELLOW}Starting infrastructure services...${NC}"
    docker run -d --name signalcli-redis -p 6379:6379 redis:7-alpine >/dev/null 2>&1 || true
    docker run -d --name signalcli-weaviate -p 8080:8080 \
        -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
        -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
        -e DEFAULT_VECTORIZER_MODULE=none \
        semitechnologies/weaviate:latest >/dev/null 2>&1 || true
    
    # Wait for infrastructure
    wait_for_service "Weaviate" "http://localhost:8080/v1/.well-known/ready"
    
    # Start Python services
    echo -e "${YELLOW}Starting API server...${NC}"
    python -m src.api.main &
    API_PID=$!
    
    echo -e "${YELLOW}Starting MCP server...${NC}"
    python -m src.mcp.main &
    MCP_PID=$!
    
    # Save PIDs
    echo $API_PID > .api.pid
    echo $MCP_PID > .mcp.pid
    
    # Wait for services
    wait_for_service "API Server" "http://localhost:8000/health"
    wait_for_service "MCP Server" "http://localhost:8001/mcp/v1/health"
    
    echo -e "${GREEN}\nAll services started successfully!${NC}"
    echo -e "${YELLOW}\nTo stop services, run: ./scripts/stop_all.sh${NC}"
    
else
    echo -e "${RED}Unknown mode: $MODE${NC}"
    echo "Usage: $0 [docker|local]"
    exit 1
fi

echo -e "${BLUE}\n======================================${NC}"
echo -e "${GREEN}SignalCLI is ready!${NC}"
echo -e "${BLUE}======================================${NC}"
