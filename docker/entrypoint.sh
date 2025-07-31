#!/bin/bash
set -e

# SignalCLI Entrypoint Script

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name to be ready at $host:$port..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s --fail "http://$host:$port/v1/.well-known/ready" > /dev/null 2>&1; then
            log "$service_name is ready!"
            return 0
        fi
        
        log "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log "ERROR: $service_name failed to become ready after $max_attempts attempts"
    exit 1
}

# Default environment variables
export WEAVIATE_HOST=${WEAVIATE_HOST:-weaviate}
export WEAVIATE_PORT=${WEAVIATE_PORT:-8080}
export API_HOST=${API_HOST:-0.0.0.0}
export API_PORT=${API_PORT:-8000}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# Create necessary directories
mkdir -p /app/logs /app/data

log "Starting SignalCLI with command: $@"

case "$1" in
    "api")
        log "Starting API server mode..."
        
        # Wait for Weaviate to be ready
        wait_for_service "$WEAVIATE_HOST" "$WEAVIATE_PORT" "Weaviate"
        
        # Start the FastAPI server
        log "Starting FastAPI server on $API_HOST:$API_PORT"
        exec uvicorn src.api.main:app \
            --host "$API_HOST" \
            --port "$API_PORT" \
            --log-level "${LOG_LEVEL,,}" \
            --access-log
        ;;
    
    "cli")
        log "Starting CLI mode..."
        shift  # Remove 'cli' from arguments
        exec python src/cli/main.py "$@"
        ;;
    
    "worker")
        log "Starting background worker..."
        exec python src/worker/main.py
        ;;
    
    "init-db")
        log "Initializing database..."
        wait_for_service "$WEAVIATE_HOST" "$WEAVIATE_PORT" "Weaviate"
        exec python src/utils/init_db.py
        ;;
    
    "health-check")
        log "Running health check..."
        exec curl -f "http://localhost:$API_PORT/health" || exit 1
        ;;
    
    *)
        log "Unknown command: $1"
        log "Available commands: api, cli, worker, init-db, health-check"
        exit 1
        ;;
esac