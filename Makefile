.PHONY: help install dev-install test lint format type-check clean run-api run-cli docker-build docker-up docker-down

# Default target
.DEFAULT_GOAL := help

# Help command
help:
	@echo "SignalCLI Development Commands"
	@echo "=============================="
	@echo "install          Install production dependencies"
	@echo "dev-install      Install development dependencies"
	@echo "test             Run all tests"
	@echo "test-unit        Run unit tests only"
	@echo "test-integration Run integration tests only"
	@echo "lint             Run linting checks"
	@echo "format           Format code with black"
	@echo "type-check       Run type checking with mypy"
	@echo "clean            Clean build artifacts"
	@echo "run-api          Run API server locally"
	@echo "run-cli          Run CLI interface"
	@echo "docker-build     Build Docker images"
	@echo "docker-up        Start Docker services"
	@echo "docker-down      Stop Docker services"

# Installation
install:
	pip install -r requirements.txt

dev-install: install
	pip install black ruff mypy pre-commit
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

# Code quality
lint:
	ruff check src tests

format:
	black src tests

type-check:
	mypy src --ignore-missing-imports

# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache
	rm -rf dist build *.egg-info

# Running locally
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-cli:
	@python -m src.cli.main $(filter-out $@,$(MAKECMDGOALS))

# Docker commands
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Development workflow
dev: dev-install
	@echo "Development environment ready!"

check: lint type-check test
	@echo "All checks passed!"

# Prevent Make from trying to interpret extra arguments
%:
	@: