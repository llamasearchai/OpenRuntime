# OpenRuntime Enhanced - Development and Deployment Makefile

.PHONY: help install dev test build docker-build docker-run clean lint format type-check security-check docs benchmark stress-test deploy stop logs

# Default target
help:
	@echo "OpenRuntime Enhanced - Available Commands"
	@echo "============================================"
	@echo ""
	@echo "Development:"
	@echo "  install        Install dependencies"
	@echo "  dev            Start development server"
	@echo "  test           Run test suite"
	@echo "  lint           Run code linting"
	@echo "  format         Format code"
	@echo "  type-check     Run type checking"
	@echo "  security-check Run security analysis"
	@echo ""
	@echo "Building & Deployment:"
	@echo "  build          Build application"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-run     Run with Docker Compose"
	@echo "  deploy         Deploy to production"
	@echo "  stop           Stop all services"
	@echo ""
	@echo "Testing & Monitoring:"
	@echo "  benchmark      Run performance benchmarks"
	@echo "  stress-test    Run stress tests"
	@echo "  logs           View application logs"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean build artifacts"
	@echo "  docs           Generate documentation"
	@echo ""

# =============================================================================
# Variables
# =============================================================================
PYTHON := python3
PIP := pip3
DOCKER := docker
DOCKER_COMPOSE := docker-compose
PROJECT_NAME := openruntime-enhanced
VERSION := $(shell grep "version=" setup.py | cut -d'"' -f2)

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# Development Setup
# =============================================================================
install:
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	@echo "$(GREEN)Dependencies installed$(NC)"

install-rust:
	@echo "$(BLUE)Installing Rust dependencies...$(NC)"
	cd rust-openai-crate && cargo build --release
	@echo "$(GREEN)Rust components built$(NC)"

dev: install
	@echo "$(BLUE)Starting development server...$(NC)"
	$(PYTHON) openruntime_enhanced.py --host 0.0.0.0 --port 8000 --reload --log-level debug

dev-docker:
	@echo "$(BLUE)Starting development environment with Docker...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.override.yml up --build

# =============================================================================
# Testing
# =============================================================================
test:
	@echo "$(BLUE)Running test suite...$(NC)"
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "$(GREEN)Tests completed$(NC)"

test-fast:
	@echo "$(BLUE)Running fast tests only...$(NC)"
	$(PYTHON) -m pytest tests/ -v -m "not slow"

test-integration:
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTHON) -m pytest tests/integration/ -v

benchmark:
	@echo "$(BLUE)Running performance benchmarks...$(NC)"
	chmod +x scripts/benchmark.sh
	./scripts/benchmark.sh
	@echo "$(GREEN)Benchmarks completed$(NC)"

stress-test:
	@echo "$(BLUE)Running stress tests...$(NC)"
	$(PYTHON) scripts/stress_test.py --concurrent 20 --total 1000
	@echo "$(GREEN)Stress tests completed$(NC)"

# =============================================================================
# Code Quality
# =============================================================================
lint:
	@echo "$(BLUE)Running code linting...$(NC)"
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "$(GREEN)Linting completed$(NC)"

format:
	@echo "$(BLUE)Formatting code...$(NC)"
	black . --line-length 127
	isort . --profile black
	@echo "$(GREEN)Code formatted$(NC)"

type-check:
	@echo "$(BLUE)Running type checking...$(NC)"
	mypy . --ignore-missing-imports
	@echo "$(GREEN)Type checking completed$(NC)"

security-check:
	@echo "$(BLUE)Running security analysis...$(NC)"
	bandit -r . -f json -o security-report.json
	safety check --json --output safety-report.json
	@echo "$(GREEN)Security analysis completed$(NC)"

# =============================================================================
# Building
# =============================================================================
build:
	@echo "$(BLUE)Building application...$(NC)"
	$(PYTHON) setup.py sdist bdist_wheel
	@echo "$(GREEN)Build completed$(NC)"

docker-build:
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER) build -t $(PROJECT_NAME):$(VERSION) -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)Docker images built$(NC)"

docker-build-dev:
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build --target development -t $(PROJECT_NAME):dev .

# =============================================================================
# Deployment
# =============================================================================
docker-run:
	@echo "$(BLUE)Starting services with Docker Compose...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Services started$(NC)"
	@echo "$(YELLOW)Application: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"

deploy: docker-build
	@echo "$(BLUE)Deploying to production...$(NC)"
	$(DOCKER_COMPOSE) -f docker-compose.yml up -d
	@echo "$(GREEN)Deployment completed$(NC)"

stop:
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)Services stopped$(NC)"

restart: stop docker-run

# =============================================================================
# Monitoring & Logs
# =============================================================================
logs:
	@echo "$(BLUE)Showing application logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f openruntime

logs-all:
	@echo "$(BLUE)Showing all service logs...$(NC)"
	$(DOCKER_COMPOSE) logs -f

health-check:
	@echo "$(BLUE)Checking service health...$(NC)"
	curl -f http://localhost:8000/health || exit 1
	@echo "$(GREEN)Health check passed$(NC)"

# =============================================================================
# Documentation
# =============================================================================
docs:
	@echo "$(BLUE)Generating documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html
	@echo "$(GREEN)Documentation generated$(NC)"
	@echo "$(YELLOW)Docs available at: docs/_build/html/index.html$(NC)"

docs-serve:
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd docs/_build/html && $(PYTHON) -m http.server 8080

# =============================================================================
# Database Management
# =============================================================================
db-migrate:
	@echo "$(BLUE)Running database migrations...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)Migrations completed$(NC)"

db-reset:
	@echo "$(BLUE)Resetting database...$(NC)"
	$(DOCKER_COMPOSE) down postgres
	$(DOCKER) volume rm openruntime-enhanced_postgres_data
	$(DOCKER_COMPOSE) up -d postgres
	sleep 10
	$(MAKE) db-migrate
	@echo "$(GREEN)Database reset$(NC)"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf benchmark_results/
	@echo "$(GREEN)Cleanup completed$(NC)"

clean-docker:
	@echo "$(BLUE)Cleaning Docker resources...$(NC)"
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	$(DOCKER) system prune -f
	@echo "$(GREEN)Docker cleanup completed$(NC)"

# =============================================================================
# Development Utilities
# =============================================================================
shell:
	@echo "$(BLUE)Starting interactive shell...$(NC)"
	$(DOCKER_COMPOSE) exec openruntime $(PYTHON)

bash:
	@echo "$(BLUE)Starting bash shell in container...$(NC)"
	$(DOCKER_COMPOSE) exec openruntime bash

install-hooks:
	@echo "$(BLUE)Installing git hooks...$(NC)"
	pre-commit install
	@echo "$(GREEN)Git hooks installed$(NC)"

requirements:
	@echo "$(BLUE)Updating requirements...$(NC)"
	pip-compile requirements.in
	pip-compile requirements-dev.in
	@echo "$(GREEN)Requirements updated$(NC)"

# =============================================================================
# CI/CD Helpers
# =============================================================================
ci-test: install test lint type-check security-check
	@echo "$(GREEN)CI tests completed$(NC)"

ci-build: build docker-build
	@echo "$(GREEN)CI build completed$(NC)"

release:
	@echo "$(BLUE)Creating release...$(NC)"
	@read -p "Enter version number: " version; \
	git tag -a $$version -m "Release $$version"; \
	git push origin $$version
	@echo "$(GREEN)Release created$(NC)"

# =============================================================================
# Performance Testing
# =============================================================================
perf-test:
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(PYTHON) -m pytest tests/performance/ -v --benchmark-only
	@echo "$(GREEN)Performance tests completed$(NC)"

load-test:
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/load_test.py --host=http://localhost:8000
	@echo "$(GREEN)Load tests completed$(NC)"

# =============================================================================
# Special Targets
# =============================================================================
.ONESHELL:
setup-dev: install install-rust install-hooks
	@echo "$(GREEN)Development environment setup completed!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make dev' to start development server"
	@echo "  3. Visit http://localhost:8000 to see the application"

quick-start: docker-build docker-run
	@echo "$(GREEN)Quick start completed!$(NC)"
	@echo "$(YELLOW)Application: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API Docs: http://localhost:8000/docs$(NC)"
	@echo "$(YELLOW)Monitoring: http://localhost:3000$(NC)"

all-tests: test test-integration benchmark stress-test
	@echo "$(GREEN)All tests completed!$(NC)"