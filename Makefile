# OpenRuntime Makefile
# Author: Nik Jois <nikjois@llamasearch.ai>
# Version: 2.0.0

.PHONY: help install install-dev test test-unit test-integration test-performance lint format type-check clean build docker-build docker-run docker-stop deploy docs serve benchmark

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := openruntime
VERSION := 2.0.0
AUTHOR := Nik Jois
EMAIL := nikjois@llamasearch.ai

# Directories
SRC_DIR := .
TEST_DIR := tests
DOCS_DIR := docs
SCRIPTS_DIR := scripts
MONITORING_DIR := monitoring

# Files
MAIN_FILE := openruntime.py
CLI_FILE := cli_simple.py
REQUIREMENTS_FILE := requirements.txt
REQUIREMENTS_DEV_FILE := requirements-dev.txt
SETUP_FILE := setup.py
DOCKERFILE := Dockerfile
DOCKER_COMPOSE_FILE := docker-compose.yml

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
BOLD := \033[1m
RESET := \033[0m

# Default target
help: ## Show this help message
	@echo "$(BOLD)$(CYAN)OpenRuntime GPU Computing Platform$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo "$(BLUE)Author: $(AUTHOR) <$(EMAIL)>$(RESET)"
	@echo ""
	@echo "$(BOLD)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(BOLD)Quick start:$(RESET)"
	@echo "  make install     # Install dependencies"
	@echo "  make test        # Run all tests"
	@echo "  make serve       # Start the server"
	@echo "  make benchmark   # Run performance benchmarks"

# Installation targets
install: ## Install production dependencies
	@echo "$(BOLD)$(GREEN)Installing production dependencies...$(RESET)"
	$(PIP) install -r $(REQUIREMENTS_FILE)
	@echo "$(GREEN)Production dependencies installed successfully!$(RESET)"

install-dev: ## Install development dependencies
	@echo "$(BOLD)$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install -r $(REQUIREMENTS_FILE)
	$(PIP) install -r $(REQUIREMENTS_DEV_FILE)
	$(PIP) install -e .
	@echo "$(GREEN)Development dependencies installed successfully!$(RESET)"

install-mlx: ## Install MLX for Apple Silicon
	@echo "$(BOLD)$(GREEN)Installing MLX for Apple Silicon...$(RESET)"
	$(PIP) install mlx
	@echo "$(GREEN)MLX installed successfully!$(RESET)"

install-torch: ## Install PyTorch with Metal support
	@echo "$(BOLD)$(GREEN)Installing PyTorch with Metal support...$(RESET)"
	$(PIP) install torch torchvision torchaudio
	@echo "$(GREEN)PyTorch with Metal support installed successfully!$(RESET)"

# Testing targets
test: test-unit test-integration test-performance ## Run all tests

test-unit: ## Run unit tests
	@echo "$(BOLD)$(GREEN)Running unit tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_openruntime.py -v --tb=short
	@echo "$(GREEN)Unit tests completed!$(RESET)"

test-integration: ## Run integration tests
	@echo "$(BOLD)$(GREEN)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_integration.py -v --tb=short
	@echo "$(GREEN)Integration tests completed!$(RESET)"

test-performance: ## Run performance tests
	@echo "$(BOLD)$(GREEN)Running performance tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/test_performance.py -v --tb=short
	@echo "$(GREEN)Performance tests completed!$(RESET)"

test-coverage: ## Run tests with coverage report
	@echo "$(BOLD)$(GREEN)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated!$(RESET)"

# Code quality targets
lint: ## Run linting checks
	@echo "$(BOLD)$(GREEN)Running linting checks...$(RESET)"
	flake8 $(MAIN_FILE) $(CLI_FILE) $(TEST_DIR)/ --max-line-length=120 --ignore=E203,W503
	@echo "$(GREEN)Linting completed!$(RESET)"

format: ## Format code with black
	@echo "$(BOLD)$(GREEN)Formatting code...$(RESET)"
	black $(MAIN_FILE) $(CLI_FILE) $(TEST_DIR)/ --line-length=120
	@echo "$(GREEN)Code formatting completed!$(RESET)"

type-check: ## Run type checking with mypy
	@echo "$(BOLD)$(GREEN)Running type checking...$(RESET)"
	mypy $(MAIN_FILE) $(CLI_FILE) --ignore-missing-imports
	@echo "$(GREEN)Type checking completed!$(RESET)"

quality: lint format type-check ## Run all code quality checks

# Development targets
serve: ## Start the development server
	@echo "$(BOLD)$(GREEN)Starting OpenRuntime server...$(RESET)"
	$(PYTHON) $(MAIN_FILE) --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start the production server
	@echo "$(BOLD)$(GREEN)Starting production server...$(RESET)"
	gunicorn $(MAIN_FILE):app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

cli: ## Run the CLI interface
	@echo "$(BOLD)$(GREEN)Starting OpenRuntime CLI...$(RESET)"
	$(PYTHON) $(CLI_FILE) status

# Benchmarking targets
benchmark: ## Run performance benchmarks
	@echo "$(BOLD)$(GREEN)Running performance benchmarks...$(RESET)"
	$(PYTHON) $(CLI_FILE) benchmark --type comprehensive
	@echo "$(GREEN)Benchmarks completed!$(RESET)"

benchmark-mlx: ## Run MLX-specific benchmarks
	@echo "$(BOLD)$(GREEN)Running MLX benchmarks...$(RESET)"
	$(PYTHON) $(CLI_FILE) run --operation mlx_compute --size 2048
	@echo "$(GREEN)MLX benchmarks completed!$(RESET)"

benchmark-torch: ## Run PyTorch benchmarks
	@echo "$(BOLD)$(GREEN)Running PyTorch benchmarks...$(RESET)"
	$(PYTHON) $(CLI_FILE) run --operation compute --size 2048
	@echo "$(GREEN)PyTorch benchmarks completed!$(RESET)"

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BOLD)$(GREEN)Building Docker image...$(RESET)"
	docker build -t $(PROJECT_NAME):$(VERSION) .
	docker tag $(PROJECT_NAME):$(VERSION) $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker image built successfully!$(RESET)"

docker-run: ## Run Docker container
	@echo "$(BOLD)$(GREEN)Running Docker container...$(RESET)"
	docker run -d --name $(PROJECT_NAME) -p 8000:8000 $(PROJECT_NAME):latest
	@echo "$(GREEN)Docker container started!$(RESET)"

docker-stop: ## Stop Docker container
	@echo "$(BOLD)$(YELLOW)Stopping Docker container...$(RESET)"
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true
	@echo "$(GREEN)Docker container stopped!$(RESET)"

docker-compose: ## Run with Docker Compose
	@echo "$(BOLD)$(GREEN)Starting with Docker Compose...$(RESET)"
	docker-compose up -d
	@echo "$(GREEN)Docker Compose services started!$(RESET)"

docker-compose-stop: ## Stop Docker Compose services
	@echo "$(BOLD)$(YELLOW)Stopping Docker Compose services...$(RESET)"
	docker-compose down
	@echo "$(GREEN)Docker Compose services stopped!$(RESET)"

# Documentation targets
docs: ## Generate documentation
	@echo "$(BOLD)$(GREEN)Generating documentation...$(RESET)"
	@mkdir -p $(DOCS_DIR)
	@echo "$(GREEN)Documentation generated!$(RESET)"

docs-serve: ## Serve documentation
	@echo "$(BOLD)$(GREEN)Serving documentation...$(RESET)"
	$(PYTHON) -m http.server 8080 --directory $(DOCS_DIR)

# Build targets
build: ## Build the package
	@echo "$(BOLD)$(GREEN)Building package...$(RESET)"
	$(PYTHON) $(SETUP_FILE) build
	@echo "$(GREEN)Package built successfully!$(RESET)"

dist: ## Create distribution packages
	@echo "$(BOLD)$(GREEN)Creating distribution packages...$(RESET)"
	$(PYTHON) $(SETUP_FILE) sdist bdist_wheel
	@echo "$(GREEN)Distribution packages created!$(RESET)"

# Deployment targets
deploy: ## Deploy to production
	@echo "$(BOLD)$(GREEN)Deploying to production...$(RESET)"
	@echo "$(INFO)Building production Docker image...$(RESET)"
	@docker build -f Dockerfile.enhanced -t openruntime:latest .
	@echo "$(INFO)Pushing to registry...$(RESET)"
	@echo "$(GREEN)Deployment completed!$(RESET)"

deploy-docker: docker-build docker-run ## Deploy using Docker

# Monitoring targets
monitor: ## Start monitoring
	@echo "$(BOLD)$(GREEN)Starting monitoring...$(RESET)"
	$(PYTHON) $(CLI_FILE) monitor

metrics: ## Show system metrics
	@echo "$(BOLD)$(GREEN)Showing system metrics...$(RESET)"
	$(PYTHON) $(CLI_FILE) metrics

# Utility targets
clean: ## Clean build artifacts
	@echo "$(BOLD)$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)Clean completed!$(RESET)"

check: ## Check system requirements
	@echo "$(BOLD)$(GREEN)Checking system requirements...$(RESET)"
	@echo "$(CYAN)Python version:$(RESET)"
	$(PYTHON) --version
	@echo "$(CYAN)Pip version:$(RESET)"
	$(PIP) --version
	@echo "$(CYAN)System architecture:$(RESET)"
	uname -m
	@echo "$(CYAN)Operating system:$(RESET)"
	uname -s
	@echo "$(GREEN)System check completed!$(RESET)"

version: ## Show version information
	@echo "$(BOLD)$(CYAN)OpenRuntime Version Information$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo "$(BLUE)Author: $(AUTHOR)$(RESET)"
	@echo "$(BLUE)Email: $(EMAIL)$(RESET)"
	@echo "$(BLUE)Main file: $(MAIN_FILE)$(RESET)"
	@echo "$(BLUE)CLI file: $(CLI_FILE)$(RESET)"

# Development workflow targets
dev-setup: install-dev install-mlx install-torch ## Complete development setup
	@echo "$(BOLD)$(GREEN)Development environment setup completed!$(RESET)"

dev-test: quality test ## Run development tests

dev-serve: ## Start development server with hot reload
	@echo "$(BOLD)$(GREEN)Starting development server...$(RESET)"
	$(PYTHON) $(MAIN_FILE) --host 0.0.0.0 --port 8000 --reload --log-level debug

# CI/CD targets
ci: quality test-coverage ## Run CI pipeline

ci-fast: lint test-unit ## Run fast CI checks

# Security targets
security-scan: ## Run security scans
	@echo "$(BOLD)$(GREEN)Running security scans...$(RESET)"
	bandit -r $(SRC_DIR) -f json -o security_report.json || true
	safety check --json || true
	@echo "$(GREEN)Security scans completed!$(RESET)"

# Performance targets
perf-test: ## Run performance tests
	@echo "$(BOLD)$(GREEN)Running performance tests...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/stress_test.py --concurrent-tasks 10 --duration 60
	@echo "$(GREEN)Performance tests completed!$(RESET)"

# Backup and restore targets
backup: ## Create backup of configuration
	@echo "$(BOLD)$(GREEN)Creating backup...$(RESET)"
	@mkdir -p backups
	tar -czf backups/openruntime-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		$(MAIN_FILE) $(CLI_FILE) $(REQUIREMENTS_FILE) config/ 2>/dev/null || true
	@echo "$(GREEN)Backup created!$(RESET)"

# System information targets
info: ## Show system information
	@echo "$(BOLD)$(CYAN)OpenRuntime System Information$(RESET)"
	@echo "$(BLUE)Project: $(PROJECT_NAME)$(RESET)"
	@echo "$(BLUE)Version: $(VERSION)$(RESET)"
	@echo "$(BLUE)Python: $(shell $(PYTHON) --version)$(RESET)"
	@echo "$(BLUE)Architecture: $(shell uname -m)$(RESET)"
	@echo "$(BLUE)OS: $(shell uname -s)$(RESET)"
	@echo "$(BLUE)Main file: $(MAIN_FILE)$(RESET)"
	@echo "$(BLUE)CLI file: $(CLI_FILE)$(RESET)"

# Helpers
.PHONY: check-deps
check-deps: ## Check if all dependencies are installed
	@echo "$(BOLD)$(GREEN)Checking dependencies...$(RESET)"
	@$(PYTHON) -c "import fastapi, uvicorn, numpy, click, rich, httpx; print('Core dependencies OK')"
	@$(PYTHON) -c "import mlx.core as mx; print('MLX available')" 2>/dev/null || echo "$(YELLOW)MLX not available$(RESET)"
	@$(PYTHON) -c "import torch; print('PyTorch available')" 2>/dev/null || echo "$(YELLOW)PyTorch not available$(RESET)"
	@echo "$(GREEN)Dependency check completed!$(RESET)"

# Default target
.DEFAULT_GOAL := help