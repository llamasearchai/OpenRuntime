.PHONY: help install install-dev test lint format clean run run-dev docker-build docker-run validate setup pre-commit docs

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
BLACK := $(PYTHON) -m black
ISORT := $(PYTHON) -m isort
FLAKE8 := $(PYTHON) -m flake8
MYPY := $(PYTHON) -m mypy
BANDIT := $(PYTHON) -m bandit

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)OpenRuntime Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e .

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	$(PIP) install pre-commit
	pre-commit install

install-all: ## Install all optional dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

setup: install-dev ## Complete development setup
	@echo "$(GREEN)Setting up OpenRuntime development environment...$(NC)"
	mkdir -p ~/.openruntime/cache
	mkdir -p ~/.openruntime/models
	@echo "$(GREEN)Setup complete!$(NC)"

test: ## Run tests with coverage
	$(PYTEST) tests/ -v --cov=openruntime --cov-report=term-missing

test-fast: ## Run tests without coverage
	$(PYTEST) tests/ -v -x

test-unit: ## Run unit tests only
	$(PYTEST) tests/ -v -m "unit"

test-integration: ## Run integration tests only
	$(PYTEST) tests/ -v -m "integration"

lint: ## Run all linters
	@echo "$(YELLOW)Running linters...$(NC)"
	$(BLACK) --check openruntime/ tests/
	$(ISORT) --check-only openruntime/ tests/
	$(FLAKE8) openruntime/ tests/ --max-line-length=100 --extend-ignore=E203,E402,F401,F841,E712,W293,W291,E722
	$(MYPY) openruntime/ --ignore-missing-imports
	$(BANDIT) -r openruntime/ --skip B101
	@echo "$(GREEN)All linters passed!$(NC)"

format: ## Format code with black and isort
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(BLACK) openruntime/ tests/
	$(ISORT) openruntime/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	@echo "$(GREEN)Cleaned!$(NC)"

run: ## Run the OpenRuntime server
	$(PYTHON) -m openruntime.main_v2

run-dev: ## Run the server in development mode with auto-reload
	$(PYTHON) -m openruntime.main_v2 --reload

run-cli: ## Run the CLI interface
	$(PYTHON) openruntime_cli_v2.py --help

validate: ## Validate all fixes and configurations
	@echo "$(YELLOW)Running validation...$(NC)"
	$(PYTHON) validate_fixes.py
	@echo "$(GREEN)Validation complete!$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks
	pre-commit autoupdate

docker-build: ## Build Docker image
	docker build -t openruntime:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 -e OPENAI_API_KEY=$(OPENAI_API_KEY) openruntime:latest

docs: ## Generate documentation
	@echo "$(YELLOW)Generating documentation...$(NC)"
	$(PYTHON) -m pydoc -w openruntime
	@echo "$(GREEN)Documentation generated!$(NC)"

spell-check: ## Run spell checker
	@echo "$(YELLOW)Running spell check...$(NC)"
	@if command -v cspell &> /dev/null; then \
		cspell "**/*.{py,md,yml,yaml,json}" --config ./cspell.json; \
	else \
		echo "$(RED)cSpell not installed. Install with: npm install -g cspell$(NC)"; \
	fi

version: ## Show version information
	@echo "$(GREEN)OpenRuntime v2.0.0$(NC)"
	@$(PYTHON) --version
	@echo "pip: $$($(PIP) --version)"

deps-update: ## Update all dependencies
	$(PIP) install --upgrade pip
	$(PIP) list --outdated
	$(PIP) install --upgrade -r requirements.txt

deps-tree: ## Show dependency tree
	@if command -v pipdeptree &> /dev/null; then \
		pipdeptree; \
	else \
		$(PIP) install pipdeptree && pipdeptree; \
	fi

security: ## Run security checks
	$(BANDIT) -r openruntime/ -f json -o security-report.json
	@if command -v safety &> /dev/null; then \
		safety check --json; \
	else \
		echo "$(YELLOW)Safety not installed. Install with: pip install safety$(NC)"; \
	fi

benchmark: ## Run performance benchmarks
	$(PYTHON) -m openruntime.benchmarks.run_benchmarks

monitor: ## Start monitoring dashboard
	@echo "$(YELLOW)Starting monitoring dashboard...$(NC)"
	@echo "Prometheus metrics available at: http://localhost:8000/v2/metrics"
	@echo "$(GREEN)Use Grafana or Prometheus to visualize metrics$(NC)"

all: clean install-dev lint test ## Run everything (clean, install, lint, test)
	@echo "$(GREEN)All tasks completed successfully!$(NC)"