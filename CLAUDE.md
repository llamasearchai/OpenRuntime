# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
OpenRuntime is an advanced GPU runtime system for macOS with Apple Silicon integration, combining GPU computing (MLX Metal), AI integration (OpenAI API, LangChain), and comprehensive performance monitoring.

## Commands

### Development Environment Setup
```bash
# Install dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running the Application
```bash
# Start the server
python -m openruntime.main

# Use CLI
python openruntime_cli.py [command]

# Run with uvicorn (development)
uvicorn openruntime.main:app --reload --host 0.0.0.0 --port 8000

# Production with gunicorn
gunicorn openruntime.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Testing
```bash
# Run all tests with coverage
python -m pytest tests/ -v --tb=short --cov=openruntime --cov-report=term-missing

# Run specific test file
python -m pytest tests/test_openruntime.py -v

# Run specific test
python -m pytest tests/test_integration.py::TestSystemIntegration::test_health_check -v

# Run tests excluding slow/integration tests
python -m pytest tests/ -v -k "not (test_list_ai_agents or test_generate_insights or test_metrics_endpoint)"

# Show available tests
python -m pytest tests/ --collect-only
```

### Code Quality
```bash
# Format code
black openruntime/ tests/
isort openruntime/ tests/

# Lint
flake8 openruntime/ tests/
mypy openruntime/

# Security check
bandit -r openruntime/
safety check

# All quality checks (as run in CI)
black --check openruntime/ tests/
isort --check-only openruntime/ tests/
flake8 openruntime/ tests/
mypy openruntime/
bandit -r openruntime/
```

### Rust Components
```bash
# Build Rust components
cd rust-openai-crate && cargo build --release

# Run Rust tests
cd rust-openai-crate && cargo test

# Lint Rust code
cd rust-openai-crate && cargo clippy -- -D warnings
```

### Docker
```bash
# Build container
docker build -t openruntime:latest .

# Run container
docker run -p 8000:8000 openruntime:latest
```

## Architecture

### Core Components

**openruntime/core/** - Core API and runtime managers
- `api.py`: FastAPI application setup with WebSocket support
- `manager.py`: GPU runtime management, task scheduling
- `metrics.py`: Prometheus metrics collection

**openruntime/security/** - Security components
- Rate limiting, input validation, authentication

**openruntime_enhanced/** - Enhanced version with AI features
- AI agent integration, workflow types (system analysis, code generation)
- LangChain integration for advanced AI workflows

**rust-openai-crate/** - Performance-critical Rust components
- High-performance OpenAI client implementation
- Used for compute-intensive operations

### Key Design Patterns

1. **Device Abstraction**: Unified interface supporting MLX Metal (Apple Silicon), PyTorch MPS, and CPU fallback
2. **Async Architecture**: FastAPI with async/await throughout, WebSocket streaming for real-time updates
3. **Configuration-Driven**: YAML configurations for deployment flexibility
4. **Hybrid Language**: Python for high-level logic, Rust for performance-critical paths

### GPU Runtime Flow
1. Request arrives at FastAPI endpoint
2. Manager selects appropriate device (MLX/PyTorch/CPU)
3. Task scheduled asynchronously
4. Results streamed via WebSocket or returned via REST
5. Metrics collected to Prometheus

### AI Integration Architecture
- **Workflow Types**: System analysis, code generation, shell commands
- **Agent Roles**: Developer, analyst, architect, tester
- **Provider Abstraction**: Currently OpenAI, designed for multi-provider support

## Important Considerations

### Apple Silicon Optimization
- Primary target is macOS with Apple Silicon (M1/M2/M3)
- MLX is preferred for GPU operations on Apple Silicon
- Automatic fallback to PyTorch MPS or CPU when MLX unavailable

### Testing Strategy
- Tests use async pytest fixtures
- Mock external services (OpenAI API) in tests
- Performance benchmarks in `tests/test_performance.py`
- Integration tests require running server

### CI/CD Pipeline
- GitHub Actions workflow in `.github/workflows/`
- Multi-stage: lint → test → build → deploy
- Container registry: ghcr.io/nemesis-collective/openruntime
- Automated security scanning with Trivy

### Monitoring
- Prometheus metrics exposed at `/metrics`
- Grafana dashboards in `monitoring/grafana/`
- Custom metrics for GPU utilization, task performance

### Environment Variables
- `OPENAI_API_KEY`: Required for AI features
- `MLFLOW_TRACKING_URI`: Optional MLflow integration
- `PROMETHEUS_MULTIPROC_DIR`: For metrics in multiprocess mode