# OpenRuntime Dockerfile
# Author: Nik Jois <nikjois@llamasearch.ai>
# Version: 2.0.0
# 
# Multi-stage build for OpenRuntime GPU Computing Platform
# Supports Apple Silicon (M1/M2/M3) with MLX Metal integration

# Use Python 3.11 slim as base image
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r openruntime && \
    useradd -r -g openruntime openruntime && \
    chown -R openruntime:openruntime /app

# Switch to non-root user
USER openruntime

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "openruntime.py", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install in development mode
RUN pip install -e .

# Expose additional ports for development
EXPOSE 8000 8080

# Development command with hot reload
CMD ["python", "openruntime.py", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Production stage
FROM base as production

# Install production-specific dependencies
RUN pip install gunicorn

# Copy production configuration
COPY docker/production.py /app/production.py

# Create production user
RUN groupadd -r openruntime-prod && \
    useradd -r -g openruntime-prod openruntime-prod && \
    chown -R openruntime-prod:openruntime-prod /app

USER openruntime-prod

# Production command with Gunicorn
CMD ["gunicorn", "openruntime:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "120"]

# Apple Silicon optimized stage
FROM python:3.11-slim as apple-silicon

# Set environment variables for Apple Silicon
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    ARCHFLAGS="-arch arm64"

# Set working directory
WORKDIR /app

# Install system dependencies optimized for ARM64
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    ca-certificates \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with ARM64 optimizations
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MLX for Apple Silicon (if available)
RUN pip install --no-cache-dir mlx || echo "MLX not available in this environment"

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r openruntime && \
    useradd -r -g openruntime openruntime && \
    chown -R openruntime:openruntime /app

USER openruntime

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for Apple Silicon
CMD ["python", "openruntime.py", "--host", "0.0.0.0", "--port", "8000"]

# Testing stage
FROM base as testing

# Install testing dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy test files
COPY tests/ /app/tests/
COPY pytest.ini /app/

# Create test user
RUN groupadd -r openruntime-test && \
    useradd -r -g openruntime-test openruntime-test && \
    chown -R openruntime-test:openruntime-test /app

USER openruntime-test

# Run tests
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]

# Benchmarking stage
FROM base as benchmark

# Install benchmarking dependencies
RUN pip install pytest-benchmark

# Copy benchmark scripts
COPY scripts/ /app/scripts/
COPY tests/ /app/tests/

# Create benchmark user
RUN groupadd -r openruntime-bench && \
    useradd -r -g openruntime-bench openruntime-bench && \
    chown -R openruntime-bench:openruntime-bench /app

USER openruntime-bench

# Run benchmarks
CMD ["python", "cli_simple.py", "benchmark", "--type", "comprehensive"]

# Monitoring stage
FROM base as monitoring

# Install monitoring dependencies
RUN pip install prometheus-client psutil

# Copy monitoring configuration
COPY monitoring/ /app/monitoring/

# Create monitoring user
RUN groupadd -r openruntime-monitor && \
    useradd -r -g openruntime-monitor openruntime-monitor && \
    chown -R openruntime-monitor:openruntime-monitor /app

USER openruntime-monitor

# Expose monitoring port
EXPOSE 8000 9090

# Start monitoring
CMD ["python", "openruntime.py", "--host", "0.0.0.0", "--port", "8000"]

# Multi-platform build stage
FROM base as multi-platform

# Install platform-specific dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Platform-specific optimizations
RUN if [ "$(uname -m)" = "aarch64" ]; then \
        echo "ARM64 detected - installing MLX"; \
        pip install mlx; \
    elif [ "$(uname -m)" = "x86_64" ]; then \
        echo "x86_64 detected - installing CPU optimizations"; \
        pip install intel-openmp; \
    fi

# Copy application code
COPY . .

# Create user
RUN groupadd -r openruntime && \
    useradd -r -g openruntime openruntime && \
    chown -R openruntime:openruntime /app

USER openruntime

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "openruntime.py", "--host", "0.0.0.0", "--port", "8000"]

# Labels
LABEL maintainer="Nik Jois <nikjois@llamasearch.ai>" \
      version="2.0.0" \
      description="OpenRuntime GPU Computing Platform" \
      org.opencontainers.image.title="OpenRuntime" \
      org.opencontainers.image.description="Advanced GPU Runtime System for macOS with MLX Metal Integration" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Nik Jois <nikjois@llamasearch.ai>" \
      org.opencontainers.image.url="https://github.com/openruntime/openruntime" \
      org.opencontainers.image.source="https://github.com/openruntime/openruntime" \
      org.opencontainers.image.licenses="MIT"