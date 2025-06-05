# Multi-stage Dockerfile for OpenRuntime Enhanced
# Optimized for Apple Silicon and x86_64 architectures

# =============================================================================
# Base Stage - Common dependencies
# =============================================================================
FROM python:3.11-slim as base

LABEL maintainer="OpenRuntime Team <team@openruntime.example.com>"
LABEL description="OpenRuntime Enhanced - Advanced GPU Runtime System"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app

# =============================================================================
# Rust Build Stage
# =============================================================================
FROM base as rust-builder

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -