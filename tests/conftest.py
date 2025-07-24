#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for OpenRuntime Enhanced tests
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from openruntime_enhanced.enhanced import app


@pytest.fixture
def client():
    """Create synchronous test client"""
    return TestClient(app)


@pytest.fixture
def async_client():
    """Create asynchronous test client using TestClient"""
    return TestClient(app)


# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for consistent testing"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("GPU_FALLBACK_TO_CPU", "true")
