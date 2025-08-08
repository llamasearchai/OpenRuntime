#!/usr/bin/env python3
"""
Integration tests for OpenRuntime Enhanced
"""

import asyncio
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from openruntime_enhanced import app

# Fixtures are now in conftest.py


class TestFullIntegration:
    """Full integration test suite"""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, async_client):
        """Test complete AI-enhanced GPU workflow"""
        # Step 1: Check system status
        async with async_client as client:
            response = await client.get("/")
            assert response.status_code == 200

            # Step 2: List available devices
            response = await client.get("/devices")
            assert response.status_code == 200
            devices = response.json()["devices"]
            assert len(devices) > 0

            # Step 3: Create enhanced task
            task_data = {
                "gpu_task": {
                    "operation": "benchmark",
                    "data": {"type": "comprehensive"},
                    "priority": 1,
                },
                "ai_task": {
                    "workflow_type": "system_analysis",
                    "prompt": "Analyze benchmark results and provide optimization recommendations",
                },
            }

            with patch("openruntime_enhanced.enhanced_runtime") as mock_runtime:

                async def mock_execute_ai_enhanced_task(*args, **kwargs):
                    return {
                        "gpu_task": {
                            "status": "completed",
                            "result": {
                                "benchmark_type": "comprehensive",
                                "scores": {"compute": 85.5},
                            },
                            "execution_time": 2.5,
                        },
                        "ai_analysis": {
                            "result": {
                                "analysis": "Good performance, consider GPU memory optimization"
                            },
                            "execution_time": 1.2,
                        },
                        "execution_time": 3.7,
                    }

                mock_runtime.execute_ai_enhanced_task = mock_execute_ai_enhanced_task

                response = await client.post("/tasks/enhanced", json=task_data)
                assert response.status_code == 200

                result = response.json()
                assert "gpu_task" in result
                assert "ai_analysis" in result

    @pytest.mark.asyncio
    async def test_ai_code_generation_integration(self, async_client):
        """Test AI code generation integration"""
        # Generate GPU kernel code
        async with async_client as client:
            code_request = {
                "language": "cuda",
                "description": "Matrix multiplication kernel optimized for GPU",
                "context": {
                    "target_gpu": "nvidia_rtx4090",
                    "matrix_size": "large",
                    "precision": "float32",
                },
                "optimization_target": "performance",
                "include_tests": True,
            }

            with patch("openruntime_enhanced.enhanced_runtime") as mock_runtime:

                async def mock_execute_ai_task(*args, **kwargs):
                    return {
                        "task_id": "code-gen-123",
                        "result": {
                            "type": "code_generation",
                            "code": "__global__ void matmul_kernel(...) { /* optimized code */ }",
                            "tests": "// Unit tests for kernel",
                            "documentation": "// Performance optimized matrix multiplication",
                        },
                    }

                mock_runtime.ai_manager.execute_ai_task = mock_execute_ai_task

                response = await client.post("/ai/code", json=code_request)
                assert response.status_code == 200

                result = response.json()
                assert "result" in result
                assert "code" in result["result"]

    class TestErrorHandling:
        """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_invalid_ai_task(self, async_client):
        """Test handling of invalid AI task requests"""
        async with async_client as client:
            invalid_task = {"workflow_type": "invalid_workflow", "prompt": "Test prompt"}

            response = await client.post("/ai/tasks", json=invalid_task)
            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_ai_service_unavailable(self, async_client):
        """Test handling when AI services are unavailable"""
        async with async_client as client:
            task_data = {"workflow_type": "system_analysis", "prompt": "Test prompt"}

            with patch("openruntime_enhanced.enhanced_runtime") as mock_runtime:

                async def mock_execute_ai_task(*args, **kwargs):
                    return {"error": "AI service unavailable", "task_id": "failed-123"}

                mock_runtime.ai_manager.execute_ai_task = mock_execute_ai_task

                response = await client.post("/ai/tasks", json=task_data)
                assert response.status_code == 200

                result = response.json()
                assert "error" in result

    @pytest.mark.asyncio
    async def test_performance_under_load(self, async_client):
        """Test performance under concurrent load"""
        async with async_client as client:
            tasks = []
            task_data = {"workflow_type": "system_analysis", "prompt": "Quick analysis"}

            # Create 5 concurrent tasks
            for _ in range(5):
                tasks.append(async_client.post("/ai/tasks", json=task_data))

            with patch("openruntime_enhanced.enhanced_runtime") as mock_runtime:

                async def mock_execute_ai_task(*args, **kwargs):
                    return {
                        "task_id": "test-task",
                        "result": {"analysis": "Test analysis"},
                        "execution_time": 0.5,
                    }

                mock_runtime.ai_manager.execute_ai_task = mock_execute_ai_task

                # Execute concurrent tasks
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successful responses
                successful_responses = [r for r in responses if not isinstance(r, Exception)]
                assert len(successful_responses) >= 3  # Allow some failures under load

    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection capabilities"""
        # This is a basic test to ensure WebSocket endpoint exists
        # Full WebSocket testing requires more complex setup
        async with async_client as client:
            response = await client.get("/")
            assert response.status_code == 200

            # Verify WebSocket is mentioned in the root response
            data = response.json()
            assert "websocket" in str(data).lower() or "ws" in str(data).lower()

    @pytest.mark.asyncio
    async def test_large_task_handling(self, async_client):
        """Test handling of large task requests"""
        large_task = {
            "workflow_type": "code_generation",
            "prompt": "Generate a complete machine learning pipeline " * 50,  # Large prompt
            "context": {
                "complexity": "high",
                "requirements": ["performance", "scalability", "maintainability"],
            },
            "max_tokens": 4000,
        }

        with patch("openruntime_enhanced.enhanced_runtime") as mock_runtime:

            async def mock_execute_ai_task(*args, **kwargs):
                return {
                    "task_id": "large-task-123",
                    "result": {
                        "type": "code_generation",
                        "code": "# Generated ML pipeline code",
                        "complexity_handled": True,
                    },
                    "execution_time": 15.0,
                }

            mock_runtime.ai_manager.execute_ai_task = mock_execute_ai_task

            response = await async_client.post("/ai/tasks", json=large_task)
            assert response.status_code == 200

            result = response.json()
            assert "result" in result
            assert "execution_time" in result
