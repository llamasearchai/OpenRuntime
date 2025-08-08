"""
Test AI-related endpoints for OpenRuntime.
"""

import pytest
from fastapi.testclient import TestClient

from openruntime.core.api import app


class TestAIEndpoints:
    """Test AI-related API endpoints."""

    def test_list_ai_agents(self, client):
        """Test listing available AI agents."""
        response = client.get("/ai/agents")
        assert response.status_code == 200
        data = response.json()

        assert "agents" in data
        assert isinstance(data["agents"], list)

        # Check agent structure
        if data["agents"]:
            agent = data["agents"][0]
            assert "role" in agent
            assert "name" in agent
            assert "description" in agent
            assert "capabilities" in agent

            # Verify known agent roles exist
            roles = [agent["role"] for agent in data["agents"]]
            expected_roles = ["DEVELOPER", "ANALYST", "OPTIMIZER", "DEBUGGER"]
            for role in expected_roles:
                assert role in roles

    def test_generate_insights(self, client):
        """Test AI insights generation."""
        request_data = {"metric_type": "performance", "timeframe": "1h"}

        response = client.post("/ai/insights", json=request_data)
        assert response.status_code == 200
        data = response.json()

        assert "insights" in data
        assert "metrics" in data
        assert "timestamp" in data

        # Verify metrics structure
        metrics = data["metrics"]
        assert "gpu_utilization" in metrics
        assert "memory_usage" in metrics
        assert "active_tasks" in metrics
        assert "ai_tasks_processed" in metrics

    def test_generate_insights_different_metrics(self, client):
        """Test AI insights with different metric types."""
        metric_types = ["performance", "memory", "efficiency", "errors"]
        timeframes = ["5m", "1h", "24h", "7d"]

        for metric_type in metric_types:
            for timeframe in timeframes[:2]:  # Test first two timeframes
                request_data = {"metric_type": metric_type, "timeframe": timeframe}

                response = client.post("/ai/insights", json=request_data)
                assert response.status_code == 200
                data = response.json()

                assert "insights" in data
                assert len(data["insights"]) > 0

    def test_metrics_endpoint(self, client):
        """Test the metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()

        # Check all required metrics
        required_metrics = [
            "gpu_utilization",
            "memory_usage",
            "active_tasks",
            "ai_tasks_processed",
            "total_devices",
            "available_devices",
        ]

        for metric in required_metrics:
            assert metric in data

        # Verify metric types
        assert isinstance(data["gpu_utilization"], (int, float))
        assert isinstance(data["memory_usage"], (int, float))
        assert isinstance(data["active_tasks"], int)
        assert isinstance(data["ai_tasks_processed"], int)
        assert isinstance(data["total_devices"], int)
        assert isinstance(data["available_devices"], int)

    def test_ai_task_with_invalid_workflow(self, client):
        """Test AI task with invalid workflow type."""
        ai_task_data = {"workflow_type": "invalid_workflow", "prompt": "Test prompt"}

        response = client.post("/ai/tasks", json=ai_task_data)
        # Should either return 422 for validation error or 200 with error in response
        assert response.status_code in [200, 422]

        if response.status_code == 200:
            data = response.json()
            assert "error" in data
