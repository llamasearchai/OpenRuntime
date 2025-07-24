"""
Test error handling and edge cases for OpenRuntime.
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from openruntime.core.api import app
from openruntime.core.models import TaskRequest, GPUDevice


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_task_with_invalid_operation(self, client):
        """Test task execution with invalid operation."""
        task_data = {
            "operation": "invalid_operation",
            "data": {}
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "failed"
        assert "error" in data
        assert "Unknown operation" in data["error"]
    
    def test_task_with_missing_parameters(self, client):
        """Test task execution with missing required parameters."""
        # Missing operation field
        task_data = {
            "parameters": {"size": 100}
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 422  # Validation error
    
    def test_task_with_invalid_device_preference(self, client):
        """Test task with invalid device preference."""
        task_data = {
            "operation": "compute",
            "data": {"size": 100},
            "device_preference": "nonexistent_device"
        }
        
        response = client.post("/tasks", json=task_data)
        # Invalid device preference should cause validation error
        assert response.status_code == 422
    
    def test_concurrent_task_limit(self, client):
        """Test behavior when hitting concurrent task limits."""
        # Submit many tasks simultaneously
        tasks = []
        for i in range(20):
            task_data = {
                "operation": "compute",
                "parameters": {"size": 1000, "iterations": 100}
            }
            tasks.append(task_data)
        
        # Submit all tasks
        responses = []
        for task in tasks:
            response = client.post("/tasks", json=task)
            responses.append(response)
        
        # All should be accepted (queued if necessary)
        for response in responses:
            assert response.status_code == 200
    
    def test_large_task_parameters(self, client):
        """Test task with very large parameters."""
        task_data = {
            "operation": "compute",
            "parameters": {
                "size": 1000000,  # Very large size
                "data": "x" * 10000  # Large string parameter
            }
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 200
    
    def test_health_check_when_unhealthy(self, client):
        """Test health check when service is unhealthy."""
        # Mock runtime manager to be None
        with patch.object(app.state, 'runtime_manager', None):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "error" in data
    
    def test_metrics_with_no_data(self, client):
        """Test metrics endpoint when no tasks have been run."""
        # Fresh start should still return valid metrics
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        
        # Should have default/zero values
        assert data["gpu_utilization"] >= 0
        assert data["memory_usage"] >= 0
        assert data["active_tasks"] >= 0
        assert data["ai_tasks_processed"] >= 0
    
    def test_ai_task_timeout(self, client):
        """Test AI task with timeout."""
        ai_task_data = {
            "workflow_type": "system_analysis",
            "prompt": "Analyze " + "complex " * 1000,  # Very long prompt
            "timeout": 1  # 1 second timeout
        }
        
        response = client.post("/ai/tasks", json=ai_task_data)
        assert response.status_code == 200
        # Should either complete or timeout gracefully
    
    def test_websocket_connection_error(self):
        """Test WebSocket behavior during connection errors."""
        client = TestClient(app)
        
        # Test invalid WebSocket endpoint
        with pytest.raises(Exception):
            with client.websocket_connect("/invalid_ws"):
                pass
    
    def test_device_registration_race_condition(self, client):
        """Test device registration under race conditions."""
        # Simulate multiple device registrations at once
        import threading
        
        results = []
        
        def get_devices():
            response = client.get("/devices")
            results.append(response.status_code)
        
        threads = []
        for _ in range(10):
            t = threading.Thread(target=get_devices)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    def test_malformed_json_request(self, client):
        """Test API behavior with malformed JSON."""
        # Send invalid JSON
        response = client.post(
            "/tasks",
            data='{"operation": "compute", invalid json}',
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code in [400, 422]  # Bad request or validation error
    
    def test_extremely_long_running_task(self, client):
        """Test handling of tasks that run for a very long time."""
        task_data = {
            "operation": "compute",
            "parameters": {
                "size": 10000,
                "iterations": 1000000  # Would take very long
            }
        }
        
        response = client.post("/tasks", json=task_data)
        assert response.status_code == 200
        
        # Task should be accepted but might be interrupted/limited
        data = response.json()
        assert "task_id" in data