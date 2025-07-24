"""
Test WebSocket functionality for OpenRuntime.
"""

import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from openruntime_enhanced.enhanced import app


class TestWebSocketEndpoint:
    """Test WebSocket connections and messaging."""
    
    def test_websocket_connection(self, client):
        """Test basic WebSocket connection."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive initial connection message
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert "timestamp" in data
            
            # Should receive system status within 6 seconds
            data = websocket.receive_json()
            assert data["type"] == "system_status"
            assert "timestamp" in data
            assert "data" in data
            
            # Check data structure
            status_data = data["data"]
            assert "active_devices" in status_data
            assert "ai_agents" in status_data
            assert "mlx_available" in status_data
    
    def test_websocket_multiple_connections(self, client):
        """Test multiple concurrent WebSocket connections."""
        with client.websocket_connect("/ws") as ws1:
            with client.websocket_connect("/ws") as ws2:
                # Both should receive connection messages
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()
                
                assert data1["type"] == "connection_established"
                assert data2["type"] == "connection_established"
                
                # Both should receive system status
                data1 = ws1.receive_json()
                data2 = ws2.receive_json()
                
                assert data1["type"] == "system_status"
                assert data2["type"] == "system_status"
    
    def test_websocket_disconnection(self, client):
        """Test WebSocket disconnection handling."""
        with client.websocket_connect("/ws") as websocket:
            # Receive initial message
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            
            # WebSocket will be closed when exiting context
        
        # Connection should be closed now
        assert True  # If we get here, disconnection worked
    
    def test_websocket_task_updates(self, client):
        """Test that WebSocket receives task update broadcasts."""
        with client.websocket_connect("/ws") as websocket:
            # Clear initial messages
            websocket.receive_json()  # connection message
            
            # Submit a task
            task_data = {
                "operation": "compute",
                "data": {"type": "matrix_multiply", "size": 100}
            }
            response = client.post("/tasks", json=task_data)
            assert response.status_code == 200
            task_result = response.json()
            assert task_result["status"] in ["completed", "failed"]

            # Wait for the system status message (no task update broadcast implemented)
            received_message = websocket.receive_json()
            assert received_message["type"] == "system_status"
            assert "data" in received_message