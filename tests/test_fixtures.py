"""
Test fixtures and utilities for OpenRuntime tests.
"""

from unittest.mock import MagicMock, AsyncMock
from openruntime.core.managers import GPURuntimeManager, AIAgentManager
from openruntime.core.models import GPUDevice, DeviceType, AgentRole


def create_mock_runtime_manager():
    """Create a mock runtime manager for testing."""
    manager = MagicMock(spec=GPURuntimeManager)
    
    # Mock devices
    mock_device = GPUDevice(
        id="test_device_0",
        name="Test GPU",
        type=DeviceType.METAL,
        memory_total=16 * 1024 * 1024 * 1024,
        memory_available=12 * 1024 * 1024 * 1024,
        compute_units=32,
        is_available=True,
        capabilities=["compute", "matrix_multiply"],
        driver_version="Test 1.0",
        compute_capability="Test"
    )
    
    manager.devices = {"test_device_0": mock_device}
    
    # Mock AI manager
    mock_ai_manager = MagicMock(spec=AIAgentManager)
    mock_ai_manager.agents = {
        "DEVELOPER": {
            "name": "Code Assistant",
            "description": "Assists with code generation",
            "role": AgentRole.CODE_GENERATOR,
            "capabilities": ["code_generation"]
        },
        "ANALYST": {
            "name": "System Analyst", 
            "description": "Analyzes system performance",
            "role": AgentRole.SYSTEM_ANALYST,
            "capabilities": ["performance_analysis"]
        },
        "OPTIMIZER": {
            "name": "Performance Optimizer",
            "description": "Optimizes system performance",
            "role": AgentRole.PERFORMANCE_OPTIMIZER,
            "capabilities": ["optimization"]
        },
        "DEBUGGER": {
            "name": "Code Debugger",
            "description": "Helps debug code issues",
            "role": AgentRole.CODE_GENERATOR,
            "capabilities": ["debugging"]
        }
    }
    
    # Mock execute_ai_task to return a valid response
    async def mock_execute_ai_task(request):
        return {
            "result": "Mock AI response for testing",
            "tokens_used": 100
        }
    
    mock_ai_manager.execute_ai_task = AsyncMock(side_effect=mock_execute_ai_task)
    manager.ai_manager = mock_ai_manager
    
    # Mock metrics
    def mock_get_metrics():
        return {
            "gpu_utilization": 50.0,
            "memory_usage": 30.0,
            "active_tasks": 2,
            "ai_tasks_processed": 10,
            "total_devices": 1,
            "available_devices": 1
        }
    
    manager.get_metrics = MagicMock(return_value=mock_get_metrics())
    
    # Mock task execution
    async def mock_execute_task(task):
        from openruntime.core.models import TaskResponse, TaskStatus
        import time
        
        if task.operation == "invalid_operation":
            return TaskResponse(
                task_id=f"task_{time.time()}",
                status=TaskStatus.FAILED,
                error="Unknown operation: invalid_operation",
                execution_time=0.001,
                device_used="test_device_0"
            )
        
        return TaskResponse(
            task_id=f"task_{time.time()}",
            status=TaskStatus.COMPLETED,
            result={"success": True, "data": "test"},
            execution_time=0.1,
            device_used="test_device_0"
        )
    
    manager.execute_task = AsyncMock(side_effect=mock_execute_task)
    
    # Mock other attributes
    manager.active_tasks = {}
    manager.monitoring_data = {}
    manager.executor = MagicMock()
    manager.executor.shutdown = MagicMock()
    
    return manager