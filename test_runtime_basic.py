#!/usr/bin/env python3
"""
Basic test of OpenRuntime v2
"""

import asyncio
from openruntime.runtime_engine import RuntimeEngine, RuntimeConfig, RuntimeBackend, TaskType


async def test_runtime():
    """Test basic runtime functionality"""
    print("Testing OpenRuntime v2...")
    
    # Create configuration
    config = RuntimeConfig(
        backend=RuntimeBackend.CPU,
        enable_monitoring=False,
        enable_caching=False
    )
    
    # Initialize engine
    engine = RuntimeEngine(config)
    await engine.initialize()
    print("Runtime initialized successfully")
    
    # Get status
    status = await engine.get_status()
    print(f"Status: {status}")
    
    # Test CPU inference
    task_id = await engine.submit_task(
        TaskType.INFERENCE,
        {
            "inputs": [1, 2, 3, 4, 5],
            "operation": "mean"
        },
        RuntimeBackend.CPU
    )
    print(f"Submitted task: {task_id}")
    
    # Wait for completion
    await asyncio.sleep(1)
    
    # Check metrics
    print(f"Metrics: {engine.metrics}")
    
    # Shutdown
    await engine.shutdown()
    print("Runtime shutdown complete")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    asyncio.run(test_runtime())