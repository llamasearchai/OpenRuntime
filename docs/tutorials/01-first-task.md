# Tutorial 1: Your First Task

Learn the basics of using OpenRuntime to execute GPU-accelerated tasks.

## Prerequisites

- OpenRuntime installed and running
- Basic knowledge of Python or command-line tools
- A terminal or Python environment

## Learning Objectives

By the end of this tutorial, you will:
- Understand how to start the OpenRuntime service
- Submit your first compute task
- Monitor task execution
- Interpret the results
- Handle errors gracefully

## Step 1: Start the OpenRuntime Service

First, let's start the OpenRuntime service:

```bash
# Start the service
python -m openruntime.main

# Or use the CLI
python openruntime_cli.py server start
```

You should see output like:
```
OpenRuntime Enhanced starting up...
Version: 2.1.0
Author: Nik Jois <nikjois@llamasearch.ai>
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 2: Verify the Service is Running

Open a new terminal and check the service status:

```bash
# Using CLI
python openruntime_cli.py status

# Using curl
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "OpenRuntime",
  "version": "2.0.0",
  "devices": 1,
  "metrics": {
    "gpu_utilization": 0,
    "memory_usage": 0,
    "active_tasks": 0
  }
}
```

## Step 3: Submit Your First Task

Let's submit a simple matrix multiplication task.

### Using the CLI

```bash
python openruntime_cli.py run --operation compute --size 1000
```

### Using Python

```python
import requests
import json

# Define the task
task = {
    "operation": "matrix_multiply",
    "parameters": {
        "size": 1000
    }
}

# Submit the task
response = requests.post('http://localhost:8000/tasks', json=task)
result = response.json()

# Print the result
print(json.dumps(result, indent=2))
```

### Using curl

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "operation": "matrix_multiply",
    "parameters": {"size": 1000}
  }'
```

## Step 4: Understanding the Response

A successful response looks like:

```json
{
  "task_id": "task_1704123456.789",
  "status": "completed",
  "result": {
    "execution_time": 0.523,
    "device_used": "device_0",
    "gflops": 3.82,
    "matrix_shape": [1000, 1000]
  },
  "execution_time": 0.523,
  "device_used": "device_0",
  "timestamp": "2024-01-01T12:34:56.789Z"
}
```

Key fields:
- **task_id**: Unique identifier for the task
- **status**: Task status (pending, running, completed, failed)
- **execution_time**: Time taken in seconds
- **device_used**: Which device executed the task
- **result**: Task-specific output data

## Step 5: Try Different Operations

OpenRuntime supports various operations:

### Vector Operations

```python
task = {
    "operation": "vector_operations",
    "parameters": {
        "size": 100000,
        "operation_type": "dot_product"
    }
}
```

### Memory Bandwidth Test

```python
task = {
    "operation": "memory_bandwidth",
    "parameters": {
        "size": 100000000,  # 100MB
        "iterations": 10
    }
}
```

### Custom Compute

```python
task = {
    "operation": "compute",
    "parameters": {
        "kernel": "custom_kernel",
        "data_size": 1000,
        "threads": 256
    }
}
```

## Step 6: Monitor Real-time Metrics

Connect to the WebSocket endpoint for real-time updates:

```python
import asyncio
import websockets
import json

async def monitor_metrics():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to OpenRuntime WebSocket")
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "heartbeat":
                metrics = data["metrics"]
                print(f"GPU: {metrics['gpu_utilization']}% | "
                      f"Memory: {metrics['memory_usage']}% | "
                      f"Active: {metrics['active_tasks']}")
            elif data["type"] == "task_update":
                print(f"Task Update: {data['data']}")

# Run the monitor
asyncio.run(monitor_metrics())
```

## Step 7: Handle Errors

Let's see how errors are handled:

```python
# Submit an invalid task
invalid_task = {
    "operation": "invalid_operation",
    "parameters": {}
}

response = requests.post('http://localhost:8000/tasks', json=invalid_task)
result = response.json()

if result["status"] == "failed":
    print(f"Error: {result['error']}")
```

## Step 8: Specify Device Preferences

You can specify which device to use:

```python
# Prefer GPU
task = {
    "operation": "matrix_multiply",
    "parameters": {"size": 2000},
    "device_preference": "gpu"
}

# Use specific device
task = {
    "operation": "matrix_multiply", 
    "parameters": {"size": 2000},
    "device_preference": "device_0"
}

# Auto-select (default)
task = {
    "operation": "matrix_multiply",
    "parameters": {"size": 2000},
    "device_preference": "auto"
}
```

## Complete Example

Here's a complete Python script that demonstrates all concepts:

```python
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def check_health():
    """Check if service is healthy."""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()["status"] == "healthy"

def submit_task(operation, parameters, device_preference="auto"):
    """Submit a task and return the result."""
    task = {
        "operation": operation,
        "parameters": parameters,
        "device_preference": device_preference
    }
    
    response = requests.post(f"{BASE_URL}/tasks", json=task)
    return response.json()

def main():
    # Check service health
    if not check_health():
        print("Service is not healthy!")
        return
    
    print("OpenRuntime is healthy and ready!")
    
    # List available devices
    devices = requests.get(f"{BASE_URL}/devices").json()
    print(f"\nAvailable devices: {len(devices['devices'])}")
    for device in devices['devices']:
        print(f"  - {device['id']}: {device['name']} ({device['type']})")
    
    # Run benchmarks
    sizes = [100, 500, 1000, 2000]
    results = []
    
    print("\nRunning matrix multiplication benchmarks:")
    for size in sizes:
        print(f"  Size {size}x{size}...", end="", flush=True)
        
        result = submit_task(
            "matrix_multiply",
            {"size": size}
        )
        
        if result["status"] == "completed":
            exec_time = result["execution_time"]
            gflops = result["result"].get("gflops", 0)
            results.append((size, exec_time, gflops))
            print(f" {exec_time:.3f}s ({gflops:.2f} GFLOPS)")
        else:
            print(f" FAILED: {result.get('error', 'Unknown error')}")
    
    # Display summary
    print("\nBenchmark Summary:")
    print("Size  | Time (s) | GFLOPS")
    print("------|----------|-------")
    for size, time, gflops in results:
        print(f"{size:5d} | {time:8.3f} | {gflops:6.2f}")

if __name__ == "__main__":
    main()
```

## Exercises

1. **Modify Task Parameters**: Try different matrix sizes and observe how execution time changes.

2. **Compare Devices**: If you have multiple devices, compare performance between them.

3. **Error Handling**: Intentionally cause errors (invalid operations, huge sizes) and handle them gracefully.

4. **Batch Processing**: Submit multiple tasks in parallel and track their completion.

5. **Performance Analysis**: Plot execution time vs. matrix size to understand scaling behavior.

## Common Issues and Solutions

### Service Connection Error

**Problem**: Cannot connect to the service
```
Connection refused to http://localhost:8000
```

**Solution**: Ensure the service is running:
```bash
ps aux | grep openruntime
python openruntime_cli.py server start
```

### Task Timeout

**Problem**: Task takes too long and times out

**Solution**: Increase timeout or reduce task size:
```python
task = {
    "operation": "matrix_multiply",
    "parameters": {"size": 5000},
    "timeout": 600  # 10 minutes
}
```

### Out of Memory

**Problem**: Task fails with memory error

**Solution**: Check available memory and reduce task size:
```bash
python openruntime_cli.py status --detailed
```

## Next Steps

Congratulations! You've successfully:
- Started the OpenRuntime service
- Submitted and monitored tasks
- Handled different operations
- Understood error handling

Next, try:
1. [Tutorial 2: Working with AI Agents](02-ai-agents.md)
2. [Tutorial 3: Real-time Monitoring](03-realtime-monitoring.md)
3. Explore the [API Reference](../api-reference.md)

## Further Reading

- [OpenRuntime Architecture](../architecture.md)
- [Performance Tuning Guide](06-performance-tuning.md)
- [CLI Command Reference](../cli-guide.md)