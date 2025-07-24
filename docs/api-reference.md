# OpenRuntime API Reference

Complete reference for the OpenRuntime REST API.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, OpenRuntime API does not require authentication for local deployments. For production deployments, see [Security Configuration](deployment.md#security).

## Common Response Format

All API responses follow this format:

```json
{
  "status": "success|error",
  "data": { ... },
  "error": "error message if applicable",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### System Endpoints

#### GET /

Get basic service information.

**Response:**
```json
{
  "name": "OpenRuntime",
  "version": "2.0.0",
  "status": "running"
}
```

#### GET /health

Check service health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "service": "OpenRuntime",
  "version": "2.0.0",
  "devices": 1,
  "metrics": {
    "gpu_utilization": 45.2,
    "memory_usage": 32.1,
    "active_tasks": 2,
    "ai_tasks_processed": 150
  }
}
```

#### GET /metrics

Get current system metrics.

**Response:**
```json
{
  "gpu_utilization": 45.2,
  "memory_usage": 32.1,
  "active_tasks": 2,
  "ai_tasks_processed": 150,
  "uptime": "2h 15m",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Device Management

#### GET /devices

List all available compute devices.

**Response:**
```json
{
  "devices": [
    {
      "id": "device_0",
      "name": "Apple M1 Max",
      "type": "gpu",
      "status": "available",
      "capabilities": ["metal", "mlx", "unified_memory"],
      "memory_total": 34359738368,
      "memory_available": 25769803776
    }
  ]
}
```

### Task Execution

#### POST /tasks

Submit a compute task for execution.

**Request Body:**
```json
{
  "operation": "matrix_multiply",
  "parameters": {
    "size": 1000,
    "iterations": 10
  },
  "device_preference": "auto",
  "priority": 1
}
```

**Parameters:**
- `operation` (string, required): Type of operation to perform
  - `compute`: General compute task
  - `matrix_multiply`: Matrix multiplication
  - `vector_operations`: Vector computations
  - `memory_bandwidth`: Memory bandwidth test
  - `mlx_operation`: MLX-specific operations
- `parameters` (object, required): Operation-specific parameters
- `device_preference` (string, optional): Device selection preference
  - `auto`: Automatic selection (default)
  - `gpu`: Prefer GPU
  - `cpu`: Prefer CPU
  - Device ID: Specific device
- `priority` (integer, optional): Task priority (1-10, default: 5)

**Response:**
```json
{
  "task_id": "task_1234567890",
  "status": "completed",
  "result": {
    "execution_time": 0.523,
    "device_used": "device_0",
    "output": { ... }
  },
  "execution_time": 0.523,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /tasks/enhanced

Get enhanced task information with detailed metrics.

**Query Parameters:**
- `limit` (integer, optional): Number of tasks to return (default: 100)
- `offset` (integer, optional): Offset for pagination

**Response:**
```json
{
  "tasks": [
    {
      "task_id": "task_1234567890",
      "operation": "matrix_multiply",
      "status": "completed",
      "performance_metrics": {
        "execution_time_ms": 523,
        "memory_used_mb": 256.5,
        "gpu_utilization": 87.3
      },
      "timestamp": "2024-01-01T00:00:00Z"
    }
  ],
  "total_count": 1250,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### AI Operations

#### GET /ai/agents

List available AI agents and their capabilities.

**Response:**
```json
{
  "agents": [
    {
      "role": "DEVELOPER",
      "name": "Code Assistant",
      "description": "Assists with code generation and refactoring",
      "capabilities": ["code_generation", "refactoring", "debugging"]
    },
    {
      "role": "ANALYST",
      "name": "System Analyst",
      "description": "Analyzes system performance and provides insights",
      "capabilities": ["performance_analysis", "optimization", "reporting"]
    }
  ]
}
```

#### POST /ai/tasks

Execute an AI-powered task.

**Request Body:**
```json
{
  "workflow_type": "code_generation",
  "prompt": "Create a Python function to calculate fibonacci numbers",
  "parameters": {
    "language": "python",
    "style": "functional"
  },
  "timeout": 30
}
```

**Parameters:**
- `workflow_type` (string, required): Type of AI workflow
  - `code_generation`: Generate code
  - `system_analysis`: Analyze system state
  - `optimization`: Optimize code or configuration
  - `model_inference`: Run ML model inference
- `prompt` (string, required): Input prompt for the AI
- `parameters` (object, optional): Additional workflow parameters
- `timeout` (integer, optional): Timeout in seconds (default: 300)

**Response:**
```json
{
  "task_id": "ai_task_123",
  "status": "completed",
  "result": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "model_used": "gpt-4",
  "tokens_used": 125,
  "execution_time": 2.3
}
```

#### POST /ai/insights

Generate AI insights for system performance.

**Request Body:**
```json
{
  "metric_type": "performance",
  "timeframe": "1h"
}
```

**Parameters:**
- `metric_type` (string, required): Type of metrics to analyze
  - `performance`: Overall performance metrics
  - `memory`: Memory usage patterns
  - `efficiency`: Resource efficiency
  - `errors`: Error patterns
- `timeframe` (string, required): Time window for analysis
  - `5m`, `15m`, `1h`, `6h`, `24h`, `7d`

**Response:**
```json
{
  "insights": "Based on the last hour of performance data:\n\n1. GPU utilization averaging 75% indicates healthy workload\n2. Memory usage spike at 14:30 suggests need for optimization\n3. Consider implementing batch processing for similar tasks",
  "metrics": {
    "gpu_utilization": 75.5,
    "memory_usage": 45.2,
    "active_tasks": 3,
    "ai_tasks_processed": 12
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### WebSocket Endpoints

#### WS /ws

Real-time metrics and task updates via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

**Message Types:**

1. **Connected Message** (received on connection):
```json
{
  "type": "connected",
  "timestamp": "2024-01-01T00:00:00Z",
  "message": "Connected to OpenRuntime WebSocket"
}
```

2. **Heartbeat Message** (every 5 seconds):
```json
{
  "type": "heartbeat",
  "timestamp": "2024-01-01T00:00:00Z",
  "metrics": {
    "gpu_utilization": 45.2,
    "memory_usage": 32.1,
    "active_tasks": 2
  }
}
```

3. **Task Update Message**:
```json
{
  "type": "task_update",
  "data": {
    "task_id": "task_123",
    "status": "completed",
    "execution_time": 1.23
  }
}
```

## Error Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid request format |
| 422 | Validation Error - Invalid parameters |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

## Rate Limiting

Local deployments have no rate limiting. Production deployments may implement:
- 1000 requests per minute per IP
- 100 concurrent connections per IP
- 10 WebSocket connections per IP

## Examples

### Python

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Get service status
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# Submit a task
task_data = {
    "operation": "matrix_multiply",
    "parameters": {"size": 1000}
}
response = requests.post(f"{BASE_URL}/tasks", json=task_data)
result = response.json()
print(f"Task completed in {result['execution_time']}s")

# Execute AI task
ai_task = {
    "workflow_type": "code_generation",
    "prompt": "Create a sorting algorithm"
}
response = requests.post(f"{BASE_URL}/ai/tasks", json=ai_task)
print(response.json()['result'])
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Submit task
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"operation": "matrix_multiply", "parameters": {"size": 1000}}'

# Get AI agents
curl http://localhost:8000/ai/agents

# Generate insights
curl -X POST http://localhost:8000/ai/insights \
  -H "Content-Type: application/json" \
  -d '{"metric_type": "performance", "timeframe": "1h"}'
```

### JavaScript/Node.js

```javascript
// Using fetch API
async function submitTask() {
  const response = await fetch('http://localhost:8000/tasks', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      operation: 'matrix_multiply',
      parameters: { size: 1000 }
    })
  });
  
  const result = await response.json();
  console.log(`Task completed in ${result.execution_time}s`);
}

// WebSocket connection
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

## SDK Support

Official SDKs are available for:
- Python: `pip install openruntime-sdk`
- JavaScript/TypeScript: `npm install @openruntime/sdk`
- Go: `go get github.com/openruntime/sdk-go`

See [SDK Documentation](sdks/index.md) for detailed usage.